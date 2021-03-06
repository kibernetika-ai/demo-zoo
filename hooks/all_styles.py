import io
import logging

import numpy as np
import torch
import glob
import os
import cv2
import mstyles.Transformer as StylesTrans
import cartoon.Transformer as CartoonTrans
from torch.autograd import Variable
from ml_serving.utils import helpers
from young.Transformer import Pipe as YoungHook

LOG = logging.getLogger(__name__)


class ArtisticStyles:
    def __init__(self, **params):
        self.shornames = {'mosaic': 'mosaic', 'picasso': 'picasso_selfport1907', 'robert': 'robert', 'candy': 'candy',
                          'scream': 'the_scream', 'composition': 'composition_vii', 'shipwreck': 'shipwreck',
                          'rain': 'rain_princess', 'mosaic1': 'mosaic_ducks_massimo', 'escher': 'escher_sphere',
                          'udnie': 'udnie', 'wave': 'wave', 'woman': 'woman-with-hat-matisse', 'la_muse': 'la_muse',
                          'seated': 'seated-nude', 'pencil': 'pencil', 'strip': 'strip', 'feathers': 'feathers',
                          'starry': 'starry_night', 'stars2': 'stars2', 'frida': 'frida_kahlo'}
        self.max_size = int(params.get('max_size', 1024))
        self.cuda = torch.cuda.is_available()
        model_path = params.get('style_model_path', None)
        if model_path is None:
            self.model = None
            return
        state = torch.load(model_path)
        self.model = StylesTrans.Net(ngf=128)
        new_state = {}
        for k, v in state.items():
            ks = k.split('.')
            if ks[-1] == 'running_mean' or ks[-1] == 'running_var':
                continue
            new_state[k] = v
        self.model.load_state_dict(new_state)
        if self.cuda:
            self.model.cuda()
        else:
            self.model.float()
        self.styles = {}
        for m in glob.glob(params.get('styles_samples_path', '/model/21styles') + '/*.jpg'):
            img = cv2.imread(m)
            style = pytorch_load_image(img, size=self.max_size).unsqueeze(0)
            if self.cuda:
                style = style.cuda()
            f = os.path.basename(m)
            f = f.split('.')[0]
            LOG.info('Add artistic style: {}'.format(f))
            self.styles[f] = style

    def process(self, image, style_name, inputs):
        if self.model is None:
            return image
        w = image.shape[1]
        h = image.shape[0]
        image = pytorch_load_image(image, size=self.max_size, keep_asp=True).unsqueeze(0)
        if self.cuda:
            image = image.cuda()
        style_name = self.shornames.get(style_name, style_name)
        style = self.styles[style_name]
        style_v = Variable(style)
        image = Variable(image)
        self.model.setTarget(style_v)
        image = self.model(image)
        image = pytorch_image_numpy(image.data[0], self.cuda)
        image = cv2.resize(image, (w, h))
        return image


class CartoonStyles:
    def __init__(self, **params):
        self.max_size = int(params.get('max_size', 1024))
        self.cuda = torch.cuda.is_available()
        model_path = params.get('cartoon_model_path', None)
        if model_path is None:
            self.models = None
            return
        self.models = {}
        for m in glob.glob(model_path + '/*.pth'):
            f = os.path.basename(m)
            LOG.info('Loading cartoon model {} {}'.format(f, m))
            model = CartoonTrans.Transformer()
            model.load_state_dict(torch.load(m))
            model.eval()
            if self.cuda:
                model.cuda()
            else:
                model.float()
            self.models[f.split('_')[0]] = model

    def process(self, image, style_name, inputs):
        if self.models is None:
            return image
        w = image.shape[1]
        h = image.shape[0]
        model = self.models[style_name]
        image = pytorch_load_image(image, size=self.max_size, keep_asp=True).unsqueeze(0)
        image = -1 + 2 * image / 255.0
        image = Variable(image, volatile=True)
        if self.cuda:
            image = image.cuda()
        image = model(image)[0]
        image = image.data.cpu().float().numpy()
        image = (image * 0.5 + 0.5) * 255
        image = np.transpose(image, (1, 2, 0)).astype(np.uint8)
        image = cv2.resize(image[:, :, ::-1], (w, h))
        return image


class YoungModel:
    def __init__(self, **params):
        model_path = params.get('beauty_model_path', None)
        if model_path is None:
            self.model = None
            return
        self.model = YoungHook({}, **params)

    def process(self, inputs, ctx, **kwargs):
        if self.model is None:
            img, is_video = helpers.load_image(inputs, 'image', rgb=False)
            img = img[:, :, ::-1]
            if not is_video:
                img = cv2.imencode('.jpg', img)[1].tostring()
            return {'output': img}
        return self.model.process(inputs, ctx, **kwargs)


def init_hook(**params):
    LOG.info('Loaded. {}'.format(params))
    default_model = params.get('default_model', 'young')
    return {'default_model': default_model, 'artistic': ArtisticStyles(**params), 'cartoon': CartoonStyles(**params),
            'young': YoungModel(**params)}


def process(inputs, ctx, **kwargs):
    style_name = helpers.get_param(inputs, 'style', None)
    default_model = ctx.global_ctx['default_model']
    if style_name == 'young' or default_model == 'young':
        return ctx.global_ctx['young'].process(inputs, ctx, **kwargs)
    img, is_video = helpers.load_image(inputs, 'image', rgb=False)
    if style_name is not None:
        p = style_name.split('_')
        model = default_model
        if len(p) > 1:
            model = p[0]
            style_name = '_'.join(p[1:])
        img = ctx.global_ctx[model].process(img, style_name, inputs)
    if not is_video:
        img = img[:, :, ::-1]
        img = cv2.imencode('.jpg', img)[1].tostring()
    return {'output': img}


def pytorch_load_image(img, size=None, scale=None, keep_asp=False):
    if size is not None:
        if keep_asp:
            size2 = int(size * 1.0 / img.shape[1] * img.shape[0])
            img = cv2.resize(img, (size, size2))
        else:
            img = cv2.resize(img, (size, size))

    elif scale is not None:
        img = cv2.resize(img, (int(img.shape[1] / scale), int(img.shape[0] / scale)))
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img


def pytorch_image_numpy(tensor, cuda=False):
    if cuda:
        img = tensor.clone().cpu().clamp(0, 255).numpy()
    else:
        img = tensor.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    return img
