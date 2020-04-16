import logging

import numpy as np
import cv2
from ml_serving.utils.helpers import get_param, load_image
import glob
import os

LOG = logging.getLogger(__name__)

backgrounds = {'None': None}
glob_background = None


def init_hook(**params):
    backgrounds_dir = params.get('backgrounds', None)
    global backgrounds
    if backgrounds_dir is not None:
        for f in glob.glob(backgrounds_dir + '/*.jpg'):
            name = os.path.basename(f)[:-4]
            LOG.info('Load: {}'.format(name))
            img = cv2.imread(f)
            backgrounds[name] = img[:, :, ::-1]
        back = params.get('background', None)
        if back is not None:
            global glob_background
            glob_background = backgrounds.get(back, None)
    LOG.info('Loaded.')

def process(inputs, ctx, **kwargs):
    image, is_video = load_image(inputs, 'inputs')
    if image is None:
        raise RuntimeError('Missing "inputs" key in inputs. Provide an image in "inputs" key')
    def _return(result):
        encoding = ''
        if not is_video:
            if result.shape[2] == 3:
                result = result[:, :, ::-1]
                result = cv2.imencode('.jpg', result)[1].tostring()
                encoding = 'jpeg'
            else:
                result = result
                result = cv2.imencode('.png', result)[1].tostring()
                encoding = 'png'
        return {'output': result, 'encoding': encoding}
    ratio = 1.0
    w = float(image.shape[1])
    h = float(image.shape[0])
    if w > h:
        if w > 1024:
            ratio = w / 1024.0
    else:
        if h > 1024:
            ratio = h / 1024.0

    if ratio > 1:
        image = cv2.resize(image, (int(w / ratio), int(h / ratio)))


    input = cv2.resize(image, (160, 160))
    input = np.asarray(input, np.float32) / 255.0
    outputs = ctx.drivers[0].predict({'image': np.expand_dims(input, axis=0)})
    mask = outputs['output'][0]
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    mask = cv2.GaussianBlur(mask, (21, 21), 11)
    mask = np.expand_dims(mask, 2)
    back_name = get_param(inputs, 'background', None)
    if back_name is not None:
        background = backgrounds.get(back_name)
    else:
        if glob_background is not None:
            background = glob_background
        else:
            background = backgrounds.get('None')

    image = image.astype(np.float32) * mask
    background = cv2.resize(background, (image.shape[1], image.shape[0]))
    background = background.astype(np.float32)
    background = background * (1 - mask)
    image = background + image
    image = image.astype(np.uint8)
    return _return(image)