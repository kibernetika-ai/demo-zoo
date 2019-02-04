import io
import logging

import numpy as np
from PIL import Image
import os
import glob
LOG = logging.getLogger(__name__)



max_size = 256
styles = {}

shornames = {'mosaic': 'mosaic', 'picasso': 'picasso_selfport1907', 'robert': 'robert', 'candy': 'candy', 'scream': 'the_scream', 'composition': 'composition_vii', 'shipwreck': 'shipwreck', 'rain': 'rain_princess', 'mosaic1': 'mosaic_ducks_massimo', 'escher': 'escher_sphere', 'udnie': 'udnie', 'wave': 'wave', 'woman': 'woman-with-hat-matisse', 'la_muse': 'la_muse', 'seated': 'seated-nude', 'pencil': 'pencil', 'strip': 'strip', 'feathers': 'feathers', 'starry': 'starry_night', 'stars2': 'stars2', 'frida': 'frida_kahlo'}


PARAMS = {
    'max_size': '256',
    'styles_path': './data/styles'
}

def init_hook(**params):
    LOG.info('Loaded. {}'.format(params))
    global PARAMS
    PARAMS.update(params)
    global max_size
    max_size = PARAMS.get('max_size', '256')
    max_size = int(max_size)
    LOG.info('Max size {}'.format(max_size))
    for m in glob.glob(PARAMS['styles_path'] + '/*.jpg'):
        img = Image.open(m)
        style = tensor_load_rgbimage(img,max_size)
        global styles
        f = os.path.basename(m)
        f = f.split('.')[0]
        LOG.info('Add style: {}'.format(f))
        styles[f] = style


def preprocess(inputs, ctx):
    content_image = Image.open(io.BytesIO(inputs['image'][0]))
    w = content_image.size[0]
    h = content_image.size[1]
    if w > h:
        if w > 1024:
            ratio = float(w) / 1024.0
            w = 1024
            h = float(h/ratio)
    else:
        if h > 1024:
            ratio = float(h) / 1024.0
            h = 1024
            w = float(w/ratio)
    h = int(h)
    w = int(w)
    ctx.original_w = w
    ctx.original_h = h

    style = inputs.get('style', ['candy'])[0].decode("utf-8")
    style = shornames.get(style,style)
    style = styles[style]

    content_image = tensor_load_rgbimage(content_image,max_size)

    return {'content_inputs': [content_image],
            'style_inputs': [style]}


def postprocess(outputs, ctx):
    image = outputs['output'][0]
    image_bytes = io.BytesIO()
    image = Image.fromarray(np.uint8(image*255))
    image = image.resize((int(ctx.original_w),int(ctx.original_h)),Image.BILINEAR)
    image.save(image_bytes, format='JPEG')
    return {'output': image_bytes.getvalue()}


def tensor_load_rgbimage(img,size):
    img = img.convert('RGB')
    img = img.resize((int(size), int(size)),Image.BILINEAR)
    return np.asarray(img,np.float32)/255.0

def tensor_load_rgbimage1(img, size=None, scale=None, keep_asp=False):
    img = img.convert('RGB')
    if size is not None:
        if keep_asp:
            size2 = int(size * 1.0 / img.size[0] * img.size[1])
            img = img.resize((size, size2), Image.ANTIALIAS)
        else:
            img = img.resize((size, size), Image.ANTIALIAS)

    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return np.asarray(img,np.float32)/255.0

