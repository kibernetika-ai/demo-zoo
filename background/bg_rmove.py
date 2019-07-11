import numpy as np
import logging
import cv2
from ml_serving.utils import helpers

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

resolution = 160

back = np.zeros((1,1,3),np.uint8)

def init_hook(**kwargs):
    global back
    back = cv2.imread('./newback.jpg')[:, :, ::-1]

def preprocess(inputs, ctx, **kwargs):
    image, is_video = helpers.load_image(inputs, 'image')
    ctx.w = image.shape[1]
    ctx.h = image.shape[0]
    ctx.is_video = is_video
    if is_video:
        ctx.input = image
    else:
        if ctx.w > ctx.h:
            if ctx.w > 1024:
                ctx.h = int(float(ctx.h) * 1024.0 / float(ctx.w))
                ctx.w = 1024
        else:
            if ctx.h > 1024:
                ctx.w = int(float(ctx.w) * 1024.0 / float(ctx.h))
                ctx.h = 1024
        ctx.input = cv2.resize(image, (ctx.w, ctx.h))
    ctx.input = np.asarray(ctx.input, np.float32)
    image = cv2.resize(image, (160, 160))
    input = np.asarray(image, np.float32)
    input = input / 255.0
    return {
        'image': np.expand_dims(input,axis=0)
    }


def postprocess(outputs, ctx, **kwargs):
    global back
    mask = outputs['output']
    mask = mask[0]
    if mask.shape[0] != ctx.h or mask.shape[1] != ctx.w:
        mask = cv2.resize(mask, (ctx.w, ctx.h))
    if len(mask.shape)<3:
        mask = np.expand_dims(mask,2)
    lback = back
    if mask.shape[0]!=lback.shape[0] or mask.shape[1]!=lback.shape[1]:
        lback = cv2.resize(back, (ctx.w, ctx.h))
        back = lback
    output = ctx.input * mask+lback*(1-mask)
    output = output.astype(np.uint8)
    if not ctx.is_video:
        output = output[:, :, ::-1]
        image_bytes = cv2.imencode('.jpg', output)[1].tostring()
    else:
        image_bytes = output

    return {'output': image_bytes}
