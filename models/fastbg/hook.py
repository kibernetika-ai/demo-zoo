import logging

import numpy as np
import cv2
from ml_serving.utils.helpers import load_image

LOG = logging.getLogger(__name__)


def init_hook(**params):
    LOG.info('Loaded.')


def process(inputs, ct_x, **kwargs):
    original_image, is_video = load_image(inputs, 'inputs')
    if original_image is None:
        raise RuntimeError('Missing "inputs" key in inputs. Provide an image in "inputs" key')

    if original_image.shape[2] > 3:
        original_image = original_image[:, :, 0:3]

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
    w = float(original_image.shape[1])
    h = float(original_image.shape[0])
    if w > h:
        if w > 1024:
            ratio = w / 1024.0
    else:
        if h > 1024:
            ratio = h / 1024.0

    if ratio > 1:
        image = cv2.resize(original_image, (int(w / ratio), int(h / ratio)))
    else:
        image = original_image

    serv_image = cv2.resize(image, (160, 160)).astype(np.float32) / 255.0
    result = ct_x.drivers[0].predict({'image': np.expand_dims(serv_image, axis=0)})
    mask = result['output'][0]
    mask[mask < 0.5] = 0
    if mask.shape != image.shape:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    mask = cv2.GaussianBlur(mask, (21, 21), 11)
    if not is_video:
        mask = (mask * 255).astype(np.uint8)
        image = image[:, :, ::-1].astype(np.uint8)
        image = np.concatenate([image, mask], axis=2)
    else:
        image = image.astype(np.float32) * mask
        image = image.astype(np.uint8)
    return _return(image)
