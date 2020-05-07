import logging
import cv2
import numpy as np
from ml_serving.utils.helpers import load_image

LOG = logging.getLogger(__name__)


def init_hook(**kwargs):
    LOG.info('init: {}'.format(kwargs))


def process(inputs, ctx, **kwargs):
    original_image, is_video = load_image(inputs, 'inputs')
    if is_video:
        return {'outputs': original_image,'status': cv2.resize(original_image, (original_image.shape[1]//2, original_image.shape[0]//2))}
    else:
        _, buf = cv2.imencode('.png', original_image[:, :, ::-1])
        image = np.array(buf).tostring()
        return {'outputs': image}
