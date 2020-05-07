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
        return {'output': original_image}
    else:
        _, buf = cv2.imencode('.png', original_image[:, :, ::-1])
        image = np.array(buf).tostring()
        return {'output': image}
