import logging
import cv2
import numpy as np

LOG = logging.getLogger(__name__)


def init_hook(**kwargs):
    LOG.info('init: {}'.format(kwargs))


def process(inputs, ctx, **kwargs):
    image = inputs.get('input')
    if image is None:
        raise RuntimeError('Missing input key in inputs. Provide an image in "input" key')
    videoInput = True
    if len(image.shape) < 3:
        ##Encoded image
        image = cv2.imdecode(np.frombuffer(image[0], np.uint8), cv2.IMREAD_COLOR)
        image = image[:, :, ::-1]
        videoInput = False

    x = int(image.shape[1] / 2)
    y = int(image.shape[0] / 2)
    cv2.putText(image, 'Powered by Kibernetika.AI', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255),
                        thickness=2, lineType=cv2.LINE_AA)
    if videoInput:
        return {'output': image}
    else:
        _, buf = cv2.imencode('.png', image[:, :, ::-1])
        image = np.array(buf).tostring()
        return {'output': image}
