import logging

import numpy as np
from scipy import ndimage
import cv2
from ml_serving.utils.helpers import get_param, load_image, boolean_string, predict_grpc
import glob
import os

LOG = logging.getLogger(__name__)

backgrounds = {'None': None}
glob_background = None
style_srv = 'styles:9000'


def init_hook(**params):
    backgrounds_dir = params.get('backgrounds', None)
    global style_srv
    style_srv = params.get('style_srv', 'styles:9000')
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


obj_classes = {
    'Person': 1
}


def limit(v, l, r, d):
    if v < l:
        return d
    if v > r:
        return d
    return v


def apply_style(img, style):
    outputs = predict_grpc({'image': img.astype(np.uint8),
                            'style': style},
                           style_srv)
    return outputs['output'][:, :, ::-1]


def process(inputs, ct_x, **kwargs):
    original_image, is_video = load_image(inputs, 'inputs')
    if original_image is None:
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

    if not boolean_string(get_param(inputs, 'return_origin_size', False)):
        original_image = image

    try:
        area_threshold = int(get_param(inputs, 'area_threshold', 0))
    except:
        area_threshold = 0
    area_threshold = limit(area_threshold, 0, 100, 0)
    try:
        max_objects = int(get_param(inputs, 'max_objects', 1))
    except:
        max_objects = 1
    max_objects = limit(max_objects, 1, 10, 1)

    try:
        pixel_threshold = int(float(get_param(inputs, 'pixel_threshold', 0.5)) * 255)
    except:
        pixel_threshold = int(0.5 * 255)

    pixel_threshold = limit(pixel_threshold, 1, 254, int(0.5 * 255))

    object_classes = [obj_classes.get(get_param(inputs, 'object_class', 'Person'), 1)]
    effect = get_param(inputs, 'effect', 'Remove background')  # Remove background,Mask,Blur

    try:
        blur_radius = int(get_param(inputs, 'blur_radius', 2))
    except:
        blur_radius = 2

    blur_radius = limit(blur_radius, 1, 10, 2)

    outputs = ct_x.drivers[0].predict({'inputs': np.expand_dims(image, axis=0)})
    num_detection = int(outputs['num_detections'][0])
    if num_detection < 1:
        return _return(original_image)

    process_width = image.shape[1]
    process_height = image.shape[0]
    image_area = process_width * process_height
    detection_boxes = outputs["detection_boxes"][0][:num_detection]
    detection_boxes = detection_boxes * [process_height, process_width, process_height, process_width]
    detection_boxes = detection_boxes.astype(np.int32)
    detection_classes = outputs["detection_classes"][0][:num_detection]
    detection_masks = outputs["detection_masks"][0][:num_detection]
    masks = []
    for i in range(num_detection):
        if int(detection_classes[i]) not in object_classes:
            continue
        box = detection_boxes[i]
        mask_image = cv2.resize(detection_masks[i], (box[3] - box[1], box[2] - box[0]), interpolation=cv2.INTER_LINEAR)
        left = max(0, box[1] - 50)
        right = min(process_width, box[3] + 50)
        upper = max(0, box[0] - 50)
        lower = min(process_height, box[2] + 50)
        box_mask = np.pad(mask_image, ((box[0] - upper, lower - box[2]), (box[1] - left, right - box[3])), 'constant')
        area = int(np.sum(np.greater_equal(box_mask, 0.5).astype(np.int32)))
        if area * 100 / image_area < area_threshold:
            continue
        masks.append((area, box_mask, [upper, left, lower, right]))

    if len(masks) < 1:
        return _return(original_image)
    masks = sorted(masks, key=lambda row: -row[0])
    total_mask = np.zeros((process_height, process_width), np.float32)
    min_left = process_width
    min_upper = process_height
    max_right = 0
    max_lower = 0
    for i in range(min(len(masks), max_objects)):
        pre_mask = masks[i][1]
        box = masks[i][2]
        left = max(0, box[1])
        right = min(process_width, box[3])
        upper = max(0, box[0])
        lower = min(process_height, box[2])
        box_mask = np.pad(pre_mask, ((upper, process_height - lower), (left, process_width - right)), 'constant')
        total_mask = np.maximum(total_mask, box_mask)
        if left < min_left:
            min_left = left
        if right > max_right:
            max_right = right
        if upper < min_upper:
            min_upper = upper
        if lower > max_lower:
            max_lower = lower
    mask = np.uint8(total_mask[min_upper:max_lower, min_left:max_right] * 255)
    box = (min_upper, min_left, max_lower, max_right)
    if len(mask.shape) > 2:
        logging.warning('Mask shape is {}'.format(mask.shape))
        mask = mask[:, :, 0]
    image = cv2.resize(image[box[0]:box[2], box[1]:box[3], :], (320, 320))
    mask = cv2.resize(mask, (320, 320))
    mask[np.less_equal(mask, pixel_threshold)] = 0
    mask[np.greater(mask, pixel_threshold)] = 255
    input_trimap = generate_trimap(mask)
    input_trimap = np.expand_dims(input_trimap.astype(np.float32), 2)
    image = image.astype(np.float32)
    input_image = image - g_mean
    outputs = ct_x.drivers[1].predict(
        {'input': np.expand_dims(input_image, axis=0), 'trimap': np.expand_dims(input_trimap, axis=0)})
    mask = outputs.get('mask', None)
    if mask is None:
        mask = outputs['output'][0] * 255
        mask = np.reshape(mask, (320, 320))
        mask = np.clip(mask, 0, 255)
        mask = mask.astype(np.uint8)
        mask = cv2.resize(mask, (box[3] - box[1], box[2] - box[0]))
    mask = mask.astype(np.float32) / 255
    c
    if mask.shape != original_image.shape:
        mask = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]))
    mask = cv2.GaussianBlur(mask, (21, 21), 11)
    if effect == 'Remove background':
        background = None
        if 'background_img' in inputs:
            background, _ = load_image(inputs, 'background_img')
        if background is None:
            back_name = get_param(inputs, 'background', None)
            if back_name is not None:
                background = backgrounds.get(back_name)
            else:
                if glob_background is not None:
                    background = glob_background
                else:
                    background = backgrounds.get('None')
        add_style = get_param(inputs, 'style', '')
        if len(add_style) > 0:
            image = apply_style(original_image, add_style).astype(np.float32)
        else:
            image = original_image.astype(np.float32)
        mask = np.expand_dims(mask, 2)
        if background is not None:
            image = image * mask
            background = cv2.resize(background, (image.shape[1], image.shape[0]))
            background = background.astype(np.float32)
            background = background * (1 - mask)
            image = background + image
            image = image.astype(np.uint8)
        else:
            if not is_video:
                mask = (mask * 255).astype(np.uint8)
                image = image[:, :, ::-1].astype(np.uint8)
                image = np.concatenate([image, mask], axis=2)
            else:
                image = image * mask
                image = image.astype(np.uint8)
    elif effect == "Mask":
        mask = mask * 255
        image = mask.astype(np.uint8)
    else:
        image = original_image.astype(np.float32)
        mask = np.expand_dims(mask, 2)
        foreground = mask * image
        radius = min(max(blur_radius, 2), 10)
        if effect == 'Grey':
            background = rgb2gray(original_image)
        else:
            background = cv2.GaussianBlur(original_image, (radius, radius), 10)
        background = (1.0 - mask) * background.astype(np.float32)
        image = foreground + background
        image = image.astype(np.uint8)

    return _return(image)


def rgb2gray(rgb):
    gray = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    return np.stack([gray, gray, gray], axis=2)


g_mean = np.array(([126.88, 120.24, 112.19])).reshape([1, 1, 3])
unknown_code = 128


def generate_trimap(alpha):
    trimap = np.copy(alpha)
    k_size = 20
    trimap[np.where((ndimage.grey_dilation(alpha[:, :], size=(k_size, k_size)) - ndimage.grey_erosion(alpha[:, :],
                                                                                                      size=(k_size,
                                                                                                            k_size))) != 0)] = unknown_code
    return trimap
