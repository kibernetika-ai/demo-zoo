import logging
import math
import time
from ml_serving.drivers import driver

import cv2
from ml_serving.utils import helpers
import numpy as np
import os

LOG = logging.getLogger(__name__)


def intersec_area(b1, b2):
    x1 = max(b1[0], b2[0])
    x2 = min(b1[2], b2[2])
    y1 = max(b1[1], b2[1])
    y2 = min(b1[3], b2[3])
    if (x2 - x1) > 0 and (y2 - y1) > 0:
        return (x2 - x1) * (y2 - y1) / ((b1[2] - b1[0]) * (b1[3] - b1[1]))
    return 0


class TFOpenCVFaces:
    def __init__(self, model_path, use_tensor_rt=False):
        self._model_path = model_path
        drv = driver.load_driver('tensorflow')
        self.serving = drv()
        _model = 'opencv_face_detector_uint8.pb'
        if use_tensor_rt:
            _model = 'opencv_face_detector_uint8_rt_fp16.pb'

        self.serving.load_model(os.path.join(self._model_path, _model), inputs='data:0',
                                outputs='mbox_loc:0,mbox_conf_flatten:0')
        configFile = self._model_path + "/detector.pbtxt"
        self.net = cv2.dnn.readNetFromTensorflow(None, configFile)

        self.prior = np.fromfile(self._model_path + '/mbox_priorbox.np', np.float32)
        self.prior = np.reshape(self.prior, (1, 2, 35568))
        self.threshold = 0.5
        ##Dry run
        self.bboxes(np.zeros((300, 300, 3), np.uint8))

    def bboxes(self, frame):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        blob = cv2.dnn.blobFromImage(frame[:, :, ::-1], 1.0, (300, 300), [104, 117, 123], False, False)
        blob = np.transpose(blob, (0, 2, 3, 1))
        result = self.serving.predict({'data:0': blob})
        probs = result.get('mbox_conf_flatten:0')
        boxes = result.get('mbox_loc:0')
        st1 = time.time()
        self.net.setInput(boxes, name='mbox_loc')
        self.net.setInput(probs, name='mbox_conf_flatten')
        self.net.setInput(self.prior, name='mbox_priorbox')
        detections = self.net.forward()
        # LOG.info('BoxDetect time {}ms'.format(int((time.time() - st1) * 1000)))
        bboxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.threshold:
                x1 = max(int(detections[0, 0, i, 3] * frameWidth), 0)
                y1 = max(int(detections[0, 0, i, 4] * frameHeight), 0)
                x2 = min(int(detections[0, 0, i, 5] * frameWidth), frameWidth)
                y2 = min(int(detections[0, 0, i, 6] * frameHeight), frameHeight)
                if x1 > x2:
                    x2 = x1
                if y1 > y2:
                    y2 = y1
                bboxes.append(np.array([x1, y1, x2, y2], np.int32))
        return bboxes

    def stop(self, ctx):
        self.serving.release()


class OpenVinoFaces:
    def __init__(self, model_path):
        self._model_path = model_path
        drv = driver.load_driver('openvino')
        self.serving = drv()
        self.serving.load_model(self._model_path)
        self.input_name, self.input_shape = list(self.serving.inputs.items())[0]
        self.output_name = list(self.serving.outputs)[0]
        self.threshold = 0.5

    def bboxes(self, frame):
        inference_frame = cv2.resize(frame, tuple(self.input_shape[:-3:-1]), interpolation=cv2.INTER_LINEAR)
        inference_frame = np.transpose(inference_frame, [2, 0, 1]).reshape(self.input_shape)
        outputs = self.serving.predict({self.input_name: inference_frame})
        output = outputs[self.output_name]
        output = output.reshape(-1, 7)
        bboxes_raw = output[output[:, 2] > self.threshold]
        # Extract 5 values
        boxes = bboxes_raw[:, 3:7]
        confidence = np.expand_dims(bboxes_raw[:, 2], axis=0).transpose()
        boxes = np.concatenate((boxes, confidence), axis=1)
        # Assign confidence to 4th
        # boxes[:, 4] = bboxes_raw[:, 2]
        xmin = boxes[:, 0] * frame.shape[1]
        xmax = boxes[:, 2] * frame.shape[1]
        ymin = boxes[:, 1] * frame.shape[0]
        ymax = boxes[:, 3] * frame.shape[0]
        xmin[xmin < 0] = 0
        xmax[xmax > frame.shape[1]] = frame.shape[1]
        ymin[ymin < 0] = 0
        ymax[ymax > frame.shape[0]] = frame.shape[0]
        boxes = []
        for i in range(len(xmin)):
            boxes.append(np.array([int(xmin[i]), int(ymin[i]), int(xmax[i]), int(ymax[i])], np.int32))

        return boxes

    def stop(self, ctx):
        pass


class YoungModel:
    def __init__(self, style_size, model_path):
        drv = driver.load_driver('tensorflow')
        self._style_driver = drv()
        self._style_driver.load_model(model_path)
        self._style_size = style_size
        self._style_input_name = list(self._style_driver.inputs.keys())[0]
        self._style_driver.predict(
            {self._style_input_name: np.zeros((1, self._style_size, self._style_size, 3), np.float32)})

    def process(self, ctx, img, box):
        img = img[box[1]:box[3], box[0]:box[2], :]
        i_img = scale_to_inference_image(img, self._style_size)
        outputs = self._style_driver.predict(
            {self._style_input_name: np.expand_dims(norm_to_inference(i_img), axis=0)})
        output = list(outputs.values())[0].squeeze()
        output = inverse_transform(output)
        output = scale(output)
        return i_img, output, box


class Pipe:
    def __init__(self, models, **params):
        self._portret = srt_2_bool(params.get('_portret', False))
        self._mirror = srt_2_bool(params.get('mirror', False))
        self._alpha = int(params.get('alpha', 255))
        self._draw_box = srt_2_bool(params.get('draw_box', False))
        self._open_vino_model_path = params.get('openvino_model_path', None)
        self._tf_opencv_model_path = params.get('tf_opencv_model_path', None)
        self._style_size = int(params.get('style_size', 256))
        self._face_detection_type = params.get('face_detection_type', None)
        self.face_detector = models.get('face_detector')
        if self.face_detector is None:
            if self._face_detection_type == 'tf-opencv':
                face_detection_tensor_rt = srt_2_bool(params.get('face_detection_tensor_rt', False))
                self.face_detector = TFOpenCVFaces(self._tf_opencv_model_path, use_tensor_rt=face_detection_tensor_rt)
            else:
                self.face_detector = OpenVinoFaces(self._open_vino_model_path)
        self._output_view = params.get('output_view', 's')
        self._transfer_mode = params.get('transfer_mode', 'box_margin')
        self._color_correction = srt_2_bool(params.get('color_correction', 'True'))
        self.style_model = models.get('style_model', None)
        if self.style_model is None:
            self.style_model = YoungModel(self._style_size, params.get('beauty_model_path', None))
        self._mask_orig = np.zeros((self._style_size, self._style_size, 3), np.float32)
        for x in range(self._style_size):
            for y in range(self._style_size):
                xv = x - self._style_size / 2
                yv = y - self._style_size / 2
                r = math.sqrt(xv * xv + yv * yv)
                if r > (self._style_size / 2 - 5):
                    self._mask_orig[y, x, :] = min((r - self._style_size / 2 + 5) / 5, 1)
        self._mask_face = 1 - self._mask_orig

        self._overlay_img = params.get('overlay', '')
        self._overlay = None
        self._background = None

    def add_overlay(self, img):
        if self._overlay_img == '':
            return img
        if self._overlay is None:
            try:
                overlay = cv2.imread('{}-{}x{}.png'.format(self._overlay_img, img.shape[1], img.shape[0]),
                                     cv2.IMREAD_UNCHANGED)
                if overlay is None:
                    self._overlay = ()
                else:
                    mask = overlay[:, :, 3:].astype(np.float32)
                    mask = mask / mask.max()
                    overlay = overlay[:, :, 0:3][:, :, ::-1].astype(np.float32) * mask
                    self._overlay = (overlay, 1 - mask)
            except:
                self._overlay = ()
        if len(self._overlay) < 1:
            return img
        img = img.astype(np.float32)
        img = img * self._overlay[1] + self._overlay[0]
        return img.astype(np.uint8)

    def process(self, inputs, ctx, **kwargs):
        alpha = int(helpers.get_param(inputs, 'alpha', self._alpha))
        original, is_video = helpers.load_image(inputs, 'image')
        if self._portret:
            original = np.transpose(original, (1, 0, 2))
        output_view = helpers.get_param(inputs, 'output_view', self._output_view)
        if output_view == 'horizontal' or output_view == 'h':
            x0 = int(original.shape[1] / 4)
            x1 = int(original.shape[1] / 2) + x0
            original = original[:, x0:x1, :]
        if output_view == 'vertical' or output_view == 'v':
            y0 = int(original.shape[0] / 4)
            y1 = int(original.shape[0] / 2) + y0
            original = original[y0:y1, :, :]
        boxes = self.face_detector.bboxes(original)
        boxes.sort(key=lambda box: abs((box[3] - box[1]) * (box[2] - box[0])), reverse=True)

        box = None
        if len(boxes) > 0:
            box = boxes[0].astype(int)
            if box[3] - box[1] < 1 or box[2] - box[0] < 1:
                box = None
        image = original.copy()
        if box is not None and self.style_model is not None:
            inference_img, output, box = self.style_model.process(ctx, image, box)
            alpha = np.clip(alpha, 1, 255)
            if srt_2_bool(helpers.get_param(inputs, 'color_correction', self._color_correction)):
                output = color_tranfer(output, inference_img)
            if helpers.get_param(inputs, 'transfer_mode', self._transfer_mode) == 'direct':
                output = (inference_img * self._mask_orig + output * self._mask_face).astype(np.uint8)
                output = cv2.resize(output, (box[2] - box[0], box[3] - box[1]), interpolation=cv2.INTER_LINEAR)
                image[box[1]:box[3], box[0]:box[2], :] = output
            else:
                output = cv2.resize(np.array(output), (box[2] - box[0], box[3] - box[1]), interpolation=cv2.INTER_AREA)
                if helpers.get_param(inputs, 'transfer_mode', self._transfer_mode) == 'box_margin':
                    xmin = max(0, box[0] - 50)
                    wleft = box[0] - xmin
                    ymin = max(0, box[1] - 50)
                    wup = box[1] - ymin
                    xmax = min(image.shape[1], box[2] + 50)
                    ymax = min(image.shape[0], box[3] + 50)
                    out = image[ymin:ymax, xmin:xmax, :]
                    center = (wleft + output.shape[1] // 2, wup + output.shape[0] // 2)
                    samples = int(helpers.get_param(inputs, 'samples', 0))
                    if samples > 1:
                        results = {'s_0': cv2.imencode('.jpg', image)[1].tostring()}
                        step_alpha = 255.0 / (samples - 1)
                        for si in range(samples - 1):
                            alpha = int(step_alpha * (si + 1))
                            s_image = image.copy()
                            out_copy = out.copy()
                            out_copy = cv2.seamlessClone(output, out_copy, np.ones_like(output) * alpha, center,
                                                         cv2.NORMAL_CLONE)
                            s_image[ymin:ymax, xmin:xmax, :] = out_copy
                            results[f's_{si + 1}'] = cv2.imencode('.jpg', s_image)[1].tostring()
                        return results
                    else:
                        out = cv2.seamlessClone(output, out, np.ones_like(output) * alpha, center, cv2.NORMAL_CLONE)
                        image[ymin:ymax, xmin:xmax, :] = out
                else:
                    center = (box[0] + output.shape[1] // 2, box[1] + output.shape[0] // 2)
                    if not (center[0] >= output.shape[1] or box[1] + output.shape[0] // 2 >= output.shape[0]):
                        image = cv2.seamlessClone(output, image, np.ones_like(output) * alpha, center, cv2.NORMAL_CLONE)
            if len(box) > 0:
                if srt_2_bool(helpers.get_param(inputs, "draw_box", self._draw_box)):
                    image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2, 8)

        result = {}
        image = self.maybe_mirror(image)
        if output_view == 'horizontal' or output_view == 'h' or output_view == 'fh':
            image = np.hstack((self.maybe_mirror(original), image))
        elif output_view == 'vertical' or output_view == 'v':
            image = np.vstack((self.maybe_mirror(original), image))
        image = self.add_overlay(image)
        image = image[:, :, ::-1]
        if not is_video:
            image_bytes = cv2.imencode('.jpg', image)[1].tostring()
        else:
            image_bytes = image
            h = 480
            w = int(480 * image.shape[1] / image.shape[0])
            result['status'] = cv2.resize(image, (w, h))

        result['output'] = image_bytes
        return result

    def maybe_mirror(self, img):
        if self._mirror:
            return img[:, ::-1, :]
        else:
            return img

    def stop(self, ctx):
        self.face_detector.stop(ctx)


def init_hook(ctx, **params):
    LOG.info('Init params:')
    return Pipe({}, **params)


def update_hook(ctx, **kwargs):
    prev_detector = None
    style_model = None
    if ctx.global_ctx is not None:
        LOG.info('close existing pipe')
        # ctx.global_ctx.stop(ctx)
        prev_detector = ctx.global_ctx.face_detector
        style_model = ctx.global_ctx.style_model
    return Pipe({'face_detector': prev_detector, 'style_model': style_model}, **kwargs)


def process(inputs, ctx, **kwargs):
    return ctx.global_ctx.process(inputs, ctx, **kwargs)


def scale(img, high=255, low=0, cmin=None, cmax=None):
    if not cmin:
        cmin = img.min()
        cmax = img.max()

    cscale = cmax - cmin
    scale = float(high - low) / cscale
    bytedata = (img - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(np.uint8)


def inverse_transform(images):
    return (images + 1.) / 2.


def scale_to_inference_image(img, style_size=256):
    return cv2.resize(img, (style_size, style_size), interpolation=cv2.INTER_LINEAR)


def norm_to_inference(img):
    return img / 127.5 - 1.


def srt_2_bool(v):
    if v == 'True':
        return True
    if v == 'False':
        return False
    return bool(v)


def color_tranfer(s, t):
    s = cv2.cvtColor(s, cv2.COLOR_RGB2LAB)
    t = cv2.cvtColor(t, cv2.COLOR_RGB2LAB)
    s_mean, s_std = get_mean_and_std(s)
    t_mean, t_std = get_mean_and_std(t)
    s = ((s - s_mean) * (t_std / s_std)) + t_mean
    s = np.clip(s, 0, 255).astype(np.uint8)
    return cv2.cvtColor(s, cv2.COLOR_LAB2RGB)


def get_mean_and_std(x):
    x_mean, x_std = cv2.meanStdDev(x)
    x_mean = np.hstack(np.around(x_mean, 2))
    x_std = np.hstack(np.around(x_std, 2))
    return x_mean, x_std
