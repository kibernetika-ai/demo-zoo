import tensorflow as tf
import os

from scipy import ndimage

from models.models.unet import unet
import logging
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from tensorflow.python.training.session_run_hook import SessionRunArgs
import math
import glob
import json
from mlboardclient.report.tensorflow_rpt import MlBoardReporter
import numpy as np
import cv2
import random

unknown_code = 128


def generate_weight(mask):
    weight = mask.astype(np.uint8)
    weight = np.reshape(weight, (weight.shape[0], weight.shape[1]))
    weight[weight > 0] = 255
    trimap = generate_trimap(weight)
    weight = weight.astype(np.float32)
    weight[weight == 0] = 1
    weight[weight == 255] = 1.5
    weight[trimap == unknown_code] = 2
    return np.reshape(weight, (weight.shape[0], weight.shape[1], 1))


def generate_trimap(alpha):
    trimap = np.copy(alpha)
    k_size = 5
    trimap[np.where((ndimage.grey_dilation(alpha[:, :], size=(k_size, k_size)) - ndimage.grey_erosion(alpha[:, :],
                                                                                                      size=(k_size,
                                                                                                            k_size))) != 0)] = unknown_code
    return trimap


def _coco_images(params):
    coco_dir = params['coco']
    with open(coco_dir + '/annotations/instances_train2017.json') as f:
        coco_data = json.load(f)
    coco_data = coco_data['annotations']
    coco_images = {}
    people = {}
    for a in coco_data:
        i_id = a['image_id']
        if a['category_id'] != 1:
            if i_id in people:
                continue
            else:
                coco_images[i_id] = True
        else:
            if i_id in coco_images:
                del coco_images[i_id]
            people[i_id] = True
    del people
    names = []
    for k in coco_images.keys():
        name = '{}/train2017/{:012d}.jpg'.format(coco_dir, int(k))
        names.append(name)
    return names


def _crop_back(img, size=160):
    w = max(img.shape[1], size)
    h = max(img.shape[0], size)
    img = cv2.resize(img, (w, h))
    x_shift = int(np.random.uniform(0, w - size))
    y_shift = int(np.random.uniform(0, h - size))
    return img[y_shift:y_shift + size, x_shift:x_shift + size, :]


def mix_fb(front, back, mask, x_shift, y_shift, use_seamless):
    w = mask.shape[1]
    h = mask.shape[0]

    rmask = np.zeros((160, 160, 1), np.float32)

    if not use_seamless:
        maskf = cv2.GaussianBlur(mask, (3, 3), 3)
        maskf = maskf.astype(np.float32) / 255
        maskf = np.reshape(maskf, (h, w, 1))
        front = front.astype(np.float32) * maskf
        back = back.astype(np.float32)
        back[y_shift:y_shift + h, x_shift:x_shift + w, :] = front + (
                    back[y_shift:y_shift + h, x_shift:x_shift + w, :] * (1 - maskf))
        mask = np.reshape(mask, (h, w, 1))
        mask = mask.astype(np.float32)
    else:
        mask = mask.astype(np.uint8)
        mask = np.reshape(mask, (h, w))
        center = (x_shift + front.shape[1] // 2, y_shift + front.shape[0] // 2)
        front = front.astype(np.uint8)
        back = cv2.seamlessClone(front, back, mask, center, cv2.MIXED_CLONE)
        mask = mask.astype(np.float32)
        mask = np.reshape(mask, (h, w, 1))
    rmask[y_shift:y_shift + h, x_shift:x_shift + w, :] = mask
    return back.astype(np.uint8), rmask.astype(np.uint8)


def video_data_fn(params, training):
    import albumentations
    data_set = params['data_set']
    files = glob.glob(data_set + '/masks/*.*')
    for i in range(len(files)):
        mask = files[i]
        img = os.path.basename(mask)
        img = data_set + '/images/' + img
        files[i] = (img, mask)
    coco_images = _coco_images(params)

    def _pre_aug(p=0.5):
        return albumentations.Compose([
            albumentations.HorizontalFlip(),
            albumentations.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=15, p=0.8),
            albumentations.GridDistortion(distort_limit=0.3, p=0.3),
            albumentations.OneOf([
                albumentations.MotionBlur(p=0.5),
                albumentations.Blur(blur_limit=3, p=0.2),
            ], p=0.2),
            albumentations.OneOf([
                albumentations.CLAHE(clip_limit=2),
                albumentations.IAASharpen(),
                albumentations.IAAEmboss(),
            ], p=0.3),
            albumentations.OneOf([
                albumentations.RandomBrightnessContrast(p=0.3),
            ], p=0.4),
            albumentations.HueSaturationValue(p=0.3),
        ], p=p)

    def _move_aug(p=0.5):
        return albumentations.Compose([
            albumentations.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=10, p=0.9),
            albumentations.GridDistortion(distort_limit=0.08, p=0.7)
        ], p=p)

    def _post_aug(p=0.5):
        return albumentations.Compose([
            albumentations.OneOf([
                albumentations.MotionBlur(p=0.5),
            ], p=0.2)
        ], p=p)

    pre_aug = _pre_aug(p=1)
    move_aug = _move_aug(p=1)
    post_aug = _post_aug(p=1)

    def make_pre_aug(img, mask):
        data = {"image": img, "mask": mask}
        data = pre_aug(**data)
        return data["image"], data["mask"]

    def make_move_aug(img, mask):
        data = {"image": img, "mask": mask}
        data = move_aug(**data)
        return data["image"], data["mask"]

    def make_post_aug(img):
        data = {"image": img}
        data = post_aug(**data)
        return data["image"]

    def _input_fn():
        def _generator():
            for i in files:
                pre_img = cv2.imread(i[0])[:, :, ::-1]
                pre_mask = cv2.imread(i[1])[:, :, :]
                if len(pre_mask.shape) == 3:
                    pre_mask = cv2.cvtColor(pre_mask, cv2.COLOR_BGR2GRAY)
                pre_mask = cv2.medianBlur(pre_mask, 3)
                pre_img, pre_mask = make_pre_aug(pre_img, pre_mask)
                s = np.random.uniform(0.5, 1)
                w0 = pre_img.shape[1]
                h0 = pre_img.shape[0]
                w = int(s * w0)
                h = int(s * h0)
                front_img1 = cv2.resize(pre_img, (w, h))
                pmask1 = cv2.resize(pre_mask, (w, h))
                front_img0, pmask0 = make_move_aug(front_img1, pmask1)
                name = random.choice(coco_images)
                back_img = cv2.imread(name)[:, :, ::-1]
                back_img = _crop_back(back_img, 160)
                x_shift = int(np.random.uniform(0, w0 - w))
                y_shift = int(np.random.uniform(0, h0 - h))
                front_img1, pmask1 = mix_fb(front_img1, back_img, pmask1, x_shift, y_shift, False)
                front_img0, pmask0 = mix_fb(front_img0, back_img, pmask0, x_shift, y_shift, False)
                front_img1 = make_post_aug(front_img1)
                front_img0 = make_post_aug(front_img0)
                thresh = cv2.cvtColor(front_img1, cv2.COLOR_BGR2GRAY)
                thresh = cv2.medianBlur(thresh, 3)
                thresh = cv2.adaptiveThreshold(thresh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                thresh = cv2.medianBlur(thresh, 3)
                thresh = np.reshape(thresh, (front_img1.shape[0], front_img1.shape[1], 1))
                thresh = thresh.astype(np.float32) / 255

                front_img1 = front_img1.astype(np.float32) / 255
                front_img0 = front_img0.astype(np.float32) / 255

                weight = generate_weight(pmask1)
                pmask1 = pmask1.astype(np.float32) / 255
                img = np.concatenate([front_img1, front_img0, thresh], axis=2)
                yield img, np.concatenate([pmask1, weight], axis=2)

        ds = tf.data.Dataset.from_generator(_generator, (tf.float32, tf.float32),
                                            (tf.TensorShape([160, 160, 7]), tf.TensorShape([160, 160, 2])))
        if training:
            ds = ds.shuffle(params['batch_size'] * 2, reshuffle_each_iteration=True)
        if training:
            ds = ds.repeat(params['num_epochs'])

        ds = ds.batch(params['batch_size'], True)

        return ds

    return len(files) // params['batch_size'], _input_fn


def augumnted_data_fn(params, training):
    import albumentations
    def _strong_aug(p=0.5):
        return albumentations.Compose([
            albumentations.HorizontalFlip(),
            albumentations.OneOf([
                albumentations.MotionBlur(p=0.2),
                albumentations.MedianBlur(blur_limit=3, p=0.1),
                albumentations.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            albumentations.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=15, p=0.3),
            albumentations.OneOf([
                albumentations.OpticalDistortion(p=0.3),
                albumentations.GridDistortion(p=0.1),
                albumentations.IAAPiecewiseAffine(p=0.3),
            ], p=0.2),
            albumentations.OneOf([
                albumentations.CLAHE(clip_limit=2),
                albumentations.IAASharpen(),
                albumentations.IAAEmboss(),
            ], p=0.3),
            albumentations.OneOf([
                albumentations.RandomBrightnessContrast(p=0.3),
            ], p=0.4),
            albumentations.HueSaturationValue(p=0.3),
        ], p=p)

    augmentation = _strong_aug(p=0.9)
    data_set = params['data_set']
    files = glob.glob(data_set + '/masks/*.*')
    for i in range(len(files)):
        mask = files[i]
        img = os.path.basename(mask)
        img = data_set + '/images/' + img
        files[i] = (img, mask)
    coco_images = _coco_images(params)

    def _input_fn():
        def _generator():
            for i in files:
                img = cv2.imread(i[0])[:, :, ::-1]
                mask = cv2.imread(i[1])[:, :, :]
                if len(mask.shape) == 3:
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                mask = cv2.medianBlur(mask, 3)
                if np.random.uniform(0, 1) > 0.2:
                    s = np.random.uniform(0.5, 1)
                    w0 = img.shape[1]
                    h0 = img.shape[0]
                    w = int(s * w0)
                    h = int(s * h0)
                    front_img1 = cv2.resize(img, (w, h))
                    pmask1 = cv2.resize(mask, (w, h))
                    name = random.choice(coco_images)
                    back_img = cv2.imread(name)[:, :, ::-1]
                    back_img = _crop_back(back_img, 160)
                    x_shift = int(np.random.uniform(0, w0 - w))
                    y_shift = int(np.random.uniform(0, h0 - h))
                    front_img1, pmask1 = mix_fb(front_img1, back_img, pmask1, x_shift, y_shift, False)
                    img = front_img1.astype(np.uint8)
                    mask = pmask1.astype(np.uint8)
                    if len(mask.shape) == 3:
                        mask = mask[:, :, 0]

                data = {"image": img, "mask": mask}
                augmented = augmentation(**data)
                img, mask = augmented["image"], augmented["mask"]
                mask = np.reshape(mask, (160, 160, 1))
                weight = generate_weight(mask)
                img = img.astype(np.float32) / 255
                mask = mask.astype(np.float32) / 255
                yield img, np.concatenate([mask, weight], axis=2)

        ds = tf.data.Dataset.from_generator(_generator, (tf.float32, tf.float32),
                                            (tf.TensorShape([160, 160, 3]), tf.TensorShape([160, 160, 2])))
        if training:
            ds = ds.shuffle(params['batch_size'] * 2, reshuffle_each_iteration=True)
        if training:
            ds = ds.repeat(params['num_epochs'])

        ds = ds.batch(params['batch_size'], True)

        return ds

    return len(files) // params['batch_size'], _input_fn


def data_fn(params, training):
    data_set = params['data_set']
    files = glob.glob(data_set + '/masks/*.*')
    for i in range(len(files)):
        mask = files[i]
        img = os.path.basename(mask)
        img = data_set + '/images/' + img
        files[i] = [img, mask]
    resolution = params['resolution']
    logging.info('Number of Files: {}'.format(files))

    def _input_fn():
        ds = tf.data.Dataset.from_tensor_slices(files)

        def _read_images(a):
            img = tf.read_file(a[0])
            img = tf.image.decode_image(img)
            mask = tf.read_file(a[1])
            mask = tf.image.decode_image(mask)
            img = tf.expand_dims(img, 0)
            mask = tf.expand_dims(mask, 0)
            img = tf.image.resize_bilinear(img, [resolution, resolution])
            mask = tf.image.resize_bilinear(mask, [resolution, resolution])
            logging.info('img: {}'.format(img.shape))
            logging.info('mask: {}'.format(mask.shape))
            img = tf.reshape(img, [resolution, resolution, 3])
            mask = tf.reshape(mask[:, :, :, 0], [resolution, resolution, 1])
            img = tf.cast(img, dtype=tf.float32) / 255
            mask = tf.cast(mask, dtype=tf.float32) / 255
            weight = tf.ones_like(mask,dtype=tf.float32)
            return img, tf.concat([mask,weight],axis=2)

        ds = ds.map(_read_images)
        if training:
            ds = ds.shuffle(params['batch_size'] * 2, reshuffle_each_iteration=True)
        if training:
            ds = ds.repeat(params['num_epochs'])

        ds = ds.batch(params['batch_size'], True)

        return ds

    return len(files) // params['batch_size'], _input_fn


def _unet_model_fn(features, labels, mode, params=None, config=None, model_dir=None):
    features_definition = params['features']
    if features_definition is None or len(features_definition) == 0:
        features_definition = [3]
    if mode == tf.estimator.ModeKeys.PREDICT:
        all_features = features['image']
    else:
        feature_chans = sum(features_definition)
        all_features = tf.reshape(features,
                                  [params['batch_size'], params['resolution'], params['resolution'], feature_chans])

    prev = 0
    refines = []
    for i in range(len(features_definition)):
        k = features_definition[i]
        if k == 0:
            continue
        f = all_features[:, :, :, prev:k + prev]
        if i == 0:
            features = f
            logging.info('Features: {}'.format(f.shape))
        else:
            logging.info('Features {}: {}'.format(i, f.shape))
            refines.append(f)
        prev = k + prev

    training = (mode == tf.estimator.ModeKeys.TRAIN)
    out_chans = 2 if params['loss'] == 'entropy' else 1
    # inputs, out_chans, chans, drop_prob, num_pool_layers, training = True
    logits = unet(features, out_chans, params['num_chans'], params['drop_prob'], params['num_pools'], refines=refines,
                  training=training)
    if params['loss'] == 'entropy':
        mask = tf.cast(tf.argmax(logits, axis=3), tf.float32)
        logging.info('Mask shape1: {}'.format(mask.shape))
        mask = tf.expand_dims(mask, -1)
        logging.info('Mask shape2: {}'.format(mask.shape))
    else:
        mask = tf.sigmoid(logits)
    logging.info('GRAPH: Features: {}'.format(features.shape))
    if labels is not None:
        logging.info('GRAPH: Lables: {}'.format(labels.shape))
    logging.info('GRAPH: mask: {}'.format(mask.shape))
    loss = None
    train_op = None
    hooks = []
    export_outputs = None
    eval_hooks = []
    chief_hooks = []
    metrics = {}
    if mode != tf.estimator.ModeKeys.PREDICT:
        learning_rate_var = tf.Variable(float(params['lr']), trainable=False, name='lr',
                                        collections=[tf.GraphKeys.LOCAL_VARIABLES])

        weights = labels[:, :, :, 1:]
        labels = labels[:, :, :, 0:1]
        flabels = tf.cast(labels, tf.float32)
        if params['loss'] == 'entropy':
            llabels = tf.cast(labels, tf.int32)
            logits = tf.reshape(logits, [tf.shape(logits)[0], -1, 2])
            llabels = tf.reshape(llabels, [tf.shape(llabels)[0], -1])
            loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=llabels)
        elif params['loss'] == 'image':
            original = features * flabels
            predicted = features * mask
            loss_content = tf.losses.absolute_difference(original, predicted)
            mask_loss = tf.losses.mean_squared_error(flabels, mask, weights=weights)
            loss = (loss_content + mask_loss) * 0.5
        else:
            loss = tf.losses.absolute_difference(flabels, mask, weights=weights)
        mse = tf.losses.mean_squared_error(flabels, mask)
        nmse = tf.norm(flabels - mask) ** 2 / tf.norm(flabels) ** 2

        global_step = tf.train.get_or_create_global_step()
        epoch = global_step // params['epoch_len']
        if training:
            tf.summary.scalar('lr', learning_rate_var)
            tf.summary.scalar('mse', mse)
            tf.summary.scalar('nmse', nmse)
            board_hook = MlBoardReporter('', {
                "_step": global_step,
                "_epoch": epoch,
                "_train_loss": loss,
                '_train_lr': learning_rate_var,
                '_train_mse': mse,
                '_train_nmse': nmse}, submit_summary=False, every_steps=params['save_summary_steps'])
            chief_hooks = [board_hook]
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                if params['optimizer'] == 'AdamOptimizer':
                    opt = tf.train.AdamOptimizer(learning_rate_var)
                else:
                    opt = tf.train.RMSPropOptimizer(learning_rate_var, params['weight_decay'])
                train_op = opt.minimize(loss, global_step=global_step)

        tf.summary.image('Src', features, 3)
        if params['loss'] == 'image':
            tf.summary.image('Reconstruction', predicted, 3)
            tf.summary.image('Original', original, 3)
        else:
            rimage = (mask - tf.reduce_min(mask))
            rimage = rimage / tf.reduce_max(rimage)
            tf.summary.image('Reconstruction', rimage, 3)
            limage = (labels - tf.reduce_min(labels))
            limage = limage / tf.reduce_max(limage)
            tf.summary.image('Original', limage, 3)
        hooks = [TrainingLearningRateHook(
            params['epoch_len'],
            learning_rate_var,
            float(params['lr']),
            int(params['lr_step_size']),
            float(params['lr_gamma']))]
        if not training:
            metrics['mse'] = tf.metrics.mean(mse)
            metrics['nmse'] = tf.metrics.mean(nmse)
    else:
        export_outputs = {
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
                mask)}

    return tf.estimator.EstimatorSpec(
        mode=mode,
        eval_metric_ops=metrics,
        predictions=mask,
        training_chief_hooks=chief_hooks,
        loss=loss,
        training_hooks=hooks,
        export_outputs=export_outputs,
        evaluation_hooks=eval_hooks,
        train_op=train_op)


class TrainingLearningRateHook(session_run_hook.SessionRunHook):
    def __init__(self, epoch_len, learning_rate_var, initial_learning_rate, lr_step_size, lr_gamma):
        self._learning_rate_var = learning_rate_var
        self._lr_step_size = lr_step_size
        self._lr_gamma = lr_gamma
        self._epoch_len = epoch_len
        self._prev_learning_rate = 0
        self._initial_learning_rate = initial_learning_rate
        self._epoch = -1

    def begin(self):
        self._global_step_tensor = training_util._get_or_create_global_step_read()  # pylint: disable=protected-access
        if self._global_step_tensor is None:
            raise RuntimeError("Global step should be created to use LearningRateHook.")
        self._args = [self._global_step_tensor, self._learning_rate_var]
        self._learning_rate_ph = tf.placeholder(tf.float32, name='learning_rate_ph')
        self._learning_rate_op = self._learning_rate_var.assign(self._learning_rate_ph)

    def after_create_session(self, session, coord):
        session.run(self._learning_rate_op, {self._learning_rate_ph: self._initial_learning_rate})

    def before_run(self, run_context):  # pylint: disable=unused-argument
        return SessionRunArgs(self._args)

    def after_run(self, run_context, run_values):
        result = run_values.results
        global_step = result[0]
        learning_rate = result[1]
        if learning_rate != self._prev_learning_rate:
            logging.warning('Set Learning rate to {} at global step {}'.format(learning_rate, global_step))
            self._prev_learning_rate = learning_rate
        epoch = global_step // self._epoch_len

        if self._epoch != epoch:
            logging.info('Start epoch {}'.format(epoch))
            self._epoch = epoch

        lr_step = epoch // self._lr_step_size
        if lr_step > 0:
            desired_learning_rate = self._initial_learning_rate * math.pow(self._lr_gamma, lr_step)
        else:
            desired_learning_rate = self._initial_learning_rate

        if self._prev_learning_rate != desired_learning_rate:
            run_context.session.run(self._learning_rate_op, {self._learning_rate_ph: desired_learning_rate})


class FastBGNet(tf.estimator.Estimator):
    def __init__(
            self,
            params=None,
            model_dir=None,
            config=None,
            warm_start_from=None
    ):
        def _model_fn(features, labels, mode, params, config):
            return _unet_model_fn(
                features=features,
                labels=labels,
                mode=mode,
                params=params,
                config=config,
                model_dir=model_dir,
            )

        super(FastBGNet, self).__init__(
            model_fn=_model_fn,
            model_dir=model_dir,
            config=config,
            params=params,
            warm_start_from=warm_start_from
        )
