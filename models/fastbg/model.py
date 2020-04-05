import tensorflow as tf
import os
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

def augumnted_data_fn_1(params, training):
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
                ],p=0.3),
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
    coco_dir = params['coco']
    with open(coco_dir+'/annotations/instances_train2017.json') as f:
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
    coco_images = list(coco_images.keys())
    def _input_fn():
        def _generator():
            for i in files:
                img = cv2.imread(i[0])[:,:,::-1]
                mask = cv2.imread(i[1])[:,:,:]
                if len(mask.shape)==3:
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                if np.random.uniform(0,1)>0.2:
                    #s = np.random.uniform(0.3,1)
                    s = 1
                    w0 = img.shape[1]
                    h0 = img.shape[0]
                    w = int(s*w0)
                    h = int(s*h0)
                    img0 = cv2.resize(img,(w,h))
                    out_mask = cv2.resize(mask, (w, h))
                    pmask = cv2.GaussianBlur(out_mask, (3, 3), 3)
                    out_mask = out_mask.astype(np.float32)/255
                    out_mask = np.reshape(out_mask,(h,w,1))
                    pmask = pmask.astype(np.float32) / 255
                    pmask = np.reshape(pmask, (h, w, 1))


                    img0 = img0.astype(np.float32)/255*pmask
                    x_shift = int(np.random.uniform(0,w0-w))
                    y_shift = int(np.random.uniform(0, h0 - h))
                    name = '{}/train2017/{:012d}.jpg'.format(coco_dir,int(random.choice(coco_images)))
                    img = cv2.imread(name)[:,:,::-1]
                    img = cv2.resize(img,(160,160))
                    img = img.astype(np.float32)/255
                    mask = np.zeros((160,160,1),np.float32)
                    img[y_shift:y_shift+h,x_shift:x_shift+w,:] = img0+(img[y_shift:y_shift+h,x_shift:x_shift+w,:]*(1-pmask))
                    mask[y_shift:y_shift + h, x_shift:x_shift + w, :] = out_mask
                    img = (img * 255).astype(np.uint8)
                    mask = (mask * 255).astype(np.uint8)
                if len(mask.shape)==3:
                    mask = mask[:, :, 0]
                data = {"image": img, "mask": mask}
                augmented = augmentation(**data)
                img, mask = augmented["image"], augmented["mask"]
                if len(mask.shape)==2:
                    mask = np.reshape(mask,(160,160,1))
                img = img.astype(np.float32)/255
                mask = mask.astype(np.float32)/255
                yield img,mask

        ds = tf.data.Dataset.from_generator(_generator, (tf.float32, tf.float32),
                                            (tf.TensorShape([160, 160, 3]),tf.TensorShape([160, 160, 1])))
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
                ],p=0.4),
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
    coco_dir = params['coco']
    with open(coco_dir+'/annotations/instances_train2017.json') as f:
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
    coco_images = list(coco_images.keys())
    def _input_fn():
        def _generator():
            for i in files:
                img = cv2.imread(i[0])[:,:,::-1]
                mask = cv2.imread(i[1])[:,:,:]
                if len(mask.shape)==3:
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                mask[mask>0] = 255
                if np.random.uniform(0,1)>0.2:
                    pmask = cv2.GaussianBlur(mask, (3, 3), 3)
                    pmask = pmask.astype(np.float32) / 255
                    front_img = img.astype(np.float32)/255.0*pmask
                    name = '{}/train2017/{:012d}.jpg'.format(coco_dir,int(random.choice(coco_images)))
                    img = cv2.imread(name)[:,:,::-1]
                    img = cv2.resize(img,(160,160))
                    img = img.astype(np.float32)/255
                    img = front_img+img*(1-pmask)
                    img = (img * 255).astype(np.uint8)
                data = {"image": img, "mask": mask}
                augmented = augmentation(**data)
                img, mask = augmented["image"], augmented["mask"]
                mask[mask > 0] = 255
                mask = np.reshape(mask,(160,160,1))
                img = img.astype(np.float32)/255
                mask = mask.astype(np.float32)/255
                yield img,mask

        ds = tf.data.Dataset.from_generator(_generator, (tf.float32, tf.float32),
                                            (tf.TensorShape([160, 160, 3]),tf.TensorShape([160, 160, 1])))
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
            img = tf.expand_dims(img,0)
            mask = tf.expand_dims(mask,0)
            img = tf.image.resize_bilinear(img, [resolution, resolution])
            mask = tf.image.resize_bilinear(mask, [resolution, resolution])
            logging.info('img: {}'.format(img.shape))
            logging.info('mask: {}'.format(mask.shape))
            img = tf.reshape(img, [resolution, resolution, 3])
            mask = tf.reshape(mask[:, :, :, 0], [resolution, resolution, 1])
            img = tf.cast(img, dtype=tf.float32) / 255
            mask = tf.cast(mask, dtype=tf.float32) / 255
            return img, mask

        ds = ds.map(_read_images)
        if training:
            ds = ds.shuffle(params['batch_size'] * 2, reshuffle_each_iteration=True)
        if training:
            ds = ds.repeat(params['num_epochs'])

        ds = ds.batch(params['batch_size'], True)

        return ds

    return len(files) // params['batch_size'], _input_fn


def _unet_model_fn(features, labels, mode, params=None, config=None, model_dir=None):
    if mode == tf.estimator.ModeKeys.PREDICT:
        features = features['image']
    else:
        features = tf.reshape(features, [params['batch_size'], params['resolution'], params['resolution'], 3])
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    out_chans = 2 if params['loss'] == 'entropy' else 1
    # inputs, out_chans, chans, drop_prob, num_pool_layers, training = True
    logits = unet(features, out_chans, params['num_chans'], params['drop_prob'], params['num_pools'], training=training)
    if params['loss'] == 'entropy':
        mask = tf.cast(tf.argmax(logits, axis=3), tf.float32)
        logging.info('Mask shape1: {}'.format(mask.shape))
        mask = tf.expand_dims(mask, -1)
        logging.info('Mask shape2: {}'.format(mask.shape))
    else:
        mask = tf.sigmoid(logits)
    logging.info('Features: {}'.format(features.shape))
    if labels is not None:
        logging.info('Lables: {}'.format(labels.shape))
    logging.info('mask: {}'.format(mask.shape))
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
            mask_loss = tf.losses.mean_squared_error(flabels, mask)
            loss = (loss_content + mask_loss) * 0.5
        else:
            loss = tf.losses.absolute_difference(flabels, mask)
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
