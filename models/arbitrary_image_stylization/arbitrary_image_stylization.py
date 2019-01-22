from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.training import session_run_hook
import logging

slim = tf.contrib.slim
from models.arbitrary_image_stylization import arbitrary_image_stylization_build_model as build_model


def _styles_model_fn(features, labels, mode, params=None, config=None, model_dir=None):
    training = (mode == tf.estimator.ModeKeys.TRAIN)

    content_inputs_ = features['content_inputs']
    style_inputs_ = features['style_inputs']

    stylized_images, total_loss, loss_dict, _ = build_model.build_model(
        content_inputs_,
        style_inputs_,
        trainable=True,
        is_training=training,
        inception_end_point='Mixed_6e',
        style_prediction_bottleneck=100,
        adds_losses=True,
        content_weights=params['content_weights'],
        style_weights=params['style_weights'],
        total_variation_weight=params['total_variation_weight'])

    export_outputs = None
    if training:
        for key, value in loss_dict.items():
            tf.summary.scalar(key, value)

        # Adding Image summaries to the tensorboard.
        tf.summary.image('image/0_content_inputs', content_inputs_, 3)
        tf.summary.image('image/1_style_inputs_aug', style_inputs_, 3)
        tf.summary.image('image/2_stylized_images', stylized_images, 3)

        optimizer = tf.train.AdamOptimizer(params['learning_rate'])
        train_op = slim.learning.create_train_op(
            total_loss,
            optimizer,
            clip_gradient_norm=params['clip_gradient_norm'],
            summarize_gradients=False)
    else:
        total_loss = None
        train_op = None
        export_outputs = {
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
                stylized_images)}

    return tf.estimator.EstimatorSpec(
        mode=mode,
        eval_metric_ops={},
        predictions=stylized_images,
        loss=total_loss,
        training_hooks=[InitVGG16Hook(params['vgg16'])],
        evaluation_hooks=[],
        export_outputs=export_outputs,
        train_op=train_op)


class Styles(tf.estimator.Estimator):
    def __init__(
            self,
            params=None,
            model_dir=None,
            config=None,
            warm_start_from=None
    ):
        def _model_fn(features, labels, mode, params, config):
            return _styles_model_fn(
                features=features,
                labels=labels,
                mode=mode,
                params=params,
                config=config,
                model_dir=model_dir,
            )

        super(Styles, self).__init__(
            model_fn=_model_fn,
            model_dir=model_dir,
            config=config,
            params=params,
            warm_start_from=warm_start_from
        )

class InitVGG16Hook(session_run_hook.SessionRunHook):
    def __init__(self, model_path):
        self._model_path = model_path
        self._ops = None

    def begin(self):
        if self._model_path is not None:
            self._ops = slim.assign_from_checkpoint_fn(self._model_path,
                                                   slim.get_variables('vgg_16'))

    def after_create_session(self, session, coord):
        logging.info('Do VGG16 Init')
        if self._ops is not None:
            self._ops(session)

    def before_run(self, run_context):  # pylint: disable=unused-argument
        return None

    def after_run(self, run_context, run_values):
        None
