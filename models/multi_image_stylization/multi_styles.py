from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import scipy
import models.multi_image_stylization.vgg19 as vgg
import tensorflow as tf
import logging

import models.multi_image_stylization.layers as layers


def _styles_model_fn(features, labels, mode, params=None, config=None, model_dir=None):
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    content_inputs = features['content_inputs']
    if params['styles_count'] is None:
        style_control = None
    else:
        style_control = features['style_control']
    result = layers.net(content_inputs)
    export_outputs = None
    if training:
        weights = scipy.io.loadmat(params['vgg19'])
        styles_net = vgg.net(weights, vgg.preprocess(labels))
        contents_net = vgg.net(weights, vgg.preprocess(content_inputs))
        results_net = vgg.net(weights, vgg.preprocess(result))
        content_size = params['batch_size'] * layers.tensor_size(results_net[vgg.CONTENT_LAYER])
        content_loss = params['content_weights'] * (2 * tf.nn.l2_loss(
            results_net[vgg.CONTENT_LAYER] - contents_net[vgg.CONTENT_LAYER]) / content_size)
        style_loss = params['style_weights'] * layers.style_loss(params['batch_size'], results_net, styles_net,vgg.STYLE_LAYERS)

        tv_loss = params['tv_weights'] * layers.total_variation(params['batch_size'], result)

        tf.summary.scalar('content_loss', content_loss)
        tf.summary.scalar('style_loss', style_loss)
        tf.summary.scalar('tv_loss', tv_loss)
        total_loss = content_loss + style_loss + tv_loss
        # Adding Image summaries to the tensorboard.
        tf.summary.image('image/0_content_inputs', content_inputs, 3)
        tf.summary.image('image/1_style_inputs_aug', labels, 3)
        tf.summary.image('image/2_stylized_images', result, 3)

        train_op = tf.train.AdamOptimizer(params['learning_rate']).minimize(total_loss,
                                                                            global_step=tf.train.get_or_create_global_step())
    else:
        total_loss = None
        train_op = None
        export_outputs = {
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
                result)}

    return tf.estimator.EstimatorSpec(
        mode=mode,
        eval_metric_ops={},
        predictions=result,
        loss=total_loss,
        training_hooks=[],
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
