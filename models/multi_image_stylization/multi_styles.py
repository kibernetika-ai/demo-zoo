from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import scipy
import numpy as np
import models.multi_image_stylization.vgg19 as vgg
import logging

from models.multi_image_stylization.layers import *


def _mst_net(x, style_control=None, reuse=False):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
        x = conv_layer(x, 32, 9, 1, style_control=style_control, name='conv1')
        x = conv_layer(x, 64, 3, 2, style_control=style_control, name='conv2')
        x = conv_layer(x, 128, 3, 2, style_control=style_control, name='conv3')
        x = residual_block(x, 3, style_control=style_control, name='res1')
        x = residual_block(x, 3, style_control=style_control, name='res2')
        x = residual_block(x, 3, style_control=style_control, name='res3')
        x = residual_block(x, 3, style_control=style_control, name='res4')
        x = residual_block(x, 3, style_control=style_control, name='res5')
        x = conv_tranpose_layer(x, 64, 3, 2, style_control=style_control, name='up_conv1')
        x = pooling(x)
        x = conv_tranpose_layer(x, 32, 3, 2, style_control=style_control, name='up_conv2')
        x = pooling(x)
        x = conv_layer(x, 3, 9, 1, relu=False, style_control=style_control, name='output')
        preds = tf.nn.tanh(x) * 150 + 255. / 2
    return preds


def _styles_model_fn(features, labels, mode, params=None, config=None, model_dir=None):
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    content_inputs = features['content_inputs']
    #if training:

    style_control = features['style_control']
    preds = _mst_net(content_inputs/255.0, style_control=style_control)
    export_outputs = None
    if training:
        style_inputs = labels
        weights = scipy.io.loadmat(params['vgg19'])
        vgg_mean = tf.constant(np.array([103.939, 116.779, 123.68]).reshape((1, 1, 1, 3)), dtype='float32')
        #logging.info("vgg19 {}".format(weights))
        content_feats = vgg.net(content_inputs - vgg_mean, weights)
        style_feats = vgg.net(style_inputs - vgg_mean, weights)
        net = vgg.net(preds - vgg_mean, weights)

        c_loss = params['content_weights'] * euclidean_loss(net[-1], content_feats[-1])
        tf.summary.scalar('content_loss', c_loss)
        s_loss = params['style_weights'] * sum([style_loss(net[i], style_feats[i]) for i in range(5)])
        tf.summary.scalar('style_loss', s_loss)
        tv_loss = params['tv_weights'] * total_variation(preds)
        tf.summary.scalar('tv_loss', tv_loss)
        total_loss = c_loss + s_loss + tv_loss
        # Adding Image summaries to the tensorboard.
        tf.summary.image('image/0_content_inputs', content_inputs, 3)
        tf.summary.image('image/1_style_inputs_aug', style_inputs, 3)
        tf.summary.image('image/2_stylized_images', preds*255.0, 3)

        train_op = tf.train.AdamOptimizer(params['learning_rate']).minimize(total_loss,
                                                                            global_step=tf.train.get_or_create_global_step())
    else:
        total_loss = None
        train_op = None
        export_outputs = {
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
                preds)}

    return tf.estimator.EstimatorSpec(
        mode=mode,
        eval_metric_ops={},
        predictions=preds,
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
