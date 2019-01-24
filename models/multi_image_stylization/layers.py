import tensorflow as tf
import logging
import functools

WEIGHTS_INIT_STDEV = .1


def net(image):
    conv1 = _conv_layer(image, 32, 9, 1)
    conv2 = _conv_layer(conv1, 64, 3, 2)
    conv3 = _conv_layer(conv2, 128, 3, 2)
    resid1 = _residual_block(conv3, 3)
    resid2 = _residual_block(resid1, 3)
    resid3 = _residual_block(resid2, 3)
    resid4 = _residual_block(resid3, 3)
    resid5 = _residual_block(resid4, 3)
    conv_t1 = _conv_tranpose_layer(resid5, 64, 3, 2)
    conv_t2 = _conv_tranpose_layer(conv_t1, 32, 3, 2)
    conv_t3 = _conv_layer(conv_t2, 3, 9, 1, relu=False)
    preds = tf.nn.tanh(conv_t3) * 150 + 255. / 2
    return preds


def _conv_layer(net, num_filters, filter_size, strides, relu=True):
    weights_init = _conv_init_vars(net, num_filters, filter_size)
    strides_shape = [1, strides, strides, 1]
    net = tf.nn.conv2d(net, weights_init, strides_shape, padding='SAME')
    net = _instance_norm(net)
    if relu:
        net = tf.nn.relu(net)

    return net


def _conv_tranpose_layer(net, num_filters, filter_size, strides):
    weights_init = _conv_init_vars(net, num_filters, filter_size, transpose=True)
    _, rows, cols, in_channels = [i.value for i in net.get_shape()]
    new_rows, new_cols = int(rows * strides), int(cols * strides)
    new_shape = [tf.shape(net)[0], new_rows, new_cols, num_filters]
    tf_shape = tf.stack(new_shape)
    strides_shape = [1, strides, strides, 1]

    net = tf.nn.conv2d_transpose(net, weights_init, tf_shape, strides_shape, padding='SAME')
    net = _instance_norm(net)
    return tf.nn.relu(net)


def _residual_block(net, filter_size=3):
    tmp = _conv_layer(net, 128, filter_size, 1)
    return net + _conv_layer(tmp, 128, filter_size, 1, relu=False)


def _instance_norm(net, train=True):
    _, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]
    mu, sigma_sq = tf.nn.moments(net, [1, 2], keep_dims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))
    epsilon = 1e-3
    normalized = (net - mu) / (sigma_sq + epsilon) ** (.5)
    return scale * normalized + shift


def _conv_init_vars(net, out_channels, filter_size, transpose=False):
    _, rows, cols, in_channels = [i.value for i in net.get_shape()]
    if not transpose:
        weights_shape = [filter_size, filter_size, in_channels, out_channels]
    else:
        weights_shape = [filter_size, filter_size, out_channels, in_channels]

    weights_init = tf.Variable(tf.truncated_normal(weights_shape, stddev=WEIGHTS_INIT_STDEV, seed=1), dtype=tf.float32)
    return weights_init


def pooling(input):
    return tf.nn.avg_pool(input, ksize=(1, 2, 2, 1), strides=(1, 1, 1, 1), padding='SAME')


def total_variation(batch_size, preds):
    _, h, w, _ = preds.get_shape().as_list()
    tv_y_size = tensor_size(preds[:, 1:, :, :])
    tv_x_size = tensor_size(preds[:, :, 1:, :])
    y_tv = tf.nn.l2_loss(preds[:, 1:, :, :] - preds[:, :h - 1, :, :])
    x_tv = tf.nn.l2_loss(preds[:, :, 1:, :] - preds[:, :, :w - 1, :])
    return 2 * (x_tv / tv_x_size + y_tv / tv_y_size) / batch_size



def euclidean_loss(input_, target_):
    # b,w,h,c = input_.get_shape().as_list()
    x = tf.shape(input_)
    n = tf.cast(x[0] * x[1] * x[2] * x[3], tf.float32)
    return 2 * tf.nn.l2_loss(input_ - target_) / n


def gram_matrix(batch_size, input):
    _, height, width, filters = map(lambda i: i.value, input.get_shape())
    size = height * width * filters
    feats = tf.reshape(input, (batch_size, height * width, filters))
    feats_T = tf.transpose(feats, perm=[0, 2, 1])
    return tf.matmul(feats_T, feats) / size


def style_loss(batch_size, x1, x2, layers):
    style_losses = []
    for style_layer in layers:
        x1_layer = x1[style_layer]
        x2_layer = x2[style_layer]
        x1_grams = gram_matrix(batch_size, x1_layer)
        x2_grams = gram_matrix(batch_size, x2_layer)
        size = tensor_size(x1_grams)
        style_losses.append(2 * tf.nn.l2_loss(x1_grams - x2_grams) / size)
    return functools.reduce(tf.add, style_losses) / batch_size


def tensor_size(tensor):
    from operator import mul
    return functools.reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)
