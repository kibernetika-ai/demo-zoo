import tensorflow as tf
import numpy as np

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
    preds = tf.nn.sigmoid(conv_t3)
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


def gram_matrix(feature_maps):
    batch_size, height, width, channels = tf.unstack(tf.shape(feature_maps))
    denominator = tf.to_float(height * width)
    feature_maps = tf.reshape(
        feature_maps, tf.stack([batch_size, height * width, channels]))
    matrix = tf.matmul(feature_maps, feature_maps, adjoint_a=True)
    return matrix / denominator


def total_variation_loss(x):
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]
    channels = shape[3]
    y_size = tf.to_float((height - 1) * width * channels)
    x_size = tf.to_float(height * (width - 1) * channels)
    y_loss = tf.nn.l2_loss(
        x[:, 1:, :, :] - x[:, :-1, :, :]) / y_size
    x_loss = tf.nn.l2_loss(
        x[:, :, 1:, :] - x[:, :, :-1, :]) / x_size
    loss = (y_loss + x_loss) / tf.to_float(batch_size)
    return loss


def content_loss(x1, x2):
    loss = tf.reduce_mean((x1 - x2) ** 2)
    return loss


def style_loss(x1, x2, layers):
    total_style_loss = np.float32(0.0)
    for style_layer in layers:
        x1_layer = x1[style_layer]
        x2_layer = x2[style_layer]
        loss = tf.reduce_mean((gram_matrix(x1_layer) - gram_matrix(x2_layer)) ** 2)
        total_style_loss += loss

    return total_style_loss
