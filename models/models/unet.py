import tensorflow as tf
import logging


def conv_block(input, out_chans, drop_prob, name, pooling, training):
    with tf.variable_scope("layer_{}".format(name)):
        out = input
        for j in range(2):
            out = tf.layers.conv2d(out, out_chans, kernel_size=3, padding='same', name="conv_{}".format(j + 1))
            out = tf.layers.batch_normalization(out, training=training, name="bn_{}".format(j + 1))
            out = tf.nn.relu(out, name="relu_{}".format(j + 1))
            if training:
                out = tf.layers.dropout(out, drop_prob, name="dropout_{}".format(j + 1))

        if not pooling:
            return out

        pool = tf.nn.max_pool(out,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME', name='pool')
        return out, pool


def upnet(name, output, num_pool_layers, down_sample_layers, ch, drop_prob, training):
    for i in range(num_pool_layers):
        down = down_sample_layers[len(down_sample_layers) - i - 1]
        _, w, h, f = down.shape
        output = tf.layers.conv2d_transpose(output, filters=f, kernel_size=[3, 3], strides=[2, 2], padding='SAME',
                                            activation=None,
                                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                            name='unpool{}_{}'.format(name, i + 1))
        output = tf.concat([output, down], 3)
        logging.info('Up{}_{} - {}'.format(name, i + 1, output.shape))
        if i < (num_pool_layers - 1):
            ch //= 2
        output = conv_block(output, ch, drop_prob, 'up{}_{}'.format(name, i + 1), False, training)
    return output


def unet(inputs, out_chans, chans, drop_prob, num_pool_layers, training=True):
    output, pull = conv_block(inputs, chans, drop_prob, 'down_1', True, training)
    logging.info('Down_1 - {}'.format(output.shape))
    down_sample_layers = [output]
    ch = chans
    for i in range(num_pool_layers - 1):
        ch *= 2
        output, pull = conv_block(pull, ch, drop_prob, 'down_{}'.format(i + 2), True, training)
        logging.info('Down_{} - {}'.format(i + 2, output.shape))
        down_sample_layers += [output]
    i += 1

    final_down = conv_block(pull, ch, drop_prob, 'down_{}'.format(i + 2), False, training)
    logging.info('Down_{} - {}'.format(i + 2, output.shape))

    output = upnet('', final_down, num_pool_layers, down_sample_layers, ch, drop_prob, training)
    output = tf.layers.conv2d(output, ch, kernel_size=1, padding='same', name="conv_1")
    output = tf.layers.conv2d(output, out_chans, kernel_size=1, padding='same', name="conv_2_mask")
    output = tf.layers.conv2d(output, out_chans, kernel_size=1, padding='same', name="final_mask")
    return output
