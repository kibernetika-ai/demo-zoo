import tensorflow as tf
import logging

def conv_layer(net, num_filters, filter_size, strides, style_control=None, relu=True, name='conv'):
    with tf.variable_scope(name):
        b,w,h,c = net.get_shape().as_list()
        weights_shape = [filter_size, filter_size, c, num_filters]
        weights_init = tf.get_variable(name, shape=weights_shape, initializer=tf.truncated_normal_initializer(stddev=.01))
        strides_shape = [1, strides, strides, 1]

        p = (filter_size - 1) / 2
        p = tf.cast(p,tf.int32)
        if strides == 1:
            net = tf.pad(net, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            net = tf.nn.conv2d(net, weights_init, strides_shape, padding="VALID")
        else:
            net = tf.nn.conv2d(net, weights_init, strides_shape, padding="SAME")

        net = conditional_instance_norm(net, style_control=style_control)
        if relu:
            net = tf.nn.relu(net)

    return net


def conv_tranpose_layer(net, num_filters, filter_size, strides, style_control=None, name='conv_t'):
    with tf.variable_scope(name):
        b, w, h, c = net.get_shape().as_list()
        weights_shape = [filter_size, filter_size, num_filters, c]
        weights_init = tf.get_variable(name, shape=weights_shape, initializer=tf.truncated_normal_initializer(stddev=.01))

        batch_size, rows, cols, in_channels = [i.value for i in net.get_shape()]
        new_rows, new_cols = int(rows * strides), int(cols * strides)
        # new_shape = #tf.pack([tf.shape(net)[0], new_rows, new_cols, num_filters])

        new_shape = [tf.shape(net)[0], new_rows, new_cols, num_filters]
        tf_shape = tf.stack(new_shape)
        strides_shape = [1,strides,strides,1]

        p = (filter_size - 1) / 2
        if strides == 1:
            net = tf.pad(net, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            net = tf.nn.conv2d_transpose(net, weights_init, tf_shape, strides_shape, padding="VALID")
        else:
            net = tf.nn.conv2d_transpose(net, weights_init, tf_shape, strides_shape, padding="SAME")
        net = conditional_instance_norm(net, style_control=style_control)

    return tf.nn.relu(net)


def residual_block(net, filter_size=3, style_control=None, name='res'):
    with tf.variable_scope(name+'_a'):
        tmp = conv_layer(net, 128, filter_size, 1, style_control=style_control)
    with tf.variable_scope(name+'_b'):
        output = net + conv_layer(tmp, 128, filter_size, 1, style_control=style_control, relu=False)
    return output


def conditional_instance_norm_1(net, style_control, name='cond_in'):
    with tf.variable_scope(name):
        _, _, _, channels = [i.value for i in net.get_shape()]
        bs,num_styles = [i.value for i in style_control.get_shape()]
        mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)

        var_shape = [channels]
        for i in range(num_styles):
            logging.info("Create i {}".format(i))
            with tf.variable_scope('bn_style_{}'.format(i)):
                shift = tf.get_variable('shift', shape=var_shape, initializer=tf.constant_initializer(0.))
                scale = tf.get_variable('scale', shape=var_shape, initializer=tf.constant_initializer(1.))
            style_scale = tf.reduce_sum(style_control*scale,axis=0)
        #x = [tf.reduce_sum(scale*i,axis=0) for i in tf.unstack(style_control)]
        #style_scale = tf.stack(x)
        #tf.matmul
        style_control = tf.broadcast_to(style_control,tf.stack([bs,num_styles,channels]))
        style_scale = tf.reduce_sum(scale*style_control,axis=0)
        #x = [tf.reduce_sum(shift*i,axis=0) for i in tf.unstack(style_control)]
        #style_shift = tf.stack(x)
        style_shift = tf.reduce_sum(shift*style_control,axis=0)
        epsilon = 1e-3
        normalized = (net-mu)/(sigma_sq + epsilon)**(.5)
        tf.nn.batch_normalization
        output = style_scale * normalized + style_shift

def conditional_instance_norm(net, style_control, name='cond_in'):
    with tf.variable_scope(name):
        _, _, _, channels = [i.value for i in net.get_shape()]
        bs,num_styles = [i.value for i in style_control.get_shape()]
        mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)

        var_shape = [channels,num_styles]
        with tf.variable_scope('bn_style'):
            shift = tf.get_variable('shift', shape=var_shape, initializer=tf.constant_initializer(0.))
            scale = tf.get_variable('scale', shape=var_shape, initializer=tf.constant_initializer(1.))

        style_control = tf.tile(style_control,[1,channels])
        style_control = tf.reshape(style_control,[1,channels,num_styles])
        style_scale = tf.reduce_sum(scale*style_control,axis=-1)
        style_shift = tf.reduce_sum(shift*style_control,axis=-1)
        epsilon = 1e-3
        normalized = (net-mu)/(sigma_sq + epsilon)**(.5)
        output = style_scale * normalized + style_shift

    return output


def instance_norm(net, train=True, name='in'):
    with tf.variable_scope(name):
        batch, rows, cols, channels = [i.value for i in net.get_shape()]
        var_shape = [channels]
        mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
        shift = tf.get_variable('shift', shape=var_shape, initializer=tf.constant_initializer(0.))
        scale = tf.get_variable('scale', shape=var_shape, initializer=tf.constant_initializer(1.))
        epsilon = 1e-3
        normalized = (net-mu)/(sigma_sq + epsilon)**(.5)
    return scale * normalized + shift

def pooling(input):
    return tf.nn.avg_pool(input, ksize=(1, 2, 2, 1), strides=(1, 1, 1, 1), padding='SAME')


def total_variation(preds):
    # total variation denoising
    b,w,h,c = preds.get_shape().as_list()
    x = tf.shape(preds)
    n = tf.cast(x[0]*x[1]*x[2]*x[3],tf.float32)
    y_tv = tf.nn.l2_loss(preds[:,1:,:,:] - preds[:,:w-1,:,:])
    x_tv = tf.nn.l2_loss(preds[:,:,1:,:] - preds[:,:,:h-1,:])
    tv_loss = 2*(x_tv + y_tv)/n
    return tv_loss


def euclidean_loss(input_, target_):
    #b,w,h,c = input_.get_shape().as_list()
    x = tf.shape(input_)
    n = tf.cast(x[0]*x[1]*x[2]*x[3],tf.float32)
    return 2 * tf.nn.l2_loss(input_- target_) / n


def gram_matrix(net):
    #b,h,w,c = tf.shape()
    x = tf.shape(net)
    feats = tf.reshape(net, (x[0], x[1]*x[2], x[3]))
    feats_T = tf.transpose(feats, perm=[0,2,1])
    n = tf.cast(x[1]*x[2]*x[3],tf.float32)
    grams = tf.matmul(feats_T, feats) / n
    return grams


def style_loss(input_, style_):
    #b,h,w,c = input_.get_shape().as_list()
    x = tf.shape(input_)
    n = tf.cast(x[0]*x[3]*x[3],tf.float32)
    input_gram = gram_matrix(input_)
    style_gram = gram_matrix(style_)
    return 2 * tf.nn.l2_loss(input_gram - style_gram)/n
