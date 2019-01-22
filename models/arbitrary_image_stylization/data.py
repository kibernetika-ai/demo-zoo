import tensorflow as tf


def train_input_fn(params):
    def _input_fn():
        images = imagenet_inputs(params['images'], params['image_size'])
        styles = arbitrary_style_image_inputs(params['styles'],
                                              image_size=params['image_size'],
                                              center_crop=params['center_crop'],
                                              shuffle=True,
                                              augment_style_images=params['augment_style_images'],
                                              random_style_image_size=params['random_style_image_size'],
                                              min_rand_image_size=128,
                                              max_rand_image_size=300)
        ds = tf.data.Dataset.zip((images, styles))

        def _feature(x,y):
            return {'content_inputs': x, 'style_inputs': y}, tf.ones([params['batch_size']], tf.int32)
        ds = ds.map(_feature)
        return ds.repeat(params['epoch']).batch(params['batch_size'])

    return _input_fn


def imagenet_inputs(dataset, image_size, shuffle=True):
    ds = tf.data.Dataset.list_files(dataset, shuffle=shuffle)

    def _extract(file):
        image_content = tf.read_file(file)
        image = tf.image.decode_image(image_content, 3)
        image.set_shape([None, None, 3])
        image = _aspect_preserving_resize(image, image_size + 2)
        image = _central_crop([image], image_size, image_size)[0]
        # pylint: enable=protected-access
        image.set_shape([image_size, image_size, 3])
        image = tf.to_float(image) / 255.0
        return image

    return ds.map(_extract).repeat()


def arbitrary_style_image_inputs(dataset,
                                 image_size=None,
                                 center_crop=True,
                                 shuffle=True,
                                 augment_style_images=False,
                                 random_style_image_size=False,
                                 min_rand_image_size=128,
                                 max_rand_image_size=300):

    ds = tf.data.Dataset.list_files(dataset, shuffle=shuffle)

    def _extract(file):
        image_content = tf.read_file(file)
        image = tf.image.decode_image(image_content, 3)
        image.set_shape([None, None, 3])
        if image_size is not None:
            image_channels = image.shape[2].value
            if augment_style_images:
                image_orig = image
                image = tf.image.random_brightness(image, max_delta=0.8)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_flip_left_right(image)
                image = tf.image.random_flip_up_down(image)
                random_larger_image_size = tf.random_uniform([],
                                                             minval=image_size + 2,
                                                             maxval=image_size + 200,
                                                             dtype=tf.int32)
                image = _aspect_preserving_resize(image, random_larger_image_size)
                image = tf.random_crop(image, size=[image_size, image_size, image_channels])
                image.set_shape([image_size, image_size, image_channels])
                image_orig = _aspect_preserving_resize(image_orig, image_size + 2)
                image_orig = _central_crop([image_orig], image_size, image_size)[0]
                image_orig.set_shape([image_size, image_size, 3])
            elif center_crop:
                image = _aspect_preserving_resize(image, image_size + 2)
                image = _central_crop([image], image_size, image_size)[0]
                image.set_shape([image_size, image_size, image_channels])

            else:
                image = _aspect_preserving_resize(image, image_size)

        image = tf.to_float(image) / 255.0

        if random_style_image_size:
            image = _aspect_preserving_resize(image,
                                              tf.random_uniform(
                                                  [],
                                                  minval=min_rand_image_size,
                                                  maxval=max_rand_image_size,
                                                  dtype=tf.int32))

        return image

    return ds.map(_extract).repeat()


def _aspect_preserving_resize(image, smallest_side):
    """Resize images preserving the original aspect ratio.
    Args:
      image: A 3-D image or a 4-D batch of images `Tensor`.
      smallest_side: A python integer or scalar `Tensor` indicating the size of
        the smallest side after resize.
    Returns:
      resized_image: A 3-D or 4-D tensor containing the resized image(s).
    """
    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

    input_rank = len(image.get_shape())
    if input_rank == 3:
        image = tf.expand_dims(image, 0)

    shape = tf.shape(image)
    height = shape[1]
    width = shape[2]
    new_height, new_width = _smallest_size_at_least(height, width, smallest_side)
    resized_image = tf.image.resize_bilinear(image, [new_height, new_width],
                                             align_corners=False)
    if input_rank == 3:
        resized_image = tf.squeeze(resized_image)
        resized_image.set_shape([None, None, 3])
    else:
        resized_image.set_shape([None, None, None, 3])
    return resized_image


def _smallest_size_at_least(height, width, smallest_side):
    """Computes new shape with the smallest side equal to `smallest_side`.
    Computes new shape with the smallest side equal to `smallest_side` while
    preserving the original aspect ratio.
    Args:
      height: an int32 scalar tensor indicating the current height.
      width: an int32 scalar tensor indicating the current width.
      smallest_side: A python integer or scalar `Tensor` indicating the size of
        the smallest side after resize.
    Returns:
      new_height: an int32 scalar tensor indicating the new height.
      new_width: and int32 scalar tensor indicating the new width.
    """
    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

    height = tf.to_float(height)
    width = tf.to_float(width)
    smallest_side = tf.to_float(smallest_side)

    scale = tf.cond(tf.greater(height, width),
                    lambda: smallest_side / width,
                    lambda: smallest_side / height)
    new_height = tf.to_int32(height * scale)
    new_width = tf.to_int32(width * scale)
    return new_height, new_width


def _central_crop(image_list, crop_height, crop_width):
    """Performs central crops of the given image list.
    Args:
      image_list: a list of image tensors of the same dimension but possibly
        varying channel.
      crop_height: the height of the image following the crop.
      crop_width: the width of the image following the crop.
    Returns:
      the list of cropped images.
    """
    outputs = []
    for image in image_list:
        image_height = tf.shape(image)[0]
        image_width = tf.shape(image)[1]

        offset_height = (image_height - crop_height) / 2
        offset_width = (image_width - crop_width) / 2

        outputs.append(_crop(image, offset_height, offset_width,
                             crop_height, crop_width))
    return outputs


def _crop(image, offset_height, offset_width, crop_height, crop_width):
    """Crops the given image using the provided offsets and sizes.
    Note that the method doesn't assume we know the input image size but it does
    assume we know the input image rank.
    Args:
      image: an image of shape [height, width, channels].
      offset_height: a scalar tensor indicating the height offset.
      offset_width: a scalar tensor indicating the width offset.
      crop_height: the height of the cropped image.
      crop_width: the width of the cropped image.
    Returns:
      the cropped (and resized) image.
    Raises:
      InvalidArgumentError: if the rank is not 3 or if the image dimensions are
        less than the crop size.
    """
    original_shape = tf.shape(image)

    rank_assertion = tf.Assert(
        tf.equal(tf.rank(image), 3),
        ['Rank of image must be equal to 3.'])
    with tf.control_dependencies([rank_assertion]):
        cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])

    size_assertion = tf.Assert(
        tf.logical_and(
            tf.greater_equal(original_shape[0], crop_height),
            tf.greater_equal(original_shape[1], crop_width)),
        ['Crop size greater than the image size.'])

    offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

    # Use tf.strided_slice instead of crop_to_bounding box as it accepts tensors
    # to define the crop size.
    with tf.control_dependencies([size_assertion]):
        image = tf.strided_slice(image, offsets, offsets + cropped_shape,
                                 strides=tf.ones_like(offsets))
    return tf.reshape(image, cropped_shape)
