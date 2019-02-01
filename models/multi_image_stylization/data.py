import tensorflow as tf

import glob
from PIL import Image
import numpy as np
import os


def train_input_fn(params):
    def _input_fn():
        images = imagenet_inputs(params['images'], params['image_size'])
        styles = style_image_inputs(params['styles'], image_size=params['image_size'], style=params['style'])
        ds = tf.data.Dataset.zip((images, styles))

        def _feature(x, y):
            return {'content_inputs': x, 'style_control': y[1]}, y[0]

        ds = ds.map(_feature)
        return ds.repeat(params['epoch']).batch(params['batch_size'])

    return _input_fn


def _extract_image(image_size):
    def _extract(file):
        image_content = tf.read_file(file)
        image = tf.image.decode_image(image_content, 3)
        image.set_shape([None, None, 3])
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, [image_size, image_size], align_corners=False)
        image = tf.squeeze(image)
        image.set_shape([image_size, image_size, 3])
        image = tf.to_float(image)
        return image

    return _extract


def imagenet_inputs(dataset, image_size, shuffle=True):
    ds = tf.data.Dataset.list_files(dataset, shuffle=shuffle)
    return ds.map(_extract_image(image_size))


def style_image_inputs(dataset, image_size, style=None):
    all_files = glob.glob(dataset)
    all_files = sorted(all_files)
    if style is not None:
        files = None
        for f in all_files:
            _, name = os.path.split(f)
            if name == style:
                files = [f]
                break
            name = name.split('.')
            if name[0] == style:
                files = [f]
                break
        if files is None:
            files = all_files[0]
    else:
        files = all_files

    def _generator():
        for i, f in enumerate(files):
            image = Image.open(f).convert('RGB')
            image = image.resize((image_size, image_size), Image.BILINEAR)
            image = np.asarray(image, np.float32)
            styles_control = np.zeros((len(files)), dtype=np.float32)
            styles_control[i] = 1
            yield (image, styles_control)

    ds = tf.data.Dataset.from_generator(_generator, (tf.float32, tf.float32),
                                        (tf.TensorShape([image_size, image_size, 3]),
                                         tf.TensorShape([len(files)])))

    return ds.shuffle(len(files)).repeat()
