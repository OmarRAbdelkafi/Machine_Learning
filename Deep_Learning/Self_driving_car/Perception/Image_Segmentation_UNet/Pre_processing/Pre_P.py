import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def process_path(image_path, mask_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=3)
    mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
    return img, mask

def preprocess(image, mask, img_height, img_width):
    input_image = tf.image.resize(image, (img_height, img_width), method='nearest')
    input_mask = tf.image.resize(mask, (img_height, img_width), method='nearest')

    return input_image, input_mask

def Pre_Processing_execution(dataset, img_height, img_width):
    image_ds = dataset.map(process_path)
    processed_image_ds = image_ds.map(lambda image, mask : preprocess(image, mask, img_height, img_width))

    return image_ds, processed_image_ds
