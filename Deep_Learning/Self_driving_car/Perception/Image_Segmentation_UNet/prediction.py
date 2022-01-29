import tensorflow as tf
import EDA.EDA_process as EDA_P

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def show_predictions(unet, dataset, num):
    """
    Displays the first image of each of the num batches
    """
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = unet.predict(image)
            EDA_P.display([image[0], mask[0], create_mask(pred_mask)])
    else:
        EDA_P.display([sample_image, sample_mask,create_mask(unet.predict(sample_image[tf.newaxis, ...]))])
