import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import imageio
import matplotlib.pyplot as plt


def load_data():
    path = ''
    image_path = os.path.join(path, './dataA/CameraRGB/')
    mask_path = os.path.join(path, './dataA/CameraSeg/')
    image_list = os.listdir(image_path)
    mask_list = os.listdir(mask_path)
    image_list = [image_path+i for i in image_list]
    mask_list = [mask_path+i for i in mask_list]

    return image_list, mask_list

def check_images(image_list, mask_list):
    N = 1
    img = imageio.imread(image_list[N])
    mask = imageio.imread(mask_list[N])

    fig, arr = plt.subplots(1, 2, figsize=(14, 10))
    arr[0].imshow(img)
    arr[0].set_title('Image')
    arr[1].imshow(mask[:, :, 0])
    arr[1].set_title('Segmentation')
    plt.show()

def split_data_img_mask(image_list, mask_list):
    image_list_ds = tf.data.Dataset.list_files(image_list, shuffle=False)
    mask_list_ds = tf.data.Dataset.list_files(mask_list, shuffle=False)
    '''
    for path in zip(image_list_ds.take(4), mask_list_ds.take(4)):
        print(path)
    '''
    image_filenames = tf.constant(image_list)
    masks_filenames = tf.constant(mask_list)
    dataset = tf.data.Dataset.from_tensor_slices((image_filenames, masks_filenames))
    '''
    for image, mask in dataset.take(1):
        print(image)
        print(mask)
    '''
    return dataset

def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

def Show_dataset(image_ds, processed_image_ds):
    for image, mask in image_ds.take(1):
        sample_image, sample_mask = image, mask
        print(mask.shape)
    display([sample_image, sample_mask])

    for image, mask in processed_image_ds.take(1):
        sample_image, sample_mask = image, mask
        print(mask.shape)
    display([sample_image, sample_mask])
