#coding:utf-8
# To activate this environment, use
#
#     $ conda activate tf
#     or
#     $ conda activate tf-gpu
#
# To deactivate an active environment, use
#
#     $ conda deactivate

#coding:utf-8

#Project description
'''
*Deep Neural Network for Image Segmentation:

*Presenting the initial data:
https://carla.readthedocs.io/en/latest/ref_sensors/#camera-semantic-segmentation
dataset A containing:
    - a training set of 1000 images RGB camera labelled with 23 classes

* Lvl humain < 0.5% accurancy error
'''

# 1. Define a mesurable objectif :
'''
* Predict the classification of cat images
* Metrics :  - Accurancy
* Objectif : V0 -> 90%
'''

import tensorflow as tf
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import EDA.EDA_process as EDA_P
import Pre_processing.Pre_P as PPP
import DeepLearningModels.Convolutional_models as DeepConv
import prediction as Pred
from tensorflow.keras.models import Model, load_model

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Num CPUs Available: ", len(tf.config.list_physical_devices('CPU')))

'''
# 2. EDA (Exploratory Data Analysis)
'''
#origin images (600x800x3)
#set image dimensions to train (power of 2)
img_height = 384 #96
img_width = 512 #128
num_channels = 3

image_list, mask_list = EDA_P.load_data()
EDA_P.check_images(image_list, mask_list)
dataset = EDA_P.split_data_img_mask(image_list, mask_list)

'''
# 3. Pre-processing
'''
image_ds, processed_image_ds = PPP.Pre_Processing_execution(dataset, img_height, img_width)
EDA_P.Show_dataset(image_ds, processed_image_ds)


Mode = 'load' #train

if Mode == 'train':

    '''
    # 4. Select the model
    '''
    unet = DeepConv.Set_unet_model_V0(img_height, img_width, num_channels)

    '''
    # 5. fit the model
    '''
    unet, train_dataset = DeepConv.Fit_unet_model_V0(unet, processed_image_ds)

    # save model
    unet.save('UNET.h5')
    print('Model Saved!')

    '''
    # 6. Prediction
    '''
    Pred.show_predictions(unet, train_dataset, 3)

if Mode == 'load':

    BUFFER_SIZE = 500
    BATCH_SIZE = 32
    processed_image_ds.batch(BATCH_SIZE)
    train_dataset = processed_image_ds.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    print(processed_image_ds.element_spec)

    # load model weight
    pre_trained_model = load_model('UNET.h5')
    print('Model Loaded!')
    pre_trained_model.summary()

    '''
    # 6. Prediction
    '''
    Pred.show_predictions(pre_trained_model, train_dataset, 3)
