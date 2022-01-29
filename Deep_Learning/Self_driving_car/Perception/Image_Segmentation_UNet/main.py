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
* Objectif : V0 -> 98%
'''

import tensorflow as tf
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Num CPUs Available: ", len(tf.config.list_physical_devices('CPU')))

'''
# 2. EDA (Exploratory Data Analysis)
'''
import EDA.EDA_process as EDA_P
#origin images (600x800x3)
#set image dimensions to train
img_height = 96
img_width = 128
num_channels = 3

image_list, mask_list = EDA_P.load_data()
EDA_P.check_images(image_list, mask_list)
dataset = EDA_P.split_data_img_mask(image_list, mask_list)

'''
# 3. Pre-processing
'''
import Pre_processing.Pre_P as PPP
image_ds, processed_image_ds = PPP.Pre_Processing_execution(dataset)
EDA_P.Show_dataset(image_ds, processed_image_ds)


'''
# 4. Select the model
'''
import DeepLearningModels.Convolutional_models as DeepConv
unet = DeepConv.Set_unet_model_V0(img_height, img_width, num_channels)

'''
# 5. fit the model
'''
unet, train_dataset = DeepConv.Fit_unet_model_V0(unet, processed_image_ds)

'''
# 6. Prediction
'''
import prediction as Pred
Pred.show_predictions(unet, train_dataset, 3)
