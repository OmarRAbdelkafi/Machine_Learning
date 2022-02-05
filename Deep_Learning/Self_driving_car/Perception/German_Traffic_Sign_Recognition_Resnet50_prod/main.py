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

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import scipy.misc
from resnets_utils import *
from ResNET50 import *
from prediction import *
import matplotlib.pyplot as plt

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Num CPUs Available: ", len(tf.config.list_physical_devices('CPU')))

if __name__ == '__main__':

    input_shape = (64, 64, 3)
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset(input_shape)
    Nb_classes = len(classes)
    print (classes)

    #Verify_images(X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes)

    # Normalize image vectors
    X_train = X_train_orig / 255.
    X_test = X_test_orig / 255.

    # Convert training and test labels to one hot matrices
    Y_train = convert_to_one_hot(Y_train_orig, len(classes)).T
    Y_test = convert_to_one_hot(Y_test_orig, len(classes)).T

    Verify_shapes(X_train, Y_train, X_test, Y_test, classes)

    Mode = 'load'

    if Mode == 'train':
        model = ResNet50(input_shape, Nb_classes)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        #print(model.summary())

        model.fit(X_train, Y_train, epochs = 7, batch_size = 32)

        Predict_evaluation(model, X_test, Y_test)

        # save model
        model.save('resnet50.h5')
        print('Model Saved!')

    if Mode == 'load':
        # load model weight
        pre_trained_model = load_model('resnet50.h5')
        print('Model Loaded!')

        Predict_evaluation(pre_trained_model, X_test, Y_test)

        Verify_predictions(X_test, Y_test_orig, pre_trained_model, classes)

        img_path = 'images/00080.png'
        My_prediction(img_path, input_shape, pre_trained_model, classes)
