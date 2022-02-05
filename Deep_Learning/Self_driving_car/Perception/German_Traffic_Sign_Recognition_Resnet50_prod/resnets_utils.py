import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import math
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt

def create_classes_dic():
    ## Creating a dictionary for class labels
    classes = { 0:'Speed limit (20km/h)',
                1:'Speed limit (30km/h)',
                2:'Speed limit (50km/h)',
                3:'Speed limit (60km/h)',
                4:'Speed limit (70km/h)',
                5:'Speed limit (80km/h)',
                6:'End of speed limit (80km/h)',
                7:'Speed limit (100km/h)',
                8:'Speed limit (120km/h)',
                9:'No passing',
                10:'No passing veh over 3.5 tons',
                11:'Right-of-way at intersection',
                12:'Priority road',
                13:'Yield',
                14:'Stop',
                15:'No vehicles',
                16:'Veh > 3.5 tons prohibited',
                17:'No entry',
                18:'General caution',
                19:'Dangerous curve left',
                20:'Dangerous curve right',
                21:'Double curve',
                22:'Bumpy road',
                23:'Slippery road',
                24:'Road narrows on the right',
                25:'Road work',
                26:'Traffic signals',
                27:'Pedestrians',
                28:'Children crossing',
                29:'Bicycles crossing',
                30:'Beware of ice/snow',
                31:'Wild animals crossing',
                32:'End speed + passing limits',
                33:'Turn right ahead',
                34:'Turn left ahead',
                35:'Ahead only',
                36:'Go straight or right',
                37:'Go straight or left',
                38:'Keep right',
                39:'Keep left',
                40:'Roundabout mandatory',
                41:'End of no passing',
                42:'End no passing veh > 3.5 tons' }
    return classes

def View_images_classes(cwd, meta):
    ## Viewing images belonging to each class
    plt.figure(figsize=(30,30))
    for i, file in enumerate(meta):
        img = Image.open(cwd+'Meta/'+file)
        ax = plt.subplot(9,5,i+1)
        ax.imshow(img)
        ax.set_title(file, size=20)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def size_images(train_df, test_df):
    print(f'minimum width: {train_df.Width.min()}')
    print(f'minimum height: {train_df.Height.min()}')
    print(f'average width: {train_df.Width.mean()}')
    print(f'average height: {train_df.Height.mean()}')

def image_datasets_to_array(input_shape, cwd, train_df, test_df):
    train_x =[]
    for i in train_df.Path:
        img = Image.open(cwd+i)       # reading image
        img = img.resize((input_shape[0],input_shape[1]))     # reasizing image
        train_x.append(np.array(img)) # saving image as array to train

    train_y = np.array(train_df.ClassId)
    train_x = np.array(train_x)

    test_x =[]
    for i in test_df.Path:
        img = Image.open(cwd+i)
        img = img.resize((input_shape[0],input_shape[1]))
        test_x.append(np.array(img))

    test_y = np.array(test_df.ClassId)
    test_x = np.array(test_x)

    return train_x, train_y, test_x, test_y

def load_dataset(input_shape):
    cwd = './datasets/gtsrb-german-traffic-sign/'
    #print(os.listdir(cwd))
    meta = os.listdir(cwd+'Meta')
    meta.remove('.~lock.ClassesInformation.ods#')
    meta.remove('.~lock.ClassesInformationStrong.ods#')
    #print(meta)

    #View_images_classes(cwd, meta)

    classes = create_classes_dic()

    train_df = pd.read_csv(cwd+'Train.csv')
    test_df = pd.read_csv(cwd+'Test.csv')
    #print(train_df.head())
    #print(train_df.describe())

    #size_images(train_df, test_df)

    train_x, train_y, test_x, test_y = image_datasets_to_array(input_shape, cwd, train_df, test_df)

    return train_x, train_y, test_x, test_y, classes

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def Verify_images(X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes):
    print(classes[Y_test_orig[1000]])
    plt.imshow(X_test_orig[1000])
    plt.axis('off')
    plt.show()

def Verify_shapes(X_train, Y_train, X_test, Y_test, classes):
    print ("number of training examples = " + str(X_train.shape[0]))
    print ("number of test examples = " + str(X_test.shape[0]))
    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(Y_train.shape))
    print ("X_test shape: " + str(X_test.shape))
    print ("Y_test shape: " + str(Y_test.shape))
    print ("Number of classes: " + str(len(classes)))
