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

import tensorflow as tf
import matplotlib.pyplot as plt    # for plotting the images
import numpy as np
from keras.models import load_model

import EDA as EDA
import models.DARKNET_19 as DKT19
import models.trained_weight.read_weight as RW
import yolo_loss as YL
import yolo_train as YT
import predict as pred

ANCHORS = np.array([1.07709888,  1.78171903,  # anchor box 1, width , height
                    2.71054693,  5.12469308,  # anchor box 2, width,  height
                    10.47181473, 10.09646365,  # anchor box 3, width,  height
                    5.48531347,  8.11011331]) # anchor box 4, width,  height

LABELS = ['aeroplane',  'bicycle', 'bird',  'boat',      'bottle',
          'bus',        'car',      'cat',  'chair',     'cow',
          'diningtable','dog',    'horse',  'motorbike', 'person',
          'pottedplant','sheep',  'sofa',   'train',   'tvmonitor']


if __name__ == '__main__':
    ## Parse annotations
    train_image_folder = "datasets/VOCdevkit/VOC2012/JPEGImages/"
    train_annot_folder = "datasets/VOCdevkit/VOC2012/Annotations/"

    train_image, seen_train_labels = EDA.parse_annotation(train_annot_folder, train_image_folder, labels=LABELS)
    print("N train = {}".format(len(train_image)))

    EDA.show_sample_image(train_image)

    #batch generation of 16 images
    BATCH_SIZE       = 16
    GRID_H,  GRID_W  = 13 , 13
    IMAGE_H, IMAGE_W = 416, 416
    TRUE_BOX_BUFFER  = 50
    BOX = int(len(ANCHORS)/2)
    CLASS = len(LABELS)

    generator_config = {
        'IMAGE_H'         : IMAGE_H,
        'IMAGE_W'         : IMAGE_W,
        'GRID_H'          : GRID_H,
        'GRID_W'          : GRID_W,
        'BOX'             : BOX,
        'LABELS'          : LABELS,
        'ANCHORS'         : ANCHORS,
        'BATCH_SIZE'      : BATCH_SIZE,
        'TRUE_BOX_BUFFER' : TRUE_BOX_BUFFER,
    }

    train_batch_generator = EDA.SimpleBatchGenerator(train_image, generator_config, norm=EDA.normalize, shuffle=True)

    [x_batch,b_batch], y_batch = train_batch_generator.__getitem__(idx=3)
    print("x_batch: (BATCH_SIZE, IMAGE_H, IMAGE_W, N channels)           = {}".format(x_batch.shape))
    print("y_batch: (BATCH_SIZE, GRID_H, GRID_W, BOX, 4 + 1 + N classes) = {}".format(y_batch.shape))
    print("b_batch: (BATCH_SIZE, 1, 1, 1, TRUE_BOX_BUFFER, 4)            = {}".format(b_batch.shape))

    for irow in range(4, 7):
        print("-"*30)
        EDA.check_object_in_grid_anchor_pair(irow, generator_config, y_batch, LABELS)
        EDA.plot_image_with_grid_cell_partition(irow, x_batch, generator_config)
        EDA.plot_grid(irow, generator_config, y_batch, LABELS)
        plt.show()

    model = DKT19.darknet_model(IMAGE_H, IMAGE_W, TRUE_BOX_BUFFER, BOX, CLASS, GRID_H, GRID_W)
    #model.summary()

    mode = 'train'

    if mode == 'load':
        path_to_weight = "./models/trained_weight/yolov2.weights"
        weight_reader = RW.WeightReader(path_to_weight)
        print("all_weights.shape = {}".format(weight_reader.all_weights.shape))

        model = RW.load_weight(weight_reader, model, GRID_H, GRID_W)

        model = YT.compile_model(train_batch_generator, generator_config, BATCH_SIZE, model)
        pred.predict_batch_image(train_image_folder, model)

    if mode == 'train':
        model = YT.compile_model(train_batch_generator, generator_config, BATCH_SIZE, model)
        pred.predict_batch_image(train_image_folder, model)
