
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import yolo_loss as YL
from tensorflow import keras
import numpy as np

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import SGD, Adam, RMSprop

def compile_model(train_batch_generator, generator_config, BATCH_SIZE, model):
    dir_log = "logs/"
    try:
        os.makedirs(dir_log)
    except:
        pass

    generator_config['BATCH_SIZE'] = BATCH_SIZE

    early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=3,mode='min',verbose=1)

    checkpoint = ModelCheckpoint('model_trained2.h5', monitor='loss', verbose=1, save_best_only=True, mode='min', save_freq=1)

    optimizer = Adam(learning_rate=0.5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss=YL.custom_loss, optimizer=optimizer)

    tf.config.run_functions_eagerly(True)

    model.fit(train_batch_generator,
            steps_per_epoch  = len(train_batch_generator),
            epochs           = 10,
            verbose          = 1,
            callbacks        = [early_stop, checkpoint],
            max_queue_size   = 3)

    return model
