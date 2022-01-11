#coding:utf-8

# To activate this environment, use
#
#     $ conda activate tf
#
# To deactivate an active environment, use
#
#     $ conda deactivate


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import StandardScaler

import sys

def train_classification():
    show_details = True

    # Load data (Fashio MNIST)
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (images, targets), (images_test, targets_test) = fashion_mnist.load_data()

    # Get only a subpart of the dataset
    images = images[:10000]
    targets = targets [:10000]

    # Reshape the dataset and convert to float
    images = images.reshape(-1, 784)
    images = images.astype(float)
    images_test = images_test.reshape(-1, 784)
    images_test = images_test.astype(float)

    scaler = StandardScaler()
    images = scaler.fit_transform(images)
    images_test = scaler.transform(images_test)

    targets_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    if show_details:
        print(images.shape)
        print(targets.shape)
        # Plot one image
        plt.imshow(np.reshape(images[8], (28, 28)), cmap="binary")
        plt.title(targets_names[targets[8]])
        plt.show()

    # Flatten
    model = tf.keras.models.Sequential()

    # Add the layers
    model.add(tf.keras.layers.Dense(256, activation="relu"))
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))

    if show_details:
        model_output = model.predict(images[0:1])
        print(model_output, targets[0:1])
        print(model.summary())

    # Compile the model
    model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
    history = model.fit(images, targets, epochs=10, validation_split=0.2)

    if show_details:
        loss_curve = history.history["loss"]
        acc_curve = history.history["accuracy"]

        loss_val_curve = history.history["val_loss"]
        acc_val_curve = history.history["val_accuracy"]

        plt.plot(loss_curve, label="Train")
        plt.plot(loss_val_curve, label="Val")
        plt.legend(loc='upper left')
        plt.title("Loss")
        plt.show()

        plt.plot(acc_curve, label="Train")
        plt.plot(acc_val_curve, label="Val")
        plt.legend(loc='upper left')
        plt.title("Accuracy")
        plt.show()

    loss, acc = model.evaluate(images_test, targets_test)

    if show_details:
        print("Test Loss", loss)
        print("Test Accuracy", acc)

if __name__ == '__main__':
    train_classification()
