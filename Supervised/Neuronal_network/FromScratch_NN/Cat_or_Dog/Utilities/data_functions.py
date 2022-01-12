import numpy as np
import matplotlib.pyplot as plt
import h5py

def load_data():
    train_dataset = h5py.File('datasets/trainset.hdf5', "r")
    X_train = np.array(train_dataset["X_train"][:]) # your train set features
    y_train = np.array(train_dataset["Y_train"][:]) # your train set labels

    test_dataset = h5py.File('datasets/testset.hdf5', "r")
    X_test = np.array(test_dataset["X_test"][:]) # your train set features
    y_test = np.array(test_dataset["Y_test"][:]) # your train set labels

    return X_train, y_train, X_test, y_test

def Explore_data(X_train, y_train, X_test, y_test):
    #train
    print(X_train.shape)
    print(y_train.shape)
    print(np.unique(y_train, return_counts = True))

    #test
    print(X_test.shape)
    print(y_test.shape)
    print(np.unique(y_test, return_counts = True))

    plt.imshow(X_train[0])
    plt.show()
