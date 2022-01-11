#coding:utf-8

import numpy as np

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import h5py
import simple_model as SM

def MinMax(data, Min_data, Max_data):
    data = (data - Min_data)/(Max_data-Min_data)
    return data

def load_data():
    train_dataset = h5py.File('datasets/trainset.hdf5', "r")
    X_train = np.array(train_dataset["X_train"][:]) # your train set features
    y_train = np.array(train_dataset["Y_train"][:]) # your train set labels

    test_dataset = h5py.File('datasets/testset.hdf5', "r")
    X_test = np.array(test_dataset["X_test"][:]) # your train set features
    y_test = np.array(test_dataset["Y_test"][:]) # your train set labels

    return X_train, y_train, X_test, y_test

def Final_model_evaluation(X_train, y_train, X_test, y_test, weight, bias):
    #train prediction
    Activation_train = SM.model(X_train, weight, bias)
    loss_cost_train = SM.log_loss(Activation_train, y_train)
    y_pred_train = SM.predict(X_train, weight, bias)
    accuracy_train = SM.accuracy_score(y_train, y_pred_train)
    print("the loss cost of the model train is: ", loss_cost_train)
    print("Accuracy of the model train: ", accuracy_train)

    #test prediction
    Activation_test = SM.model(X_test, weight, bias)
    loss_cost_test = SM.log_loss(Activation_test, y_test)
    y_pred_test = SM.predict(X_test, weight, bias)
    accuracy_test = SM.accuracy_score(y_test, y_pred_test)
    print("the loss cost of the model test is: ", loss_cost_test)
    print("Accuracy of the model test: ", accuracy_test)


def train_classification():
    '''
        We use a dataset of 1000 pictures 64*64 = 4096 features so we try to optimize Z = sum(Wi * ai) + b
    '''
    show_details = False

    X_train, y_train, X_test, y_test = load_data()

    if show_details:
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

    #flaten the images
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])

    #normalize with MinMax, we use min and max of the train set and we apply it on test set to get correct normalization
    Min_data = X_train.min()
    Max_data = X_train.max()
    X_train = MinMax(X_train, Min_data, Max_data)
    X_test = MinMax(X_test, Min_data, Max_data)

    #param
    learning_rate = 0.01
    n_iterations = 4000

    weight, bias = SM.artificial_neuron(X_train, y_train, learning_rate, n_iterations)

    Final_model_evaluation(X_train, y_train, X_test, y_test, weight, bias)

if __name__ == '__main__':
    train_classification()
