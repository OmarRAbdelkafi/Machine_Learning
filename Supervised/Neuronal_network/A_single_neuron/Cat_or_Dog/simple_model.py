#coding:utf-8

import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def predict(X, weight, bias):
    Activation = model(X, weight, bias)
    prediction = np.zeros(Activation.shape[0])
    for i in range(Activation.shape[0]):
        if Activation[i] >= 0.5:
            prediction[i] = 1
    return prediction

def log_loss(A, y):
    '''
        We use log_loss as cost fonction for our prediction
        L = 1/m sum(-y*log(A) - (1-y)*log(1-A))
    '''
    epsilon = 1e-15
    m = len(y) #Number of samples in our datasets
    L = 1/m * np.sum(-y * np.log(A + epsilon) - (1-y) * np.log(1 - A + epsilon))
    return L


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def model(X, W, b):
    '''
        Compute the prediction with the model
    '''
    #Matriciel product for pre-activation
    Z = X.dot(W) + b
    #Activation
    A = sigmoid(Z)

    return A

def initialization(X):
    W = np.random.randn(X.shape[1],1) #the weight vector to define randomly at the bigining w1 and w2
    b = np.random.randn(1)
    return W, b

def grad(A, X, y):
    '''
        Compute the gradient for the gradient descent
        derive(L%W) = 1/m * trans(X) * (A - y)
        derive(L%b) = 1/m * sum(A - y)
    '''
    m = len(y) #size datasets
    Trans = X.T #invers matrix X
    dW = 1/m * np.dot(Trans, A - y)
    db = 1/m * np.sum(A - y)
    return dW, db

def gradient_descent(X, y, W, b, learning_rate, n_iterations):

    sample = int(n_iterations/10)
    cost_history = np.zeros(sample+1) #learning curve
    accurancy_history = np.zeros(sample+1) #accurancy curve

    Activation = model(X, W, b)
    cost_history[0] = log_loss(Activation, y)

    y_pred = predict(X, W, b)
    accurancy_history[0] = accuracy_score(y, y_pred)
    k = 0

    from tqdm import tqdm
    for i in tqdm(range(0, n_iterations)):
        dW, db = grad(Activation, X, y)
        W = W - learning_rate * dW
        b = b - learning_rate * db
        Activation = model(X, W, b)
        if i%10 == 0:
            k+=1
            cost_history[k] = log_loss(Activation, y)
            y_pred = predict(X, W, b)
            accurancy_history[k] = accuracy_score(y, y_pred)

    return W, b, cost_history, accurancy_history

def artificial_neuron(X, y, learning_rate, n_iterations):
    show_details = True

    W, b = initialization(X)
    weight, bias, cost_history, accurancy_history = gradient_descent(X, y, W, b, learning_rate, n_iterations)

    if show_details:
        sample = int(n_iterations/10)
        plt.plot(range(sample+1), cost_history)
        plt.show()
        plt.plot(range(sample+1), accurancy_history)
        plt.show()

    return weight, bias
