#coding:utf-8

import numpy as np
from sklearn.datasets import make_blobs
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
    m = len(y) #Number of samples in our datasets
    return 1/m * np.sum(-y * np.log(A) - (1-y) * np.log(1-A))

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

    cost_history = np.zeros(n_iterations+1) #learning curve

    Activation = model(X, W, b)
    cost_history[0] = log_loss(Activation, y)

    for i in range(0, n_iterations):
        dW, db = grad(Activation, X, y)
        W = W - learning_rate * dW
        b = b - learning_rate * db
        Activation = model(X, W, b)
        cost_history[i+1] = log_loss(Activation, y)

    return W, b, cost_history

def artificial_neuron(X, y, learning_rate, n_iterations):
    show_details = False

    W, b = initialization(X)
    weight, bias, cost_history = gradient_descent(X, y, W, b, learning_rate, n_iterations)

    if show_details:
        print("last loss cost:", cost_history[n_iterations])
        plt.plot(range(n_iterations+1), cost_history)
        plt.show()

    return weight, bias

def train_classification():
    '''
        We use a dataset of n features so we try to optimize Z = sum(Wi * ai) +b
        for two features we optimize Z = w1 * a1 + w2 * a2 + b
        We use sigmoid as an activaion function for our neuron (sigmoid = 1 / (1 + exp(-z)))
    '''
    show_details = True

    N = 2
    X, y = make_blobs(n_samples=100, n_features=N, centers=2, random_state=0)
    #reshape y for (m,1)
    y = y.reshape(y.shape[0],1)

    if show_details:
        plt.scatter(X[:,0], X[:,1], c=y, cmap='summer')
        plt.show()

    #param
    learning_rate = 0.1
    n_iterations = 100

    weight, bias = artificial_neuron(X, y, learning_rate, n_iterations)

    final_activation = model(X, weight, bias)
    loss_cost = log_loss(final_activation, y)
    y_pred = predict(X, weight, bias)
    accuracy = accuracy_score(y, y_pred)

    if show_details:
        print(y_pred)
        print("the loss cost of the model is: ", loss_cost)
        print("Accuracy of the model: ", accuracy)

        #decision boundray W0*x1 + W1*x2 + b = 0 <=> x2 = (-(w0*x1) - b) / w1
        x1 = np.linspace(-1, 5, 100)
        x2 = (-weight[0] * x1 - bias) / weight[1]

        plt.scatter(X[:,0], X[:,1], c=y, cmap='summer')
        #plt.scatter(X[:,0], X[:,1], c=y_pred, cmap='summer')
        plt.plot(x1, x2, c="red", lw=3)
        plt.show()


if __name__ == '__main__':
    train_classification()
