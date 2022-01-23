import numpy as np
import Artificial_Neuron_class.Activations as Activ

def model(X, W, b):
    '''
        Compute the prediction with the model
    '''
    #Matriciel product for pre-activation
    Z = X.dot(W) + b
    #Activation
    A = Activ.sigmoid(Z)

    return A

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

def initialization(X):
    W = np.random.randn(X.shape[1],1) #the weight vector to define randomly at the bigining w1 and w2
    b = np.random.randn(1)
    return W, b

def predict(X, weight, bias):
    Activation = model(X, weight, bias)
    prediction = np.zeros(Activation.shape[0])
    for i in range(Activation.shape[0]):
        if Activation[i] >= 0.5:
            prediction[i] = 1
    return prediction
