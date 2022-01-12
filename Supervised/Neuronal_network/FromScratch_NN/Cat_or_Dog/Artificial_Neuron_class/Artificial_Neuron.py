#coding:utf-8

import numpy as np
import Artificial_Neuron_class.Activations as Activ
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import Artificial_Neuron_class.Evaluations as Eval
import Artificial_Neuron_class.functions as F

class Artificial_Neuron:

    # instance attributes
    def __init__(self, learning_rate, n_iterations):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def gradient_descent(self, X, y):
        W, b = F.initialization(X)

        sample = int(self.n_iterations/10)
        cost_history = np.zeros(sample+1) #learning curve
        accurancy_history = np.zeros(sample+1) #accurancy curve

        Activation = F.model(X, W, b)
        cost_history[0] = Eval.log_loss(Activation, y)

        y_pred = F.predict(X, W, b)
        accurancy_history[0] = accuracy_score(y, y_pred)
        k = 0

        from tqdm import tqdm
        for i in tqdm(range(0, self.n_iterations)):
            dW, db = F.grad(Activation, X, y)
            W = W - self.learning_rate * dW
            b = b - self.learning_rate * db
            Activation = F.model(X, W, b)
            if i%10 == 0:
                k+=1
                cost_history[k] = Eval.log_loss(Activation, y)
                y_pred = F.predict(X, W, b)
                accurancy_history[k] = accuracy_score(y, y_pred)

        return W, b, cost_history, accurancy_history

    def Analyse_Neuron(self, cost_history, accurancy_history):
        sample = int(self.n_iterations/10)
        plt.plot(range(sample+1), cost_history)
        plt.show()
        plt.plot(range(sample+1), accurancy_history)
        plt.show()
