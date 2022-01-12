#coding:utf-8

#Project description
'''
The objectif of this work is to implement a sinlgle neuron
The dataset:
*Presenting the initial data:
    - We use a dataset of 1000 pictures 64*64 = 4096 features so we try to optimize Z = sum(Wi * ai) + b
'''

# 1. Define a mesurable objectif :

'''
* Predict the classification of cat or dog
* Metrics :  - Accurancy
* Objectif : Accurancy : 0.5 (since we use a linear model for a non linear problem)
'''
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

import Utilities.data_functions as UDF
import Utilities.preprocessing_functions as UPP
import Artificial_Neuron_class.Artificial_Neuron as AN
import Artificial_Neuron_class.Evaluations as Eval
import Artificial_Neuron_class.functions as AN_F

def Final_model_evaluation(X_train, y_train, X_test, y_test, weight, bias):
    #train prediction
    Activation_train = AN_F.model(X_train, weight, bias)
    loss_cost_train = Eval.log_loss(Activation_train, y_train)
    y_pred_train = AN_F.predict(X_train, weight, bias)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    print("the loss cost of the model train is: ", loss_cost_train)
    print("Accuracy of the model train: ", accuracy_train)

    #test prediction
    Activation_test = AN_F.model(X_test, weight, bias)
    loss_cost_test = Eval.log_loss(Activation_test, y_test)
    y_pred_test = AN_F.predict(X_test, weight, bias)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    print("the loss cost of the model test is: ", loss_cost_test)
    print("Accuracy of the model test: ", accuracy_test)


def Train_Dog_Cat_classification():
    show_details = True

    #trainset / testset
    X_train, y_train, X_test, y_test = UDF.load_data()

    #EDA
    if show_details:
        UDF.Explore_data(X_train, y_train, X_test, y_test)

    #Pre-processing
    X_train, X_test = UPP.flatten(X_train, X_test)
    X_train, X_test = UPP.Normalization_Min_Max(X_train, X_test)

    #param
    learning_rate = 0.01
    n_iterations = 4000

    Output_Neuron = AN.Artificial_Neuron(learning_rate, n_iterations)
    weight, bias, cost_history, accurancy_history = Output_Neuron.gradient_descent(X_train, y_train)

    if show_details:
        Output_Neuron.Analyse_Neuron(cost_history, accurancy_history)

    Final_model_evaluation(X_train, y_train, X_test, y_test, weight, bias)

if __name__ == '__main__':
    Train_Dog_Cat_classification()
