#coding:utf-8

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import Neural_Network.utils.initialization as NN_Init
import Neural_Network.utils.Forward.forward as NN_F
import Neural_Network.utils.Backward.backward as NN_B
import Neural_Network.utils.update as NN_Up
import Neural_Network.utils.evaluation as NN_Eval

def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations, print_cost, print_curve):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    costs = []                         # keep track of cost

    # Parameters initialization.
    parameters = NN_Init.initialize_parameters_deep(layers_dims)

    # Loop (gradient descent)
    for i in tqdm(range(0, num_iterations)):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = NN_F.L_model_forward(X, parameters)

        # Compute cost.
        cost = NN_Eval.compute_cost(AL, Y)

        # Backward propagation.
        grads = NN_B.L_model_backward(AL, Y, caches)

        # Update parameters.
        parameters = NN_Up.update_parameters(parameters, grads, learning_rate)

        # Print the cost every 100 iterations
        if print_cost and i % 100 == 0 or i == num_iterations - 1:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)

    if print_curve:
        # Plot learning curve (with costs)
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

    return parameters, costs
