import matplotlib.pyplot as plt
import numpy as np
import Neural_Network.L_Layer_NN as LLNN

def simple_LR_tune(train_x, train_y, layers_dims, learning_rates, num_iterations):
    all_costs = {}
    for lr in learning_rates:
        print ("Training a model with learning rate: " + str(lr))
        learning_rate=lr
        _, all_costs[str(lr)] = LLNN.L_layer_model(train_x, train_y, layers_dims, learning_rate, num_iterations, print_cost=False, print_curve=False)
        print ('\n' + "-------------------------------------------------------" + '\n')

    for lr in learning_rates:
        plt.plot(all_costs[str(lr)], label=lr)

    plt.ylabel('cost')
    plt.xlabel('iterations (hundreds)')

    legend = plt.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()
