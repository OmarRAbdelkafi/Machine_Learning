#coding:utf-8

#Project description
'''
*Deep Neural Network for Image Classification:

*Presenting the initial data:
dataset ("data.h5") containing:
    - a training set of 'm_train' images labelled as cat (1) or non-cat (0)
    - a test set of 'm_test' images labelled as cat and non-cat
    - each image is of shape (num_px, num_px, 3) where 3 is for the 3 channels (RGB).
    - num_px = 64

* Lvl humain < 0.5% accurancy
'''

# 1. Define a mesurable objectif :
'''
* Predict the classification of cat images
* Metrics :  - Accurancy
* Objectif : Accurancy > 98%
'''

import EDA.EDA_process as EDA_P
import Pre_processing.Pre_P as PPP
import Neural_Network.L_Layer_NN as LLNN
import prediction as Pr
import Neural_Network.Tuning.Learning_rate_tune as LRT


'''
# 2. EDA (Exploratory Data Analysis)
'''
train_x_orig, train_y, test_x_orig, test_y, classes = EDA_P.load_data()
#EDA_P.Form_analysis(train_x_orig, train_y, test_x_orig, test_y, classes)
#EDA_P.Content_analysis(train_x_orig, train_y, test_x_orig, test_y, classes)

'''
# 3. Pre-processing
'''
train_x, test_x = PPP.Pre_Processing_execution(train_x_orig, train_y, test_x_orig, test_y, classes)

'''
# 4. Modelling
'''
### CONSTANTS ###
input_layer_size = train_x.shape[0]
layers_dims = [input_layer_size, 20, 7, 5, 1] #  4-layer model
learning_rate = 0.0075
num_iterations = 500

parameters, costs = LLNN.L_layer_model(train_x, train_y, layers_dims, learning_rate, num_iterations, print_cost=False, print_curve=False)


'''
# 5. Prediction and Analysis
'''
pred_train = Pr.predict(train_x, train_y, parameters)
pred_test = Pr.predict(test_x, test_y, parameters)
#Pr.print_mislabeled_images(classes, test_x, test_y, pred_test)

#New prediction
num_px = train_x_orig.shape[1]
my_image = "my_image.jpg" # change this to the name of your image file
fname = "images/" + my_image
my_label_y = [1] # the true class of your image (1 -> cat, 0 -> non-cat)
Pr.New_prediction(num_px, parameters, classes, fname, my_label_y)

'''
# 6. Tuning
    A- Go as close as possible from lvl humain {if underfitting with high variance} (learning rate, optimization algorithm, size NN, architecture, ...)
    B- If overfitting with high bias {more data id possible else :L2 regularizations, dropout, change architecture and/or optimization algorithm}
    C- Else mismatch data, change data, cost function (analyse the results)
'''

#learning_rates = [0.01, 0.001, 0.0001]
#LRT.simple_LR_tune(train_x, train_y, layers_dims, learning_rates, num_iterations)
