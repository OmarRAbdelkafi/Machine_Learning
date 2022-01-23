#Evaluation functions
import numpy as np

def log_loss(A, y):
    '''
        We use log_loss as cost fonction for our prediction
        L = 1/m sum(-y*log(A) - (1-y)*log(1-A))
    '''
    epsilon = 1e-15
    m = len(y) #Number of samples in our datasets
    L = 1/m * np.sum(-y * np.log(A + epsilon) - (1-y) * np.log(1 - A + epsilon))
    return L
