#coding:utf-8

def Pre_Processing_execution(train_x_orig, train_y, test_x_orig, test_y, classes):
    # Reshape the training and test examples
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten/255.
    test_x = test_x_flatten/255.

    #print ("train_x's shape: " + str(train_x.shape))
    #print ("test_x's shape: " + str(test_x.shape))

    return train_x, test_x
