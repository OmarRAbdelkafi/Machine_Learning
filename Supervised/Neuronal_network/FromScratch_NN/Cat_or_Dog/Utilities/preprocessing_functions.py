def flatten(X_train, X_test):
    #flaten the images
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])

    return X_train, X_test

def MinMax(data, Min_data, Max_data):
    data = (data - Min_data)/(Max_data-Min_data)
    return data

def Normalization_Min_Max(X_train, X_test):
    #normalize with MinMax, we use min and max of the train set and we apply it on test set to get correct normalization
    Min_data = X_train.min()
    Max_data = X_train.max()
    X_train = MinMax(X_train, Min_data, Max_data)
    X_test = MinMax(X_test, Min_data, Max_data)

    return X_train, X_test
