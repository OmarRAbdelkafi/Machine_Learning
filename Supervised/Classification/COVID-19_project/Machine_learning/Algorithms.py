#coding:utf-8
'''
Notes :
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import f1_score, confusion_matrix, classification_report, recall_score
from sklearn.model_selection import learning_curve

from sklearn.model_selection import GridSearchCV

def evaluation(model, X_train, X_test, y_train, y_test):
    print('\n')

    model.fit(X_train, y_train)

    y_predict_test = model.predict(X_test)
    y_predict_train = model.predict(X_train)

    train_score = f1_score(y_train, y_predict_train)
    print('train score =', train_score)

    test_score = f1_score(y_test, y_predict_test)
    print('test score =', test_score)

    print(confusion_matrix(y_test, y_predict_test))
    print(classification_report(y_test, y_predict_test))

    '''
    learning curve visualisation
    '''


    N, train_score, val_score = learning_curve(model, X_train, y_train, train_sizes = np.linspace(0.1, 1.0, 10), cv=4, scoring='f1')

    print(N)
    plt.plot(N,train_score.mean(axis=1), label='train')
    plt.plot(N,val_score.mean(axis=1), label='validation')
    plt.xlabel('train_sizes')
    plt.legend()
    plt.show()


    print('-----End Evaluation-----\n')

def final_model(model, X_test, threshold=0):
    return model.decision_function(X_test) > threshold

def training_models(X_train, X_test, y_train, y_test):

    preprocessor = make_pipeline(SelectKBest(f_classif,k = 10))

    RandomForest = make_pipeline(preprocessor, RandomForestClassifier(random_state = 0))
    AdaBoost = make_pipeline(preprocessor, AdaBoostClassifier(random_state = 0))

    #the following models need standardization not like the two first based on decision forest
    SVM = make_pipeline(preprocessor, StandardScaler(), SVC(random_state = 0))
    KNN = make_pipeline(preprocessor, StandardScaler(), KNeighborsClassifier())

    dict_of_models = {'RandomForest':RandomForest,
                      'AdaBoost':AdaBoost,
                      'SVM':SVM,
                      'KNN': KNN}
    '''
    for name, model in dict_of_models.items():
        print('Evaluation of -----',name)
        evaluation(model, X_train, X_test, y_train, y_test)
    '''

    # - We focus on adaboost and SVM regarding the results

    # 1. SVM
    '''
    hyper_params = {'pipeline__selectkbest__k':range(5,10),
                    'svc__C':[1000,2000]}

    grid = GridSearchCV(SVM, hyper_params, scoring='f1', cv=4)

    grid.fit(X_train, y_train)

    #meilleurs parmetre :
    print(grid.best_params_)
    print(grid.best_score_)

    model = grid.best_estimator_

    evaluation(model, X_train, X_test, y_train, y_test)
    '''

    # 2. Adaboost

    hyper_params = {'pipeline__selectkbest__k':range(5,10),
                    'adaboostclassifier__n_estimators':[50,100,150]}

    grid = GridSearchCV(AdaBoost, hyper_params, scoring='f1', cv=4)
    #if we have many hyper parameters, we can use 'RandomizedSearchCV'
    #from sklearn.model_selection import RandomizedSearchCV
    #grid = RandomizedSearchCV(AdaBoost, hyper_params, scoring='f1', cv=4, n_iter=40) #with n_iter the number of random combination

    grid.fit(X_train, y_train)

    #meilleurs parmetre :
    print(grid.best_params_)
    print(grid.best_score_)

    model = grid.best_estimator_

    #evaluation(model, X_train, X_test, y_train, y_test)

    '''
    Last part of this problem :
    Precision Recall Curve
    '''

    '''
    from sklearn.metrics import precision_recall_curve

    precision, recall, threshold = precision_recall_curve(y_test, grid.best_estimator_.decision_function(X_test))

    plt.figure()
    plt.plot(threshold, precision[:-1], label = 'precision')
    plt.plot(threshold, recall[:-1], label = 'recall')
    plt.legend()
    plt.show()

    #the cross of the graphics show that the best threshold for us is -0.05
    '''

    y_pred = final_model(grid.best_estimator_, X_test, threshold = -0.05)

    f1_test_score = f1_score(y_test, y_pred)
    print('f1 test score =', f1_test_score)

    recall_test_score = recall_score(y_test, y_pred)
    print('recall test score =', recall_test_score)
