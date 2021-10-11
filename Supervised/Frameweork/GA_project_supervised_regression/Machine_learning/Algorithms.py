#coding:utf-8
'''
Notes :
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import median_absolute_error


def Regression_Report(y_test, y_predict_test):
    MSE_score = mean_squared_error(y_test, y_predict_test)
    MAE_score = mean_absolute_error(y_test, y_predict_test)
    MEAE_score = median_absolute_error(y_test, y_predict_test)
    R2_score = r2_score(y_test, y_predict_test)
    print('\n----- Report -----')
    print('MSE_score =', MSE_score)
    print('MAE_score =', MAE_score)
    print('MEAE_score =', MEAE_score)
    print('R2_score =', R2_score)

def evaluation(model, X_train, X_test, y_train, y_test):

    print('\n ----- RMSE Evaluation : \n')

    y_test = np.log1p(y_test)
    y_train = np.log1p(y_train)

    model.fit(X_train, y_train)

    y_predict_test = model.predict(X_test)
    y_predict_train = model.predict(X_train)

    train_score = mean_squared_error(y_train, y_predict_train, squared=False)
    print('train score =', train_score)

    test_score = mean_squared_error(y_test, y_predict_test, squared=False)
    print('test score =', test_score)

    '''
    learning curve visualisation
    '''
    '''
    N, train_score, val_score = learning_curve(model, X_train, y_train, train_sizes = np.linspace(0.1, 1.0, 10), cv=4, scoring='neg_root_mean_squared_error')

    print(N)
    plt.plot(N,train_score.mean(axis=1), label='train')
    plt.plot(N,val_score.mean(axis=1), label='validation')
    plt.xlabel('train_sizes')
    plt.legend()
    plt.show()
    '''
    '''
    Metric report
    '''
    Regression_Report(y_test, y_predict_test)

    print('-----End Evaluation-----\n')


def FocusOn_Ridge(Ridge_algo, X_train, X_test, y_train, y_test):
    hyper_params = {'ridge__alpha':[0.2, 0.5, 0.7, 1],
                    'ridge__random_state': [0],
                    'ridge__solver':['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'] }

    grid = GridSearchCV(Ridge_algo, hyper_params, scoring='neg_root_mean_squared_error', cv=4)

    grid.fit(X_train, y_train)

    #meilleurs parmetre :
    print(grid.best_params_)#{'ridge__alpha': 1, 'ridge__random_state': 0, 'ridge__solver': 'lsqr'}
    print(grid.best_score_)#2.04

    model = grid.best_estimator_

    evaluation(model, X_train, X_test, y_train, y_test)

def FocusOn_SVR(SVR, X_train, X_test, y_train, y_test):
    hyper_params = {'linearsvr__loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
                    'linearsvr__C': [0.2, 0.5, 1],
                    'linearsvr__random_state': [0],
                    'linearsvr__max_iter': [500, 1000, 2000]}

    grid = GridSearchCV(SVR, hyper_params, scoring='neg_root_mean_squared_error', cv=4)

    grid.fit(X_train, y_train)

    #meilleurs parmetre :
    print(grid.best_params_)#{'linearsvr__C': 0.2, 'linearsvr__loss': 'epsilon_insensitive', 'linearsvr__max_iter': 500, 'linearsvr__random_state': 0}
    print(grid.best_score_)#2.07

    model = grid.best_estimator_

    evaluation(model, X_train, X_test, y_train, y_test)

def FocusOn_NKR(NKR, X_train, X_test, y_train, y_test):
    hyper_params = {'kneighborsregressor__n_neighbors': [5, 7, 9, 11],
                    'kneighborsregressor__n_jobs': [-1]}

    grid = GridSearchCV(NKR, hyper_params, scoring='neg_root_mean_squared_error', cv=4)

    grid.fit(X_train, y_train)

    #meilleurs parmetre :
    print(grid.best_params_)#{'kneighborsregressor__n_jobs': -1, 'kneighborsregressor__n_neighbors': 11}
    print(grid.best_score_)#2.04

    model = grid.best_estimator_

    evaluation(model, X_train, X_test, y_train, y_test)

def FocusOn_DTR(DTR, X_train, X_test, y_train, y_test):
    hyper_params = {'decisiontreeregressor__random_state': [0],
                    'decisiontreeregressor__splitter': ['best', 'random']}

    grid = GridSearchCV(DTR, hyper_params, scoring='neg_root_mean_squared_error', cv=4)

    grid.fit(X_train, y_train)

    #meilleurs parmetre :
    print(grid.best_params_)#{'decisiontreeregressor__random_state': 0, 'decisiontreeregressor__splitter': 'best'}
    print(grid.best_score_)#2.35

    model = grid.best_estimator_

    evaluation(model, X_train, X_test, y_train, y_test)


def FocusOn_BOOST(BOOST, X_train, X_test, y_train, y_test):
    hyper_params = {'gradientboostingregressor__random_state': [0],
                    'gradientboostingregressor__loss': ['ls', 'lad', 'huber', 'quantile'],
                    'gradientboostingregressor__learning_rate': [0.05, 0.1, 0.2],
                    'gradientboostingregressor__n_estimators': [100, 200]}

    grid = GridSearchCV(BOOST, hyper_params, scoring='neg_root_mean_squared_error', cv=4)

    grid.fit(X_train, y_train)

    #meilleurs parmetre :
    print(grid.best_params_)#{'gradientboostingregressor__learning_rate': 0.2, 'gradientboostingregressor__loss': 'quantile', 'gradientboostingregressor__n_estimators': 200, 'gradientboostingregressor__random_state': 0}
    print(grid.best_score_)#3.32

    model = grid.best_estimator_

    evaluation(model, X_train, X_test, y_train, y_test)

def Isolation_Forest_algo(X_train, y_train):

    from sklearn.ensemble import IsolationForest

    model = IsolationForest(contamination = 0.01) #1% of outliers
    model.fit(X_train)
    prediction = model.predict(X_train)

    retain_row = []
    for idx, value in np.ndenumerate(prediction):
        #rows with "-1" prediction means outliers
        if value == -1:
            retain_row.append(idx[0])

    #print(retain_row)

    return X_train.drop(X_train.index[retain_row]), y_train.drop(y_train.index[retain_row])


def Filling_Replacing_Values(df):
    # fillna numeric feature
    df['totals.pageviews'].fillna(df['totals.pageviews'].value_counts().index[0], inplace=True) #filling NA's with the most Frequent value
    df["totals.pageviews"] = df["totals.pageviews"].astype(int) # setting numerical to int
    df["totals.hits"] = df["totals.hits"].astype(int) # setting numerical to int
    df["totals.transactionRevenue"] = df["totals.transactionRevenue"].fillna(0.0).astype(float)

    #object feature
    # --> Replace unknown values
    df.loc[df['channelGrouping'] == '(Other)', 'channelGrouping'] = np.nan
    df.loc[df['device.operatingSystem'] == '(not set)', 'device.operatingSystem'] = np.nan

    df.dropna(axis = 0, inplace=True)

    return df #return the transformed dataframe

def training_models(data):

    df = data.copy()

    '''
    # The copy past section for cleaning data ##########################
    '''
    print('\n')

    #1. Drop unique columns
    to_drop_unique_columns = ['socialEngagementType', 'device.browserVersion', 'device.browserSize', 'device.operatingSystemVersion',
     'device.mobileDeviceBranding', 'device.mobileDeviceModel', 'device.mobileInputSelector', 'device.mobileDeviceInfo',
     'device.mobileDeviceMarketingName', 'device.flashVersion', 'device.language', 'device.screenColors', 'device.screenResolution',
     'geoNetwork.cityId', 'geoNetwork.latitude', 'geoNetwork.longitude', 'geoNetwork.networkLocation', 'totals.visits', 'totals.bounces',
     'totals.newVisits', 'trafficSource.adwordsClickInfo.criteriaParameters', 'trafficSource.isTrueDirect',
     'trafficSource.adwordsClickInfo.adNetworkType', 'trafficSource.adwordsClickInfo.isVideoAd']

    df.drop(to_drop_unique_columns, axis=1, inplace=True)
    print("Total unique features dropped: ", len(to_drop_unique_columns))

    #2. Drop shape size columns
    to_drop_shape_columns = []

    df.drop(to_drop_shape_columns, axis=1, inplace=True)
    print("Total id features dropped: ", len(to_drop_shape_columns))

    #3. Drop columns
    to_drop_columns = ['trafficSource.keyword', 'trafficSource.referralPath', 'trafficSource.adwordsClickInfo.page',
    'trafficSource.adwordsClickInfo.slot', 'trafficSource.adwordsClickInfo.gclId', 'trafficSource.adContent',
    'sessionId', 'visitId', 'visitStartTime', 'date', 'trafficSource.campaign', 'geoNetwork.metro', 'geoNetwork.region',
    'geoNetwork.networkDomain', 'geoNetwork.city', 'trafficSource.medium', 'fullVisitorId']

    df.drop(to_drop_columns, axis=1, inplace=True)
    print("Total features dropped: ", len(to_drop_columns))

    #4. Drop rows
    to_drop_rows = []

    df.drop(to_drop_rows, axis=0, inplace=True)
    print("Total rows dropped: ", len(to_drop_rows))

    #5.
    drop_features = ['device.browser', 'device.isMobile', 'device.deviceCategory',
                         'geoNetwork.continent', 'geoNetwork.subContinent', '_year']

    df.drop(drop_features, axis=1, inplace=True)
    print("Total dropped features: ", len(drop_features))

    print('\n')

    #6.filling - replacing data
    df = Filling_Replacing_Values(df)

    print("Shape after dropping: ", df.shape)

    '''
    # ################### End copy past section for cleaning data ################################
    '''

    #. Creation target - features
    y = df['totals.transactionRevenue']
    X = df.drop('totals.transactionRevenue', axis=1)

    #5. train/test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    #. Encoding
    X_copy = X_train
    encoder = OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value = -999)
    encoder.fit(X_train)

    X_train = encoder.transform(X_train)
    X_train = pd.DataFrame(data = X_train, index = X_copy.index, columns = X_copy.columns)

    X_test = encoder.transform(X_test)

    #Outlier ? (isolation_forest)
    #X_train, y_train = Isolation_Forest_algo(X_train, y_train)
    #print('Rows after isolation forest:', X_train.shape[0])

    #X_train = encoder.inverse_transform(X_train)
    #X_test = encoder.inverse_transform(X_test)

    preprocessor = make_pipeline(OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value = -999))

    Ridge_algo = make_pipeline(preprocessor, StandardScaler(), Ridge(random_state = 0, alpha=.5))
    SVR = make_pipeline(preprocessor, StandardScaler(), LinearSVR(random_state = 0))
    NKR = make_pipeline(preprocessor, StandardScaler(), KNeighborsRegressor(n_jobs = -1))
    DTR = make_pipeline(preprocessor, StandardScaler(), DecisionTreeRegressor(random_state = 0))
    BOOST = make_pipeline(preprocessor, StandardScaler(), GradientBoostingRegressor(random_state = 0))

    dict_of_models = {'Ridge':Ridge_algo,
                      'SVR':SVR,
                      'NKR':NKR,
                      'DTR':DTR,
                      'BOOST':BOOST}
    '''
    for name, model in dict_of_models.items():
        print('Evaluation of -----',name)
        evaluation(model, X_train, X_test, y_train, y_test)
    '''

    from sklearn.metrics import make_scorer
    #define your own mse and set greater_is_better=False
    rmse = make_scorer(mean_squared_error, greater_is_better=False)

    # 1. Ridge
    #FocusOn_Ridge(Ridge_algo, X_train, X_test, y_train, y_test)

    # 2. SVR
    #FocusOn_SVR(SVR, X_train, X_test, y_train, y_test)

    # 3. NKR
    #FocusOn_NKR(NKR, X_train, X_test, y_train, y_test)

    # 4. DTR
    #FocusOn_DTR(DTR, X_train, X_test, y_train, y_test)

    # 5. BOOST
    #FocusOn_BOOST(BOOST, X_train, X_test, y_train, y_test)

    '''
    Best model :
    '''

    preprocessor = make_pipeline(OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value = -999))
    BestModel = make_pipeline(preprocessor, StandardScaler(), KNeighborsRegressor(n_jobs = -1, n_neighbors = 11))

    print('Evaluation of the best model')
    evaluation(BestModel, X_train, X_test, y_train, y_test)

    '''
    Save model :
    '''

    import pickle

    # save the model to disk
    filename = 'finalized_model.sav'
    pickle.dump(BestModel, open(filename, 'wb'))
