#coding:utf-8
'''
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def Isolation_Forest_algo(X_train, y_train):

    from sklearn.ensemble import IsolationForest

    model = IsolationForest(contamination = 0.02) #2% of outliers
    model.fit(X_train)
    prediction = model.predict(X_train)

    retain_row = []
    for idx, value in np.ndenumerate(prediction):
        #rows with "-1" prediction means outliers
        if value == -1:
            retain_row.append(idx[0])

    #print(retain_row)

    return X_train.drop(X_train.index[retain_row]), y_train.drop(y_train.index[retain_row])

def PCA_algo(X_train, X_test):

    from sklearn.decomposition import PCA

    '''
    the right number of dimensions
    '''

    '''
    model = PCA(n_components=X_train.shape[1])
    X_reduced = model.fit_transform(X_train)

    V = model.explained_variance_ratio_ #les % de variances pour chaque compostante
    print(np.cumsum(V)) #faire la somme jusqu'à 100%, on observe qu'apartir de la 40eme dimension on arrive à 99%
    print('nombre de dimension à garder avec un seuil 99% :', np.argmax(np.cumsum(V) > 0.99)  )
    '''

    '''
    CC : nombre de dimension à réduire = 5
    '''

    model = PCA(n_components=5)
    X_train = pd.DataFrame(model.fit_transform(X_train))
    X_test = pd.DataFrame(model.fit_transform(X_test))

    return X_train, X_test

def Regression_Report(y_test, y_predict_test):
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import r2_score
    from sklearn.metrics import median_absolute_error

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
    from sklearn.metrics import mean_squared_error

    from sklearn.model_selection import learning_curve

    print('\n ----- RMSE Evaluation of : \n', model)

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

    N, train_score, val_score = learning_curve(model, X_train, y_train, train_sizes = np.linspace(0.1, 1.0, 10), cv=4, scoring='neg_root_mean_squared_error')

    print(N)
    plt.plot(N,train_score.mean(axis=1), label='train')
    plt.plot(N,val_score.mean(axis=1), label='validation')
    plt.xlabel('train_sizes')
    plt.legend()
    plt.show()

    '''
    Metric report
    '''

    Regression_Report(y_test, y_predict_test)

    print('-----End Evaluation-----\n')


def DTR(X_train, X_test, y_train, y_test, X):

    from sklearn.tree import DecisionTreeRegressor

    model = DecisionTreeRegressor(random_state = 0)

    evaluation(model, X_train, X_test, y_train, y_test)

    '''
    features Selection section
    '''

    IFe = model.feature_importances_
    print(pd.DataFrame(IFe, index=X.columns))
    pd.DataFrame(IFe, index=X.columns).plot.bar()
    plt.show()

    #select feature with less than 0.02 importance
    selection = X.columns[IFe<0.02]
    print('\n')
    print(selection)


# seting the function to show
def knowningData(df, data_type=object, limit=10): #seting the function with df,
    n = df.select_dtypes(include=data_type) #selecting the desired data type
    for column in n.columns: #initializing the loop
        print("##############################################")
        print("Name of column ", column, ': \n', "Uniques: ", df[column].unique()[:limit], "\n",
              " | ## Total nulls: ", (round(df[column].isnull().sum() / len(df[column]) * 100,2)),
              " | ## Total unique values: ", df.nunique()[column]) #print the data and % of nulls)
        # print("Percentual of top 3 of: ", column)
        # print(round(df[column].value_counts()[:3] / df[column].value_counts().sum() * 100,2))
        print("#############################################")

def Filling_Replacing_Values(df):
    # fillna numeric feature
    df['totals.pageviews'].fillna(df['totals.pageviews'].value_counts().index[0], inplace=True) #filling NA's with the most Frequent value
    df["totals.pageviews"] = df["totals.pageviews"].astype(int) # setting numerical to int

    df["totals.hits"] = df["totals.hits"].astype(int) # setting numerical to int

    df["totals.transactionRevenue"] = df["totals.transactionRevenue"].fillna(0.0).astype(float)

    #df.fillna(-999, inplace=True)
    #df.fillna(df.mean(), inplace=True)


    #object feature
    # --> Replace unknown values
    df.loc[df['channelGrouping'] == '(Other)', 'channelGrouping'] = np.nan
    df.loc[df['device.operatingSystem'] == '(not set)', 'device.operatingSystem'] = np.nan
    #df.loc[df['geoNetwork.continent'] == '(not set)', 'geoNetwork.continent'] = np.nan

    #df.loc[df['geoNetwork.region'] == '(not set)', 'geoNetwork.region'] = np.nan
    #df.loc[df['geoNetwork.region'] == 'not available in demo dataset', 'geoNetwork.region'] = np.nan

    #df.loc[df['geoNetwork.metro'] == '(not set)', 'geoNetwork.metro'] = np.nan
    #df.loc[df['geoNetwork.metro'] == 'not available in demo dataset', 'geoNetwork.metro'] = np.nan

    #df.loc[df['geoNetwork.city'] == 'not available in demo dataset', 'geoNetwork.city'] = np.nan

    #df.loc[df['geoNetwork.networkDomain'] == '(not set)', 'geoNetwork.networkDomain'] = np.nan
    #df.loc[df['geoNetwork.networkDomain'] == 'unknown.unknown', 'geoNetwork.networkDomain'] = np.nan

    #df.loc[df['trafficSource.campaign'] == '(not set)', 'trafficSource.campaign'] = np.nan

    #df.loc[df['trafficSource.medium'] == '(none)', 'trafficSource.medium'] = np.nan
    #df.loc[df['trafficSource.medium'] == '(not set)', 'trafficSource.medium'] = np.nan

    # --> fillna object feature
    #df['geoNetwork.city'].fillna("Other", inplace=True)

    df.dropna(axis = 0, inplace=True)
    #df.dropna(how='all' , axis = 0, inplace=True)

    return df #return the transformed dataframe


def Pre_Processing_execution(data):

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

    #knowningData(df, data_type= object)


    #. Creation target - features
    y = df['totals.transactionRevenue']
    X = df.drop('totals.transactionRevenue', axis=1)

    #5. train/test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    #knowningData(New_X, data_type= object)

    #Verify the Target
    '''
    print('train :')
    print("Target Min Value: ", y_train.min()) # printing the min value
    print("Target Mean Value: ", y_train.mean()) # mean value
    print("Target Median Value: ", y_train.median()) # median value
    print("Target Max Value: ", y_train.max()) # the max value

    print('test :')
    print("Target Min Value: ", y_test.min()) # printing the min value
    print("Target Mean Value: ", y_test.mean()) # mean value
    print("Target Median Value: ", y_test.median()) # median value
    print("Target Max Value: ", y_test.max()) # the max value
    '''

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

    # Reduction of dimension ? (PCA)
    #X_train, X_test = PCA_algo(X_train, X_test)

    # Normalization ?
    '''
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    '''
    
    # feature engineering : create new value or polynomial features ?
    '''
    from sklearn.preprocessing import PolynomialFeatures
    poly_features = PolynomialFeatures(degree = 2, include_bias = False)
    print(X_train.shape)
    X_train = poly_features.fit_transform(X_train)
    print(X_train.shape)
    '''

    #Linear_Regression(X_train, X_test, y_train, y_test, X)
    DTR(X_train, X_test, y_train, y_test, X)
