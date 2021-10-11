#coding:utf-8

# Necessary librarys
import os # it's a operational system library, to set some informations
import random # random is to generate random values

import pandas as pd # to manipulate data frames
import numpy as np # to work with matrix
from scipy.stats import ttest_ind

# library of datetime
from datetime import datetime

import matplotlib.pyplot as plt # to graphics plot
import seaborn as sns # a good library to graphic plots

import json # to convert json in df
from pandas import json_normalize # to normalize the json file

#Code to transform the json format columns in table
def json_read(df):

    columns = ['device', 'geoNetwork', 'totals', 'trafficSource'] # Columns that have json format

    # p is a fractional number to skiprows and read just a random sample of the our dataset.
    p = 0.1 # *** In this case we will use 10% of data set *** #

    #joining the [ path + df received]
    data_frame = df

    #Importing the dataset
    random.seed(0)
    df = pd.read_csv(data_frame,
                     converters={column: json.loads for column in columns}, # loading the json columns properly
                     dtype={'fullVisitorId': 'str'}, # transforming this column to string
                     skiprows=lambda i: i>0 and random.random() > p # Number of rows that will be imported randomly
                     )

    for column in columns: #loop to finally transform the columns in data frame
        #It will normalize and set the json to a table
        column_as_df = json_normalize(df[column])
        # here will be set the name using the category and subcategory of json columns
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        # after extracting the values, let drop the original columns
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)

    # Printing the shape of dataframes that was imported
    print(f"Loaded {os.path.basename(data_frame)}. Shape: {df.shape}")
    return df # returning the df after importing and transforming


# This function is to extract date features
def date_process(df):
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d") # seting the column as pandas datetime
    df["_weekday"] = df['date'].dt.weekday #extracting week day
    df["_day"] = df['date'].dt.day # extracting day
    df["_month"] = df['date'].dt.month # extracting day
    df["_year"] = df['date'].dt.year # extracting day
    df['_visitHour'] = (df['visitStartTime'].apply(lambda x: str(datetime.fromtimestamp(x).hour))).astype(int)

    return df #returning the df after the transformations

def Read_data():

    # We will import the data using the name and extension that will be concatenated with dir_path
    data = json_read("train.csv")
    data = date_process(data)

    return data

def details_missing_columns(df):
    total = df.isna().sum().sort_values(ascending = False) # getting the sum of null values and ordering
    percent = (df.isna().sum() / df.isna().count() * 100 ).sort_values(ascending = False) #getting the percent and order of null
    state_missing_value = pd.concat([total, percent], axis=1, keys=['Total', 'Percent']) # Concatenating the total and percent
    print("\nTotal missing value columns: ")
    print(state_missing_value) # Returning values of nulls different of 0

def missing_values_columns(df):

    hight_missing_value = [col for col in df.columns if (df[col].isna().sum() / df[col].isna().count() * 100) > 50] #missing column more than 50%
    print('\nNumber of high missing columns: ', len(hight_missing_value))
    print('high missing columns rate: ', hight_missing_value)

    # Automatic drop if needed
    #df.drop(hight_missing_value, axis=1, inplace=True)

def missing_values_rows(df):
    retain_row = []
    for index, row in df.iterrows():
        #rows with more than 50% of missing values
        if (df.loc[index, :].isna().sum())/df.shape[1] >= 0.5:
            retain_row.append(index)

    print('\nThe number of hight missing rows: ', len(retain_row))
    print('high missing rows rate: ', retain_row)

    # Automatic drop if needed
    #df.drop(retain_row, axis=0, inplace=True)

def unique_columns(df):

    # all columns where we have a unique value (constants)
    # It is useful because this columns give us none information
    discovering_consts = [col for col in df.columns if df[col].nunique() == 1]

    # printing the total of columns dropped and the name of columns
    print("\nNumber of columns with just one value: ", len(discovering_consts), "columns")
    print("Name of constant columns: \n", discovering_consts)

    # Automatic drop if needed
    #df.drop(discovering_consts, axis=1, inplace=True)

def Shape_size_columns(df):

    # all columns where we have the same size than shape like id
    # It is useful because this columns give us none information
    discovering_shape = [col for col in df.columns if df[col].nunique() == df.shape[0]]

    # printing the total of columns dropped and the name of columns
    print("\nNumber of columns with shape size: ", len(discovering_shape), "columns")
    print("Name of constant columns: \n", discovering_shape)

    # Automatic drop if needed
    #df.drop(discovering_shape, axis=1, inplace=True)


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
    df.loc[df['geoNetwork.continent'] == '(not set)', 'geoNetwork.continent'] = np.nan

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

def Form_analysis(data):

    df = data.copy()

    print(f'Number of ligne : {df.shape[0]} / number of columns : {df.shape[1]}')

    print('\n types of variables :\n')
    print(df.dtypes.value_counts())
    #pd.set_option('display.max_row', df.shape[1]) #show all the row of the results
    #print(df.dtypes) # show all the variables

    #pd.set_option('display.max_column', df.shape[1]) #show all the colomn
    #print(df.head())
    #print(df.describe())

    #plt.figure(figsize=(20,10))
    #sns.heatmap(df.isna(), cbar = False) #to see in white where we have the missing data
    #plt.show()

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
    'trafficSource.adwordsClickInfo.slot', 'trafficSource.adwordsClickInfo.gclId', 'trafficSource.adContent', 'fullVisitorId',
    'sessionId', 'visitId', 'visitStartTime', 'date', 'trafficSource.campaign', 'geoNetwork.metro', 'geoNetwork.region',
    'geoNetwork.networkDomain', 'geoNetwork.city', 'trafficSource.medium'  ]

    df.drop(to_drop_columns, axis=1, inplace=True)
    print("Total features dropped: ", len(to_drop_columns))

    #4. Drop rows
    to_drop_rows = []

    df.drop(to_drop_rows, axis=0, inplace=True)
    print("Total rows dropped: ", len(to_drop_rows))

    print('\n')

    #5.filling - replacing data
    df = Filling_Replacing_Values(df)

    print("Shape after dropping: ", df.shape)
    '''
    # ################### End copy past section for cleaning data ################################
    '''

    #unique columns
    unique_columns(df)

    #shape shape size columns
    Shape_size_columns(df)

    #missing value on columns
    missing_values_columns(df)

    #missing values on rows
    missing_values_rows(df)

    #print(df.head())

    details_missing_columns(df)

    '''
    print("\n Object --------")
    knowningData(df, data_type= object)
    print("\n int --------")
    knowningData(df, data_type= int)
    print("\n float --------")
    knowningData(df, data_type= float)
    print("\n bool --------")
    knowningData(df, data_type= bool)
    '''

def CalcOutliers(df_num):
    '''
    For numerical features
    '''
    # calculating mean and std of the array
    data_mean, data_std = np.mean(df_num), np.std(df_num)

    # seting the cut line to both higher and lower values
    # You can change this value
    cut = data_std * 3

    #Calculating the higher and lower cut values
    lower, upper = data_mean - cut, data_mean + cut

    # creating an array of lower, higher and total outlier values
    outliers_lower = [x for x in df_num if x < lower]
    outliers_higher = [x for x in df_num if x > upper]
    outliers_total = [x for x in df_num if x < lower or x > upper]

    # array without outlier values
    outliers_removed = [x for x in df_num if x > lower and x < upper]

    print('Identified lowest outliers: %d' % len(outliers_lower)) # printing total number of values in lower cut of outliers
    print('Identified upper outliers: %d' % len(outliers_higher)) # printing total number of values in higher cut of outliers
    print('Identified outliers: %d' % len(outliers_total)) # printing total number of values outliers of both sides
    print('Non-outlier observations: %d' % len(outliers_removed)) # printing total number of non outlier values
    print("Total percentual of Outliers: ", round((len(outliers_total) / len(outliers_removed) )*100, 4)) # Percentual of outliers in points

def Content_analysis(data):

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
    'trafficSource.adwordsClickInfo.slot', 'trafficSource.adwordsClickInfo.gclId', 'trafficSource.adContent', 'fullVisitorId',
    'sessionId', 'visitId', 'visitStartTime', 'date', 'trafficSource.campaign', 'geoNetwork.metro', 'geoNetwork.region',
    'geoNetwork.networkDomain', 'geoNetwork.city', 'trafficSource.medium'  ]

    df.drop(to_drop_columns, axis=1, inplace=True)
    print("Total features dropped: ", len(to_drop_columns))

    #4. Drop rows
    to_drop_rows = []

    df.drop(to_drop_rows, axis=0, inplace=True)
    print("Total rows dropped: ", len(to_drop_rows))

    print('\n')

    #5.filling - replacing data
    df = Filling_Replacing_Values(df)

    print("Shape after dropping: ", df.shape)
    '''
    # ################### End copy past section for cleaning data ################################
    '''

    '''
    Visualisation of features - target :
    '''
    '''

    # Printing some statistics of our data
    print("Target Min Value: ", df["totals.transactionRevenue"].min()) # printing the min value
    print("Target Mean Value: ", df["totals.transactionRevenue"].mean()) # mean value
    print("Target Median Value: ", df["totals.transactionRevenue"].median()) # median value
    print("Target Max Value: ", df["totals.transactionRevenue"].max()) # the max value

    sns.displot(data = df, x="totals.transactionRevenue")
    plt.show()
    '''

    '''
    Visualization of some features
    '''

    '''
    # seting the graph size
    plt.figure(figsize=(14,6))
    # Let explore the browser used by users
    sns.countplot(data = df, x = df[df['device.browser'].isin(df['device.browser'].value_counts()[:10].index.values)]['device.browser'], palette="hls") # It's a module to count the category's
    plt.title("TOP 10 Most Frequent Browsers", fontsize=20) # Adding Title and seting the size
    plt.xlabel("Browser Names", fontsize=16) # Adding x label and seting the size
    plt.ylabel("Count", fontsize=16) # Adding y label and seting the size
    plt.xticks(rotation=45) # Adjust the xticks, rotating the labels
    plt.show() #use plt.show to render the graph that we did above
    '''

    '''
    Detect numerical outliers
    '''

    '''
    n = df.select_dtypes(int) #selecting the desired data type (int or float)
    for column in n.columns: #initializing the loop
        print('\n----------->',column)
        CalcOutliers(df[column])
    '''

    '''
    Visualisation of features - target :
    '''

    '''
    # type
    knowningData(df, data_type= object) #set type (object, float, int , bool)

    sns.displot(data = df, x = "channelGrouping", y="totals.transactionRevenue")
    plt.xticks(rotation=45) # Adjust the xticks, rotating the labels
    plt.show() #use plt.show to render the graph that we did above
    '''

    '''
    Visualisation of features - features :
    '''

    '''
    #we try to see the correlation between variables to collect information (only for numerical)
    sns.heatmap(df.corr()) #the white color mean correlation and lenear relation between the variables
    plt.show()


    ## I will use the crosstab to explore two categorical values
    # At index I will use set my variable that I want analyse and cross by another
    crosstab_eda = pd.crosstab(index=df['channelGrouping'], normalize=True,
                           # at this line, I am using the isin to select just the top 5 of browsers
                           columns=df[df['device.browser'].isin(df['device.browser'].value_counts()[:5].index.values)]['device.browser'])
    # Ploting the crosstab that we did above
    crosstab_eda.plot(kind="bar",    # select the bar to plot the count of categoricals
                 figsize=(14,7), # adjusting the size of graphs
                 stacked=True)   # code to unstack
    plt.title("Channel Grouping % for which Browser", fontsize=20) # seting the title size
    plt.xlabel("The Channel Grouping Name", fontsize=18) # seting the x label size
    plt.ylabel("Count", fontsize=18) # seting the y label size
    plt.xticks(rotation=0)
    plt.show() # rendering
    '''

    '''
    Statistic test hypothesis if two categories : we use the student test here
    '''

    '''
    True_df = df[df['device.isMobile'] == 'True']
    False_df = df[df['device.isMobile'] == 'False']
    False_neg = False_df.sample(True_df.shape[0])#we chose a sample False equal to TRUE

    for col in df.select_dtypes('int'):
        alpha = 0.01 #if only 1% are same, the sample are significatively diffrent
        stat, p = ttest_ind(False_df[col].dropna(), True_df[col].dropna())
        if p < alpha:
            print(f'{col :-<50} signif diff')
        else:
            print(f'{col :-<50} NOT signif diff')
    '''
