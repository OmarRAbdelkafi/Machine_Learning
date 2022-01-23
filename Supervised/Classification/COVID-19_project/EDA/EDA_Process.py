#coding:utf-8
'''
form :
    - three type of variables : float/int/object(str)
    - A lot of NaN > 90% (more than the half)
    - The patient id is not interesting for as
    - for the rest three different groups :
        * no missing values for the target
        * no missing values for the admission of the patients
        * ~76% of missing value for test (bacterial, virus ...)
        * ~85% of missing value for the ratio of homoglobine (red, white...)
Content :
    - Only 10% of positifs in the target
    - some continues variable are standard
    - age quantile is not clear
    - quantitative variable are binary (almost all negative), only Rhinovirus/Enterovirus seem to be diffrent
    - It seem that platelets, monocytes and leukocytes are related to covid-19. beacause of the diffrent density in blood
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

def Read_data():
    data = pd.read_excel('dataset_covid.xlsx')
    '''
    print('\n')
    print(type(data))
    print('\n')
    print(data.head())
    print('\n')
    '''

    df = data.copy()

    return df

def Form_analysis(df):
    print("the target variable is (SARS-Cov-2 exam result)")
    print(f'Number of ligne : {df.shape[0]} / number of columns : {df.shape[1]}')

    print('\n types of variables :\n')
    print(df.dtypes.value_counts())

    #pd.set_option('display.max_row', df.shape[1]) #show all the row of the results
    #print(df.dtypes) # show all the variables

    #pd.set_option('display.max_column', df.shape[1]) #show all the colomn
    #print(df.head())
    #print(df.describe())

    plt.figure(figsize=(20,10))
    sns.heatmap(df.isna(), cbar = False) #to see in white where we have the missing data
    #plt.show()

    #print("\nPourcentage des valeur manquante par features :")
    #pd.set_option('display.max_row', df.shape[1]) #show all the colomn
    #print( ( (df.isna().sum())/df.shape[0] ).sort_values() )

    #features with less than 90% of missing values
    #retain_col = df.columns[(df.isna().sum())/df.shape[0] < 0.9]
    #print(retain_col)
    #print(retain_col.size)

    #values with less than 60% of missing values
    retain_row = np.array([])
    for index, row in df.iterrows():
        if (df.loc[index, :].isna().sum())/df.shape[1] < 0.6:
            retain_row = np.append(retain_row,index)

    print(retain_row)



def Content_analysis(df):

    #eliminate the values with more than 90% of missinf values :
    df = df[df.columns[(df.isna().sum())/df.shape[0] < 0.9]]
    df = df.drop('Patient ID', axis = 1)

    #analyse target
    print('Analyse target :')
    print(df['SARS-Cov-2 exam result'].value_counts(normalize = True)) #percentage with normalize = True
    print('\n')

    #continues variables
    for col in df.select_dtypes('float'):
        print(col)
    print('\n')

    #histograms :
    #sns.displot(df['Hematocrit']) #replace the name variable from col
    #plt.show()

    #discret variables
    for col in df.select_dtypes('int'):
        print(col)
    print('\n')

    #histograms :
    #sns.displot(df['Patient age quantile']) #replace the name variable from col
    #plt.show()

    #Qualitative variables
    for col in df.select_dtypes('object'):
        print(col, df[col].unique())
    print('\n')

    #pies :
    #df['Rhinovirus/Enterovirus'].value_counts().plot.pie()
    #plt.show()

    '''
    cc:
    'Parainfluenza 2' has only "not detected" attribute so we can remove it in preprocessing
    '''
    df = df.drop('Parainfluenza 2', axis = 1)


    '''
    Visualisation of features - target :
    '''
    #1. blood variable
    #sns.displot(data = df, x="Platelets", hue="SARS-Cov-2 exam result", stat="density", common_norm=False) #the last two attribute allow a normalization of each class separetly
    #plt.show()

    # 2. patient age
    #sns.displot(data = df, x="Patient age quantile", hue="SARS-Cov-2 exam result", multiple="dodge") #without the attribute stat we have a count
    #plt.show()

    # 3. viral variable (quantitative)
    #We can star with cross the tables
    CrossT = pd.crosstab(df['SARS-Cov-2 exam result'], df['Rhinovirus/Enterovirus'])
    print(CrossT)
    #sns.heatmap(CrossT, annot=True, fmt='d')
    #plt.show()

    '''
    Visualisation of features - features :
    '''
    #we try to see the correlation between variables to collect information
    #sns.heatmap(df.corr()) #the white color mean correlation and lenear relation between the variables
    #plt.show()

    '''
    Analyse NaN
    '''
    #we can see how many ligne we get if we eliminate the NaN ligne
    #print(df.dropna().count())
    #in this case we will have only 99 ligne wich is small, we decide to fill the NaN in this case

    '''
    test hypothesis : if Leukocytes, monocytes and platelets are sinificatively diffrent
    we use the student test here
    '''

    Neg_df = df[df['SARS-Cov-2 exam result'] == 'negative']
    Pos_df = df[df['SARS-Cov-2 exam result'] == 'positive']
    S_neg = Neg_df.sample(Pos_df.shape[0])#we chose a sample of negative equal to positive

    for col in df.select_dtypes('float'):
        alpha = 0.01 #if only 1% are same, the sample are significatively diffrent
        stat, p = ttest_ind(Neg_df[col].dropna(), Pos_df[col].dropna())
        if p < alpha:
            print(f'{col :-<50} signif diff')
        else:
            print(f'{col :-<50} NOT signif diff')
