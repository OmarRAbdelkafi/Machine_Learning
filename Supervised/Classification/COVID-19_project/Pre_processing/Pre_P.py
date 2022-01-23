#coding:utf-8
'''
1. Simple preprocessing :
    - Clean NaN
    - Encoding
    - trainset/testset

2. test Model

3. Advenced preprocessing
    - Feature Selection and/or
    - Feature engineering and/or
    - Feature scalling and/or
    - delete outliers and
    - back to 2

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

def evaluation(model, X_train, X_test, y_train, y_test):
    from sklearn.metrics import f1_score, confusion_matrix, classification_report
    from sklearn.model_selection import learning_curve

    print('\n -----Evaluation of : \n',model)

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

def Decision_Forest(X_train, X_test, y_train, y_test):

    from sklearn.tree import DecisionTreeClassifier

    model = DecisionTreeClassifier(random_state = 0)

    evaluation(model, X_train, X_test, y_train, y_test)

    '''
    features Selection section
    '''
    #IFe = model.feature_importances_
    #print(pd.DataFrame(IFe, index=X_train.columns))
    #pd.DataFrame(IFe, index=X_train.columns).plot.bar()
    #plt.show()

    #select feature with more than 0.02 importance
    #print(pd.DataFrame(IFe[IFe>0.02], index=X_train.columns[IFe>0.02]))

    #selection = X_train.columns[IFe<0.02]
    #print(selection)

def RandomForest(X_train, X_test, y_train, y_test):

    from sklearn.ensemble import RandomForestClassifier

    from sklearn.pipeline import make_pipeline
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.preprocessing import PolynomialFeatures

    model = make_pipeline(SelectKBest(f_classif,k = 10),
                        RandomForestClassifier(random_state = 0))

    evaluation(model, X_train, X_test, y_train, y_test)


def basic_imputation(df):
    return df.dropna(axis = 0)

def onlyall_imputation(df):
    return df.dropna(how='all' , axis = 0)

def rows_imputation(df):
    retain_row = np.array([])
    for index, row in df.iterrows():
        #rows with more than 20% of missing values
        if (df.loc[index, :].isna().sum())/df.shape[1] >= 0.2:
            retain_row = np.append(retain_row,index)

    #print(retain_row.size)
    return df.drop(retain_row)


def fillna_limit(df):
    return df.fillna(-999)

def fillna_means(df):
    return df.fillna(df.mean())

def encoding(df):
    code = {'positive':1,
            'negative':0,
            'detected':1,
            'not_detected':0
            }

    for col in df.select_dtypes('object').columns:
        df.loc[:,col] = df[col].map(code)

    #print(print(df.dtypes.value_counts())) #verify that we don't have object anymore

    return df

def Pre_Processing_execution(df):
    '''
    # 1. Clean NaN columns
    '''
    missing_rate = (df.isna().sum())/df.shape[0]
    df = df[df.columns[missing_rate < 0.9]]
    df = df.drop('Patient ID', axis = 1)
    df = df.drop('Parainfluenza 2', axis = 1)
    #pd.set_option('display.max_row', df.shape[1])
    #print( ((df.isna().sum())/df.shape[0] ).sort_values() )

    '''
    feature selection after the result of the first learning
    '''

    df = df.drop(['Patient addmited to semi-intensive unit (1=yes, 0=no)',
    'Patient addmited to intensive care unit (1=yes, 0=no)', 'Hematocrit',
       'Hemoglobin', 'Red blood Cells', 'Lymphocytes',
       'Mean corpuscular volume (MCV)', 'Respiratory Syncytial Virus',
       'Influenza A', 'Parainfluenza 1', 'Rhinovirus/Enterovirus',
       'Coronavirus HKU1', 'Parainfluenza 3', 'Chlamydophila pneumoniae',
       'Adenovirus', 'Parainfluenza 4', 'Coronavirus229E', 'CoronavirusOC43',
       'Inf A H1N1 2009', 'Bordetella pertussis', 'Metapneumovirus'], axis = 1)

    '''
    # Encoding
    '''
    df = encoding(df)

    '''
    feature engineering : create new value or polynomial features
    '''

    '''
    # imputation of NaN values rows
    '''
    #df = basic_imputation(df)
    #df = onlyall_imputation(df)
    df = rows_imputation(df)
    #df = fillna_limit(df)
    df = fillna_means(df)

    '''
    # Creation target - features : train/test set
    '''
    y = df['SARS-Cov-2 exam result']
    X = df.drop('SARS-Cov-2 exam result', axis=1)
    #print(y)
    #print(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    #Verify the ratio of the target classes
    #print(y_train.value_counts())
    #print(y_test.value_counts())

    #Decision_Forest(X_train, X_test, y_train, y_test)
    #RandomForest(X_train, X_test, y_train, y_test)


    return X_train, X_test, y_train, y_test
