#coding:utf-8

#Project description
'''
This project is around the Covid-19 virus
- datasets sur kaggle
- Diagnosis of Covid-19 and its clinical spectrum
'''

# 1. Define a mesurable objectif :
'''
* Predict if a person is infected or not
* the classes are probably not equilibred so we didn't chose the acurancy metric
* Metric :  - Precision : True positif/ (True positif + false positif) --> reduce the false positif rate
            - Recall : True negative/ (True positif + False negatif) --> reduce the false négatifs rate
            We use the confusion matrix Metric
            - Score F1 : Precision/recall

* Objectif : F1 >= 50% and recall >=70% (not rushed)
'''

# 2. EDA (Exploratory Data Analysis)
'''
Objectif : understanding the data to create a modeling strategy

* Forme analysis :
    - Identification of the target
    - Shape (ligne/column)
    - Variable types (discret, continues)
    - missing values
    - ...

* Content analysis :
    - Eliminate the not importent features to simplify if possible
    - Visualisation of the target (Histogramme if continue / Boxplot if discret)
    - Understanding the different variables (research in internet to understand the variale)
    - Identification of the outliers variable to eliminate them in preprocessing
    - Visualisation of the relation between features - target (Histogramme / Boxplot)
    - ...
'''

# 3. Pre-processing
'''
Pre-processing notes :

Objectif : clean and transform the data to prepare it for the ML model
To do at begining :
    - Create the train set / tests set
    - Eliminate of the NaN : dropna(), imputation, Empty colonnes
    - Encoding
    - Feature Selection (no interesting feature of repetitive features)
    - Feature engineering (combine variable or add new information from professionels/ transform linear variable on polynomial...)
    - Try to use PCA to reduce the dimension and keep 99% variation
    - Feature scalling (Normalisation : MinMax or standar scaler
    - Pipelines
    - ...

To do after the first model :
    - Delete the outliers (not good for the model) : isolation Forest
    - Feature engineering again
    - ...

'''

# 4. Modelling
'''
Modelling notes :

Objectif : develop a model of ML to rush the objectif and metrics
To do :
    - Define the evaluation function of the model (Supervised/Unsupervised : regression, classification)
    - train different model
    - Optimization with GridSearchCV
    - CrossValidation with GridSearchCV
    - Evaluation system to compare models and make algorithm decision
    - Learning Curve and make decision to recolt or not new data
    - If necessary go back to EDA and Pre-processing
'''

import EDA.EDA_Process as EDA_P
import Pre_processing.Pre_P as PPP
import Machine_learning.Algorithms as ML

df = EDA_P.Read_data()
#EDA_P.Form_analysis(df)
#EDA_P.Content_analysis(df)

X_train, X_test, y_train, y_test = PPP.Pre_Processing_execution(df)

ML.training_models(X_train, X_test, y_train, y_test)
