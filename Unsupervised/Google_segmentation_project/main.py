#coding:utf-8

#Project description
'''
*The GA dataset: https://www.kaggle.com/c/ga-customer-revenue-prediction/data

*Presenting the initial data:
    -Data Fields: fullVisitorIdv - A unique identifier for each user of the Google Merchandise Store.
    -channelGrouping - The channel via which the user came to the Store.
    -date - The date on which the user visited the Store.
    -device - The specifications for the device used to access the Store.
    -geoNetwork - This section contains information about the geography of the user.
    -sessionId - A unique identifier for this visit to the store.
    -socialEngagementType - Engagement type, either "Socially Engaged" or "Not Socially Engaged".
    -totals - This section contains aggregate values across the session.
    -trafficSource - This section contains information about the Traffic Source from which the session originated.
    -visitId - An identifier for this session. This is part of the value usually stored as the _utmb cookie. This is only unique to the user. For a completely unique ID, you should use a combination of fullVisitorId and visitId.
    -visitNumber - The session number for this user. If this is the first session, then this is set to 1.
    -visitStartTime - The timestamp (expressed as POSIX time).
'''

# 1. Define a mesurable objectif :
'''
* Create segments
* Metrics :  -

* Objectif :


----- Report -----

'''

import pandas as pd # to manipulate data frames
import numpy as np # to work with matrix

import EDA.EDA_Process as EDA_P
import Machine_learning.Algorithms as ML

'''
# 2. ETL (Extract Transform Load)
'''
#data = EDA_P.Extract()
#clean_data = EDA_P.Transform(data)
#EDA_P.Load_to_csv(clean_data)


'''
# 3. Modelling
'''
list_of_data = ["clean_data_small.csv", "clean_data_medium1.csv", "clean_data_medium2.csv", "clean_data_big.csv"]

for d in list_of_data:
    print("\n", d,"\n")
    data = pd.read_csv(d)
    ML.training_models_unsupervised(data)
