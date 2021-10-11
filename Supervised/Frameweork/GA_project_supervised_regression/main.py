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
* Predict the transaction value (continues value)
* Metrics :  - Root Mean Squared Error (RMSE): mean_squared_error(y_true, y_pred, squared=False) #give more importance to errors
             - r2_score : r2_score(y_true, y_pred)
             - MAE : mean_absolute_error(y_true, y_pred)
             - MEAE median_absolute_error(y_true, y_pred) : non sensitive to outliers

* Objectif : RMSE < 2 / r2_score > 10% / MAE < 0.5 / MEAE < 0.5

RMSE = 1.7554171528739868

----- Report -----
MSE_score = 3.0814893806042143
MAE_score = 0.33846755964698666
MEAE_score = 0.0
R2_score = 0.17010443348847526
'''

import EDA.EDA_Process as EDA_P
import Pre_processing.Pre_P as PPP
import Machine_learning.Algorithms as ML
import Prediction as PRED

'''
# 2. EDA (Exploratory Data Analysis)
'''
data = EDA_P.Read_data()
EDA_P.Form_analysis(data)
#EDA_P.Content_analysis(data)

'''
# 3. Pre-processing
'''
#data = EDA_P.Read_data()
#PPP.Pre_Processing_execution(data)

'''
# 4. Modelling
'''
#data = EDA_P.Read_data()
#ML.training_models(data)

'''
# 5. Make a prediction
'''
#PRED.makeAprediction()
