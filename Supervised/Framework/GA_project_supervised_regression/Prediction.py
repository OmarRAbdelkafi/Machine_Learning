#coding:utf-8

'''
Input : features to predict
'''

import pandas as pd
import numpy as np
import pickle

def makeAprediction():

    wanted_zero = pd.DataFrame({'channelGrouping': ['Organic Search'], 'visitNumber': [1], 'device.operatingSystem': ['Windows'],
                        'geoNetwork.country': ['Austria'], 'totals.hits': [1], 'totals.pageviews':[1], 'trafficSource.source' : ['google'],
                        '_weekday': [4], '_day': [2], '_month': [9], '_visitHour': [10]},
                        index = [0])

    wanted_other = pd.DataFrame({'channelGrouping': ['Referral'], 'visitNumber': [4], 'device.operatingSystem': ['Macintosh'],
                            'geoNetwork.country': ['United States'], 'totals.hits': [67], 'totals.pageviews':[44], 'trafficSource.source' : ['google'],
                            '_weekday': [3], '_day': [26], '_month': [1], '_visitHour': [1]},
                            index = [0])

    #file name
    filename = 'finalized_model.sav'

    # load the model and encoder from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    y_pred_zero = loaded_model.predict(wanted_zero)
    y_pred_more = loaded_model.predict(wanted_other)
    #y_pred_t = loaded_model.predict(wanted_tablet)

    print('Zero ? ', y_pred_zero)
    print('191170000 ? ', y_pred_more)
