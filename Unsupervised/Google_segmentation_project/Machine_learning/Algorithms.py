#coding:utf-8
'''
Notes :
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder

from sklearn.cluster import KMeans

def details_segments(Segment, Total_transaction, Total_hits):
    Size = Segment.shape[0]
    print("Size = ", Size)

    Segment_transaction = Segment["totals.transactionRevenue"].sum()
    print("Market share on transaction = ", (Segment_transaction/Total_transaction)*100,"%")

    Market_share_hits = (Segment["totals.hits"].sum()/Total_hits)*100
    print("Market share on hits = ", Market_share_hits,"%")

    Segment_CRO_transaction = Segment[Segment["totals.transactionRevenue"]>0].shape[0]
    print("==> CRO transaction: ", (Segment_CRO_transaction/Size)*100)

    print("Segment mean pageview :", Segment["totals.pageviews"].mean())

    #Country
    count_country = Segment["geoNetwork.country"].value_counts(normalize=True)
    print("Most of the people are from", count_country.idxmax(),"with [", count_country.max()*100, "%]")

    #channelGrouping
    count_channelGrouping = Segment["channelGrouping"].value_counts(normalize=True)
    print("They use the channelGrouping", count_channelGrouping.idxmax(),"with [", count_channelGrouping.max()*100, "%]")

    #Browser
    count_browser = Segment["device.browser"].value_counts(normalize=True)
    print("They use the browser", count_browser.idxmax(),"with [", count_browser.max()*100, "%]")

    #time
    Segment_spending_money = Segment[Segment["totals.transactionRevenue"]>0]
    if Segment_spending_money.shape[0] == 0:
        print("no one is spending money in this segment")
    else:
        count_month = Segment_spending_money["_month"].value_counts(normalize=True)
        print("-->",count_month.max()*100,"%"," of the people spending money are doing it in month", count_month.idxmax())

        print("they connect between", Segment_spending_money["_visitHour"].min(),"and", Segment_spending_money["_visitHour"].max())

        #count_day = Segment_spending_money["_weekday"].value_counts(normalize=True)
        #print("here is the distribution for each weekday :")
        #print(count_day)


def K_means_algo(X):

    #Pour déteter le bon nombre de cluster il faut détercter la zone de coude, la ou l'inertia devient faible mais sans créer un cluster par point car ça n'a plus de sens sinon
    inertia = []
    step = 0.05 #5% of variation

    #Init
    model = KMeans(n_clusters = 1)
    model.fit(X)
    best_inertia = model.inertia_
    inertia.append(model.inertia_)

    max_range = 100
    k=2
    improve = True

    Variation = 0.0
    seuil = 0.0

    while improve and k <= max_range:
        model = KMeans(n_clusters = k)
        model.fit(X)

        Variation = best_inertia * step

        #if model.inertia_ <  best_inertia:
        if model.inertia_ < (best_inertia - Variation):
            k = k+1
            best_inertia = model.inertia_
            inertia.append(model.inertia_)
        else:
            improve = False

    #Entrainer le modele de K-Means clustering with k
    if(k > max_range):
        k = k-1

    model = KMeans(n_clusters = k)
    model.fit(X)
    predictions = model.predict(X)

    return k, predictions

def training_models_unsupervised(data):

    df = data.copy()

    #print(df.columns)

    X_train = df

    '''
    ###### Preprocessing
    '''
    #. Encoding
    encoder = OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value = -999)
    encoder.fit(X_train)

    X_train = encoder.transform(X_train)

    #. Normalization
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)

    X_train = pd.DataFrame(data = X_train, index = df.index, columns = df.columns)

    '''
    ###### Segmentation
    '''
    k, predictions = K_means_algo(X_train)
    New_df=df.assign(pred=predictions)

    print("\nGlobal information :")

    Total_transaction = New_df["totals.transactionRevenue"].sum()
    print("The total transactions is", Total_transaction,"$")
    Total_hits = New_df["totals.hits"].sum()
    print("The total hits is", Total_hits)
    Total_conversion_transaction = New_df[New_df["totals.transactionRevenue"]>0].shape[0]
    Global_CRO = (Total_conversion_transaction/New_df.shape[0])*100
    print("The global CRO on transaction is", Global_CRO)
    print("Mean pageview or all the dataset :", New_df["totals.pageviews"].mean())

    print("\nNumber of segments = ",k)

    CRO_best = -1
    Best_Seg = -1

    CRO_worst = np.inf
    worst_Seg = np.inf

    list_meet_creteria = []

    for i in range(k):
        Segment = New_df[New_df["pred"]==i]
        Size = Segment.shape[0]

        Segment_conversion_transaction = Segment[Segment["totals.transactionRevenue"]>0].shape[0]
        CRO = (Segment_conversion_transaction/Size)*100

        if CRO > CRO_best:
            Best_Seg = i
            CRO_best = CRO

        if CRO < CRO_worst:
            worst_Seg = i
            CRO_worst = CRO

        if CRO < 0.1:
            list_meet_creteria.append(i)

    print("\n################# Best Segment =", Best_Seg)
    Segment = New_df[New_df["pred"]==Best_Seg]
    details_segments(Segment, Total_transaction, Total_hits)
    print("################# End best Segment\n")

    '''
    print("\n################# Worst Segment =", worst_Seg)
    Segment = New_df[New_df["pred"]==worst_Seg]
    details_segments(Segment, Total_transaction, Total_hits)
    print("################# End worst Segment\n")
    '''

    print("*************Meet the criteria***************")
    for val in list_meet_creteria:
        print("\n#################Segment =", val)
        Segment = New_df[New_df["pred"]==val]
        details_segments(Segment, Total_transaction, Total_hits)
        print("################# End Segment\n")
