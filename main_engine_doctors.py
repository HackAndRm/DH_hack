# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 14:33:23 2017

@author: mskara
"""

from pandas import read_csv 
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from RF_model import ModelForest

import warnings
warnings.simplefilter(action = "ignore", category = DeprecationWarning)
import pymongo
import pickle

import os
os.chdir('C:\\Users\\mskara\\Desktop\\Hackhaton digital health')
### Now importing data locally: the following function can import data from Mongo:
def getResultsAsDF_MongoAggregate(self, pipeline, collection=''):
    """
    get data from MongoDB by aggregate    
    """
    if collection=='':
        collection=self.mongo_collection
        #returns results of query in pandas.DataFrame format
    try:
        client = pymongo.MongoClient(self.mongo_host)
        db = client[self.mongo_db]
        res = db[collection].aggregate(pipeline, allowDiskUse=True)
        df = pd.DataFrame(list(res))
        res.close()
        client.close()
    except:
        print("Exception thrown in getResults - Mongo Aggregate!")
        pass
    else:
        return df
        
def getDummies(dfz, col, minCtn = 10):
    '''
    function which create dummy variables 
    for the different categories
    '''    
    df2 = dfz.copy()
    df2['_id'] = 1
    df_aux = df2.groupby(col).aggregate({'_id':'count'}).reset_index() 
    df_aux = df_aux[df_aux._id>=minCtn]
    topColTypes = list(set(df_aux[col].values))
    dfz[col] = dfz.apply(lambda r: r[col] if r[col] in topColTypes else 'OTHER' , axis=1)
    dummies = pd.get_dummies(dfz[col], prefix=col) # +'_')
    
    return dummies, topColTypes

    
filename1 = '1_1_Production_data_doctors.csv' 
filename3 = '3_1_Production_data_doctors.csv' 

dataframe_profits = read_csv(filename1, encoding="ISO-8859-1") 
dataframe_cash    = read_csv(filename3, encoding="ISO-8859-1") 


def profits_data_preparation(dataframe_profits):
    '''
    working on the profits dataset (1_1)
    Takes the dataframe and rework values
    As output, the useful data used for training/testing
    
    '''
    array_profits = dataframe_profits.values 
    X_profits     = array_profits[:, 1:6] 
    Y_profits     = array_profits[:, 0] # prepare models 
    
    dumSex,      sex      = getDummies(dataframe_profits,"Sex")
    dumAge,      age      = getDummies(dataframe_profits,"Age")
    dumRegion,   region   = getDummies(dataframe_profits,"Region")
    dumCity,     city     = getDummies(dataframe_profits,"City_type")
    dumBusiness, business = getDummies(dataframe_profits,"Business_dimension")
    
    Xfull_profits = pd.concat([dumSex, dumAge, dumRegion, dumCity, dumBusiness], axis=1) 
    X_profits     = Xfull_profits.values
    names_profits = Xfull_profits.columns
    
    num_trees    = 100
    max_features = 6
    min_leafs    = 3
    regressor = RandomForestRegressor(min_samples_leaf = min_leafs, n_estimators = num_trees, max_features = max_features)
    regressor.fit(X_profits, Y_profits)

    test_profits = X_profits[1,]
    regressor.predict(test_profits)
    
    return Y_profits, X_profits, Xfull_profits, names_profits

def cash_data_preparation(dataframe_cash):
    '''
    working on the investment/cash required dataset (3_1)
    Takes the dataframe and rework values
    As output, the useful data used for training/testing
    '''
    array_cash = dataframe_cash.values 
    X_cash     = array_cash [:, 1:7] 
    Y_cash     = array_cash [:, 0] # prepare models 
  
    dumSex3,    sex       = getDummies(dataframe_cash, "Sex")
    dumAge3,    age       = getDummies(dataframe_cash, "Age")
    dumRegion3, region    = getDummies(dataframe_cash, "Region")
    dumCity3,   city      = getDummies(dataframe_cash, "City_type")

    Xfull_cash = pd.concat([dumSex3, dumAge3, dumRegion3, dumCity3, dataframe_cash["Business_duration"], 
                            dataframe_cash["Insurance_allowed"], dataframe_cash["Partnership"]], axis=1
                            ) 
    X_cash     = Xfull_cash.values
    names_cash = Xfull_cash.columns
    
    num_trees = 100
    max_features = 6
    min_leafs=3
    regressor_cash = RandomForestRegressor(min_samples_leaf=min_leafs,n_estimators=num_trees, max_features = max_features)
    regressor_cash.fit(X_cash, Y_cash)

    test_cash = X_cash[20,]
    regressor_cash.predict(test_cash)
    
    return Y_cash, X_cash, Xfull_cash, names_cash

    
Y_cash,    X_cash,    Xfull_cash,    names_cash    = cash_data_preparation   (dataframe_cash)
Y_profits, X_profits, Xfull_profits, names_profits = profits_data_preparation(dataframe_profits)

        
Forest_cash    = ModelForest(X_cash,    Y_cash)    # z_transform = True
Forest_profits = ModelForest(X_profits, Y_profits) # z_transform = True


###Saving on different files the inputs that will be used in the prediction model
output = open('Forest_cash_doctors.pkl', 'wb')
pickle.dump(Forest_cash, output)
output.close()

output = open('Forest_profits_doctors.pkl', 'wb')
pickle.dump(Forest_profits, output)
output.close()


NewQuery_cash    = Xfull_cash.loc[1]    * 0 #array_cash    [1, 1:7]
NewQuery_profits = Xfull_profits.loc[1] * 0 #array_profits [1, 1:6]  

output = open('NewQuery_cash_doctors.pkl', 'wb')
pickle.dump(NewQuery_cash, output)
output.close()

output = open('NewQuery_profits_doctors.pkl', 'wb')
pickle.dump(NewQuery_profits, output)
output.close()



