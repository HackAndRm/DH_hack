# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 14:33:23 2017

@author: mskara
"""

from pandas import read_csv 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble         import RandomForestRegressor
from sklearn.cross_validation import KFold, cross_val_score, train_test_split
from sklearn.grid_search      import GridSearchCV

from datetime import datetime
import pymongo

import os
os.chdir('C:\\Users\\mskara\\Desktop\\Hackhaton digital health')
### Now importing data locally: the following function can import data from Mongo

def getResultsAsDF_MongoAggregate(self,pipeline, collection=''):
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

def getDummies(dfz,col,minCtn=10):
    # create dummy variables for goods type
    df2 = dfz.copy()
    df2['_id'] = 1
    df_aux = df2.groupby(col).aggregate({'_id':'count'}).reset_index() 
    df_aux = df_aux[df_aux._id>=minCtn]
    topColTypes = list(set(df_aux[col].values)) + ["OTHER"]
    dfz[col] = dfz.apply(lambda r: r[col] if r[col] in topColTypes else 'OTHER' , axis=1) 
    dummies = pd.get_dummies(dfz[col], prefix=col) #+'_')
    return dummies, topColTypes

def cleaning(strV):
        strV = strV.replace(',',' ')
        strV = strV.replace('â','a')
        strV = strV.replace('ä','ae')
        strV = strV.replace('ß','ss')
        strV = strV.replace('ö','oe')
        strV = strV.replace('ü','ue')
        strV = strV.replace('ó','o')
        strV = strV.replace('è','e')
        strV = strV.replace('é','e')
        strV = strV.replace('ê','e')
        strV = strV.replace('à','a') 
        return strV.lower() 
        
def getDummies(dfz,col,minCtn=10):
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
    dummies = pd.get_dummies(dfz[col], prefix=col) #+'_')
    return dummies, topColTypes


filename1 = '1_1_Production_data.csv' 
filename3 = '3_1_Production_data.csv' 

dataframe_profits = read_csv(filename1, encoding="ISO-8859-1") 
array_profits = dataframe_profits.values 
X_profits = array_profits[:,1:6] 
Y_profits = array_profits[:,0] # prepare models 

dumSex,      sex      = getDummies(dataframe_profits,"Sex")
dumAge,      age      = getDummies(dataframe_profits,"Age")
dumRegion,   region   = getDummies(dataframe_profits,"Region")
dumCity,     city     = getDummies(dataframe_profits,"City_Type")
dumBusiness, business = getDummies(dataframe_profits,"Business_dimension")

Xfull_profits = pd.concat([dumSex, dumAge, dumRegion, dumCity, dumBusiness], axis=1) 
X_profits     = Xfull_profits.values
Names_profits = Xfull_cash.columns


model = [] 
num_trees = 100 
max_features = 6
min_leafs=3
model.append(('RF', RandomForestRegressor(min_samples_leaf=min_leafs,n_estimators=num_trees, max_features=max_features))) 
scoring = 'neg_mean_squared_error' 

kfold = KFold(n=200,n_folds=10, random_state=7) 

#now doing very simply---- cv_results = cross_val_score(model, X_profits, Y_profits, cv=kfold, scoring=scoring) 
regressor = RandomForestRegressor(min_samples_leaf=min_leafs,n_estimators=num_trees, max_features=max_features)
regressor.fit(X_profits, Y_profits)

test_profits = X_profits[1,]
print (test)
regressor.predict(test)

results.append(cv_results)
names.append(name)
msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
print(msg) 



dataframe_cash = read_csv(filename3, encoding="ISO-8859-1") 
array_cash     = dataframe_cash.values 
X_cash = array_cash [:,1:7] 
Y_cash = array_cash [:,0] # prepare models 

dumSex3       = getDummies(dataframe_cash,"Sex")
dumAge3       = getDummies(dataframe_cash,"Age")
dumRegion3    = getDummies(dataframe_cash,"Region")
dumCity3      = getDummies(dataframe_cash,"City_Type")
dumBusiness3  = getDummies(dataframe_cash,"Business_dimension")
dumInsurance3 = getDummies(dataframe_cash,"Insurance_allowed")
dumPartner3   = getDummies(dataframe_cash,"Partnership")

Xfull_cash = pd.concat([dumSex3,dumAge3,dumRegion3,dumCity3,dumBusiness3,dataframe_cash["Business_duration"],dataframe_cash["Insurance_allowed"],dataframe_cash["Partnership"]], axis=1) 
X_cash     = Xfull_cash.values
Names_cash = Xfull_cash.columns

regressor_cash = RandomForestRegressor(min_samples_leaf=min_leafs,n_estimators=num_trees, max_features=max_features)
regressor_cash.fit(X_cash, Y_cash)

test_cash = X_cash[20,]
regressor_cash.predict(test_cash)

estimated_investment = regressor_cash.predict(test_cash)
## In R was estimated_investment <- predict(model_rf_cash, test_cash)

## Example_shown of test_cash
Business_dim="medium"
if(estimated_investment>200000):  Business_dim="big"
if(estimated_investment<120000):  Business_dim="small"
##need to modify different data from chatbot--> Business_dim estimated from the previous step
#test_profit
#Sex Age         Region City_Type Business_dimension
#1   w <40 Sachsen-Anhalt       big             medium

test_profit = sample_profit[1,-1]
####test_profit[,5]="small"
estimated_profit <- predict(model_rf_profit, test_profit)
net_salary       = 8000 * 13 * 0.75 #TaxRate * 
payback_time     = estimated_investment/(estimated_profit*0.7 - net_salary)

print(test_profit)
print(estimated_profit)
print(payback_time)





def mape_in_bounds_fun(y, y_hat):
    '''
    function used in ModelForest which calculates MAPE 
    in bounds over a specific trashold
    '''
    percentage_error = (y_hat - y) / y
    underpricing     = np.sum(percentage_error < -0.1) # underpricing error:
    overpricing      = np.sum(percentage_error > 0.2)  # overpricing error:
    overpricing     += 0.5 * np.sum((percentage_error > 0.1) & (percentage_error < 0.2))
    
    return 1 - (underpricing + overpricing) / len(y)

class ModelForest:
    '''
    Class that deals with RandomForest modelling
    Parameter settings can be improved at param_grid
    and at grid_searcher.

    '''
    def __init__(self, X,y, z_transform=False):
        self.z_transform = z_transform
        if z_transform:
            self.mean_y  = np.mean(y)
            self.std_y   = np.std(y)
            y = (y - self.mean_y) / self.std_y
        # keeping X_test and y_test to determine error quantiles
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1)
        
        now = datetime.now()
        print ("\nStarting to train this model:")
        print (now)
        # here we detrmine via crossvalidation the best parameters, but remember that we kept 10% of the datapoints out,
        # which can be used to determine the error quantiles
        max_depth_range    = [7, 10, 12, 15, 20]
        min_samples_leaf   = [max(10, int(0.005 * len(X)))]
        n_estimators_range = [5, 9, 13, 17]
        max_features_range = ["auto", "sqrt"]

        param_grid = dict(n_estimators     = n_estimators_range, 
                          min_samples_leaf = min_samples_leaf, 
                          max_depth        = max_depth_range,
                          max_features     = max_features_range)
        ### this is not working properly----- cv = KFold(n=30,n_folds=5,random_state=7,shuffle=True) 
        grid_searcher = GridSearchCV(RandomForestRegressor(), 
                                     param_grid = param_grid,
                                     cv      = 10,
                                     scoring = 'mean_squared_error', #accuracy #r2
                                     refit   = True, 
                                     n_jobs  = 2)
        best_forest = None
        best_score  = -1000000000000000
        for i in range(20):
            grid_searcher.fit(X_train, y_train)
            if grid_searcher.best_score_ > best_score:
                best_forest = grid_searcher.best_estimator_
                best_score  = grid_searcher.best_score_
        best_forest
        print("Best parameters: ")
        print(grid_searcher.best_params_)

        # now we use X_test and y_test to determine the error quantiles
        # and the mape_in_limits and mape
        pred   = best_forest.predict(X_test)
        errors = pred - y_test
        mape   = np.mean(np.abs((pred - y_test) / y_test))
        mape_in_bounds = mape_in_bounds_fun(y_test, pred)

        if z_transform:
        # back-transform the z-transformation
            errors = errors * self.std_y

        df_err = pd.DataFrame(columns = ['err','actual','pred'])
        df_err['err']    = errors
        df_err['actual'] = y_test
        df_err['pred']   = pred
        # remove outliers (if any)
        df_err = df_err[df_err.err >= np.percentile(errors, 1.5)]
        df_err = df_err[df_err.err <= np.percentile(errors, 98.5)]
        errors = df_err['err'].values
        mu    = np.mean(errors)
        sigma = np.std(errors)
        quantiles = {}
        for q in [60, 50, 40]: quantiles["%d" % q] = np.percentile(errors, 100.0 - q)

        # put train and test data back together and train the final model on the previously determined best parameters
        X = np.concatenate((X_train, X_test), axis = 0)
        y = np.concatenate((y_train, y_test), axis = 0)
        new_best_forest = RandomForestRegressor(**grid_searcher.best_params_)
        new_best_forest.fit(X, y)

        self.reg             = new_best_forest
        self.error_mu        = mu
        self.error_sigma     = sigma
        self.error_quantiles = quantiles
        self.mape            = mape
        self.mape_in_bounds  = mape_in_bounds

        print ("Finished training this model.")
        print ("Validation MAPE: " + str(mape))
        print ("Validation MAPE in bounds: " + str(mape_in_bounds))
        print ("\nTime used to train the model:")
        print (datetime.now()- now)

    def predict ( self , X ):
        '''
        function which predict each data
        Used to improve the capability of the model        
        '''
        return_dict = {"mu"        : self.error_mu,
                       "sigma"     : self.error_sigma,
                       "quantiles" : self.error_quantiles}
        y_predicted =  self.reg.predict(X)[0]

        if self.z_transform:
            return_dict["estimate"] = y_predicted * self.std_y + self.mean_y

        else:
            return_dict["estimate"] = y_predicted



Forest_cash    = ModelForest(X_cash,    Y_cash)    # z_transform = True
Forest_profits = ModelForest(X_profits, Y_profits) # z_transform = True




test_cash
Out[297]: 
array([ 0.,  0.,  1.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,
        0.,  0.,  1.,  0.,  1.,  1.,  0.])

'''
New Data processing and testing
To be implemented in a different file
'''

## We are assuming we have male customers
## However a bank should know the information and it can be modified 


Xfull_cash = pd.concat([dataframe_cash["Business_duration"],dumSex3,dumAge3,dumRegion3,dumCity3,dumBusiness3,dumInsurance3,dumPartner3], axis=1) 
X_cash     = Xfull_cash.values
NewQuery_cash    = array_cash    [1, 1:7]
NewQuery_profits = array_profits [1, 1:6]  


def processNewQuery(template_cash, template_profits,)
    '''
    take data from Mongo of a user of the Chatbot
    transform data in a format usable by the Random
    Forest model in order to provide estimates
    '''
def transform_cash(array):


if str(NewQuery_cash[0]) in self.sex:
   arg['Sex_%s' % str(NewQuery_cash[0])] = 1

if str(NewQuery_cash[1]) in self.age:
   arg['Age_%s' % str(NewQuery_cash[1])] = 1

if str(NewQuery_cash[2]) in self.region:
   arg['Region_%s' % str(NewQuery_cash[2])] = 1

if str(NewQuery_cash[3]) in self.region:
   arg['Region_%s' % str(NewQuery_cash[3])] = 1

if str(NewQuery_cash[2]) in self.region:
   arg['Region_%s' % str(NewQuery_cash[2])] = 1


dumSex,      sex      = getDummies(dataframe_profits,"Sex")
dumAge,      age      = getDummies(dataframe_profits,"Age")
dumRegion,   region   = getDummies(dataframe_profits,"Region")
dumCity,     city     = getDummies(dataframe_profits,"City_Type")
dumBusiness, business = getDummies(dataframe_profits,"Business_dimension")

def transform_profits(array):


'''
X_profits = array_profits[:,1:6] 
Y_profits = array_profits[:,0] # prepare models 

dumSex      = getDummies(dataframe_profits,"Sex")
dumAge      = getDummies(dataframe_profits,"Age")
dumRegion   = getDummies(dataframe_profits,"Region")
dumCity     = getDummies(dataframe_profits,"City_Type")
dumBusiness = getDummies(dataframe_profits,"Business_dimension")
'''




            
    
    