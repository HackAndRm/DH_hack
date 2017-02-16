# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 15:45:49 2017

@author: mskara
"""
import pickle
from RF_model import ModelForest
import numpy as np
import pandas as pd
import pymongo

import os
os.chdir('C:\\Users\\mskara\\Desktop\\Hackhaton digital health')

#test_cash    = X_cash[1,]  # test_profits = X_profits[1,]
#names_cash                 # names_profits
'''
File for Data processing and testing
This section takes a new query, preprocess it and run
the two models to provides estimates for investment and profit
TO DO: To be implemented in a different file
! We are assuming only male customers, however a bank should know its customer informations
'''
forest_cash      = pickle.load(open("Forest_cash_doctors.pkl",      "rb"))
forest_profits   = pickle.load(open("Forest_profits_doctors.pkl",   "rb"))
newQuery_cash    = pickle.load(open("NewQuery_cash_doctors.pkl",    "rb"))
newQuery_profits = pickle.load(open("NewQuery_profits_doctors.pkl", "rb"))


new_query = {}

new_query['Sex']       = 'm'       #options: m, f  only set m for now
new_query['Age']       = '<40'     #options:<40, 40-50, >50
new_query['Region']    = 'Bayerns' #options: 17 regions (check the file.csv t Filter by name to know the exacts denominations)
new_query['City_type'] = 'big'     #options: big, small_medium, rural
new_query["Business_duration"] = 0 #set to 0 (the user wants to start a new business)
new_query["Insurance_allowed"] = 0 #options: 0 no  insurance, 1 yes insurance
new_query["Partnership"]       = 0 #options: 0 no partnership, 1 yes partnership


def process_NewQuery(NewQuery_cash, NewQuery_profits, model_cash, model_profits, new_query):
    '''
    1) take data from Mongo of a user of the Chatbot
    2) Rransform data in a format usable by the Random Forest
    3) Generate estimates and return them
    '''
    #for now is set as fixed (only male considered) due to the fact there is no question about sex
    NewQuery_cash['Sex_m']    = 1
    NewQuery_profits['Sex_m'] = 1
    
    NewQuery_cash   ['Age_%s' % str(new_query["Age"])] = 1
    NewQuery_profits['Age_%s' % str(new_query["Age"])] = 1

    if 'Region_%s' % str(new_query["Region"]) in NewQuery_cash:    
        NewQuery_cash['Region_%s' % str(new_query["Region"])]    = 1
    else: 
        NewQuery_cash['Region_OTHER']    = 1
    if 'Region_%s' % str(new_query["Region"]) in NewQuery_profits:    
        NewQuery_profits['Region_%s' % str(new_query["Region"])] = 1
    else:
        NewQuery_profits['Region_OTHER'] = 1
    
    NewQuery_cash    ['City_type_%s' % str(new_query["City_type"])] = 1 
    NewQuery_profits ['City_type_%s' % str(new_query["City_type"])] = 1

    NewQuery_cash['Business_duration'] = new_query["Business_duration"]
    NewQuery_cash['Insurance_allowed'] = new_query["Insurance_allowed"]
    NewQuery_cash['Partnership']       = new_query["Partnership"]
          
    ## now run model, find estimated investement, calculate business dimentsion, find profits
    #--> alternative approach  if str(NewQuery_cash[2]) in self.region:   arg['Region_%s' % str(NewQuery_cash[2])] = 1
    estimated_investment = model_cash.predict(NewQuery_cash)["estimate"]

    ##Data from chatbot --> Business_dimension estimated from the previous step
    if   estimated_investment > 200000: 
        NewQuery_profits['Business_dimension_big']      = 1
    elif estimated_investment < 120000: 
        NewQuery_profits['Business_dimension_small']    = 1
    else: NewQuery_profits['Business_dimension_medium'] = 1

    expected_profit = model_profits.predict(NewQuery_profits)["estimate"]
    net_salary       = 8000 * 12 * 0.75 #Net salary of a Doctor (monthly salary * months * 1 - tax rate)
    ### Payback time formula: invesment / (extra gain = profit - salary with no_investment)
    payback_time     = estimated_investment / (expected_profit * 0.75  - net_salary)
    # we could calcolate also easily Internal Rate Of Return  IRR
    
    #rounding profits and investment
    estimated_investment = round(estimated_investment / 100, 0) * 100
    expected_profit      = round(expected_profit      / 100, 0) * 100
    payback_time         = round(payback_time, 1)
    
    return estimated_investment, expected_profit, payback_time


estimated_investement, expected_profit, payback_time = process_NewQuery(
                                                                        NewQuery_cash    = newQuery_cash,
                                                                        NewQuery_profits = newQuery_profits,
                                                                        model_cash       = forest_cash,
                                                                        model_profits    = forest_profits,
                                                                        new_query        = new_query
                                                                       )
    
print(estimated_investement)
print(expected_profit)
print(payback_time)
