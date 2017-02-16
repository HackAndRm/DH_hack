# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 16:22:27 2017

@author: mskara
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble         import RandomForestRegressor
from sklearn.cross_validation import train_test_split ###on some python requires sklearn.model_selection
from sklearn.grid_search      import GridSearchCV

from datetime import datetime
import warnings
warnings.simplefilter(action = "ignore", category = DeprecationWarning)

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
        max_depth_range    = [5, 7, 10, 12, 15, 20]
        min_samples_leaf   = [max(5, int(0.002 * len(X)))]
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
                                     n_jobs  = 4)
        best_forest = None
        best_score  = -1000000000000000
        for i in range(1):
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

        importances = new_best_forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in new_best_forest.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]
        # Print the feature ranking
        print("Feature ranking:")
        for f in range(int(X.shape[1]/2)):
            print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances of the model")
        plt.bar(range(X.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
        plt.xticks(range(X.shape[1]), indices)
        plt.xlim([-1, X.shape[1]])
        plt.show()


        print ("Validation MAPE: " + str(mape))
        print ("Validation MAPE in bounds: " + str(mape_in_bounds))
        print ("\nTime used to train the model:")
        print (datetime.now()- now)

    def predict ( self , X ,output=False):
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
        
        return return_dict