# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 11:12:59 2021

@author: George Craciun
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

def model_loading():
    
    #Bagging Techniques
    clf=RandomForestClassifier(bootstrap=False, max_depth=20, max_features='sqrt',
                           min_samples_leaf=4, min_samples_split=5,
                           n_estimators=800)
    
    
    return clf


def estimation(clf,x_train,y_train):
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    
    clf.get_params()
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    
    
    
    cv_random=RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    
    cv_random.fit(x_train,y_train)
    
    cv_random.best_estimator_
    
    cv_random.best_score_
