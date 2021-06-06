# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 11:12:59 2021

@author: George Craciun
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import sys
date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# sys.stdout=open(f"log_random_forest_{date}.txt","w")
def model_loading():
    
    #Bagging Techniques
    clf=RandomForestClassifier(bootstrap=False, max_depth=20, max_features='sqrt',
                           min_samples_leaf=4, min_samples_split=5,
                           n_estimators=800)
                 
    
    return clf

from sklearn.model_selection import RandomizedSearchCV
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

import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate

if __name__ == "__main__":
    input_shape=30
    #model=load_from_file1("./algorithms/xg_boost_sklearn.joblib")
    model = model_loading()
    print("Random forest")

    df = pd.read_csv("../heart.csv")
    X=df.drop('target',axis=1)
    y=df.target
    # input_shape=30
    # model=loading_binary(input_shape)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
   

    #keras evalutation function
    # score = print_score(model, X_train, y_train, X_test, y_test,train=False)
    # y_pred=model.predict(X_test)
    estimator =model
    x_total=pd.concat([X_train,X_test],axis=0,ignore_index=True,verify_integrity=True)
    y_total=pd.concat([y_train,y_test],axis=0,ignore_index=True,verify_integrity=True)
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_micro': 'recall_macro',
           'prec':'precision',
           'f1':'f1'}
    results = cross_validate(estimator, x_total, y_total, cv=kfold,scoring=scoring)
    print('results are',results)
    