# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 12:32:19 2021

@author: George Craciun
"""


from xgboost import XGBClassifier

def model_loading(input_shape):
    

    model=XGBClassifier(n_estimators=1600,learning_rate=0.01,max_depth=10,gamma=0)
    return model


