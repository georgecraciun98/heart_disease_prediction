# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 12:32:19 2021

@author: George Craciun
"""

import pandas as pd
from xgboost import XGBClassifier
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")
        
    elif train==False:
        pred = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Test Result:\n================================================")        
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")
def model_loading(input_shape):
    

    model=XGBClassifier(n_estimators=1600,learning_rate=0.01,max_depth=10,gamma=0,use_label_encoder=False)
    return model

def save_model(model):
    dump(model,"xg_boost_sklearn.joblib")

def load_from_file(path):
    return load(path)
if __name__ == "__main__":   
   # stuff only to run when not called via 'import' here


    input_shape=30
    model=load_from_file("xg_boost_sklearn.joblib")

    df = pd.read_csv("../heart.csv")
    X=df.drop('target',axis=1)
    y=df.target
    # input_shape=30
    # model=loading_binary(input_shape)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
   
    model.fit(X_train,y_train)
    #keras evalutation function
    score = print_score(model, X_train, y_train, X_test, y_test,train=False)

    predict=model.predict(X_test)
    save_model(model)
    
    



