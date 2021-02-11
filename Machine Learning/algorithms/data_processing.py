# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 17:39:21 2021

@author: George Craciun
"""
import pandas as pd


from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

import tensorflow as tf
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix,roc_curve, f1_score,roc_auc_score

def label_encoder(df1,column):
    le = preprocessing.LabelEncoder()
    df1[column]=le.fit_transform(df1[column])
    ohe = preprocessing.OneHotEncoder()
    temp_array = ohe.fit_transform(df1[[column]]).toarray()
    column_names = [column+"_"+str(m) for m in le.classes_]
    return(pd.DataFrame(data=temp_array,columns = column_names))

def min_max_scaler(new_df1):
    scaler = MinMaxScaler()
    scaler.fit(new_df1.to_numpy())
    new_df1_np = scaler.transform(new_df1.to_numpy())
    new_df1=pd.DataFrame(new_df1_np)
    return new_df1

def normalize_data(df1):

    categorical_variables = ["sex","cp",'fbs','restecg','exang','slope','ca','thal']
    target_variable = ["target"]
    numeric_variables = list(set(df1.columns.values) - set(categorical_variables) -set(target_variable))
    
    new_df1 = df1[numeric_variables]

    for column in categorical_variables:
        new_df1= pd.concat([new_df1,label_encoder(df1,column)],axis=1)
    #data normalization by std deviation
    for column in numeric_variables:
        new_df1[column]=new_df1[column]/np.std(new_df1[column], axis = 0)
        
    return new_df1,target_variable
        

def split_data(df1):
    new_df1,target_variable=normalize_data(df1)
    
    x_train, x_test, y_train , y_test = train_test_split(new_df1,df1[target_variable],test_size=0.3)
    
    x_val = x_train[-30:]
    y_val = y_train[-30:]
    x_train = x_train[:-30]
    y_train = y_train[:-30]
    return x_train,y_train,x_test,y_test,x_val,y_val
def remove_cat_value(df,categorical_val,continous_val):
    
    for column in df.columns:
        print('==============================')
        print(f"{column} : {df[column].unique()}")
        if len(df[column].unique()) <= 10:
            categorical_val.append(column)
        else:
            continous_val.append(column)
        
    categorical_val.remove('target')
    dataset = pd.get_dummies(df, columns = categorical_val)
    
    s_sc=StandardScaler()
    col_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    dataset[col_to_scale] = s_sc.fit_transform(dataset[col_to_scale])
    return dataset

def accuracy_metrics(y_test,y_pred):

    
    print('accuracy score',accuracy_score(y_test,y_pred))
    print('confusion matrix ',confusion_matrix(y_test,y_pred))
    print('Precision matrix ',precision_score(y_test,y_pred))
    print('recall score ',recall_score(y_test,y_pred))
    print('F1 score ',f1_score(y_test,y_pred))
    
    
    fpr,tpr,thresholds =  roc_curve(y_test,y_pred)
    
    plt.plot(fpr,tpr,"b")
    plt.plot([0,1],[0,1],"r-")
    plt.show()
    print('roc auc score',roc_auc_score(y_test,y_pred))
    
def load_encoder(name='encoder_30.h5'):

    #Using feature extraction
    from tensorflow.keras.models import load_model
    encoder = load_model(name)
    tf.keras.utils.plot_model(encoder, 'autoencoder_no_compress.png',\
                              show_shapes=True)
    return encoder

 