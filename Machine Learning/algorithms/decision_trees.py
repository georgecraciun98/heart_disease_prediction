# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 19:16:54 2021

@author: George Craciun
"""



import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import RandomFourierFeatures
import numpy as np

df1 = pd.read_csv("../heart.csv")

df1.shape
df1.columns
df1.head(10)



def label_encoder(df1,column):
    le = preprocessing.LabelEncoder()
    df1[column]=le.fit_transform(df1[column])
    ohe = preprocessing.OneHotEncoder()
    temp_array = ohe.fit_transform(df1[[column]]).toarray()
    column_names = [column+"_"+str(m) for m in le.classes_]
    return(pd.DataFrame(data=temp_array,columns = column_names))


categorical_variables = ["sex","cp",'fbs','restecg','exang','slope','ca','thal']
target_variable = ["target"]
numeric_variables = list(set(df1.columns.values) - set(categorical_variables) -set(target_variable))
new_df1 = df1[numeric_variables]
for column in categorical_variables:
    new_df1= pd.concat([new_df1,label_encoder(df1,column)],axis=1)
    
new_df1=pd.concat([new_df1,df1['target']],axis=1)


#scale all the data to 0, 1 values
#Normalized data
for column in numeric_variables:
    new_df1[column]=new_df1[column]/np.max(new_df1[column])

n_folds = 5
max_depth = 5
min_size = 10

