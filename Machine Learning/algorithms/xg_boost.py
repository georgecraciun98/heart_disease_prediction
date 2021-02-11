# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 12:32:19 2021

@author: George Craciun
"""
import pandas as pd

df1 = pd.read_csv("../heart.csv")
df1.shape
df1.columns
df1.head(10)


from sklearn import preprocessing
from sklearn.model_selection import train_test_split


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


from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import RandomFourierFeatures
import tensorflow as tf

#keras model 


import numpy as np
#scale all the data to 0, 1 values

for column in numeric_variables:
    new_df1[column]=new_df1[column]/np.max(new_df1[column])

x_train, x_test, y_train , y_test = train_test_split(new_df1,df1[target_variable],test_size=0.3)
from xgboost import XGBClassifier


model2=XGBClassifier(n_estimators=1000,learning_rate=0.01,max_depth=50,gamma=1)

model2.fit(x_train,y_train)

y_pred=model2.predict(x_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix,roc_curve, f1_score,roc_auc_score
accuracy_score(y_test,y_pred)
confusion_matrix(y_test,y_pred)


precision_score(y_test,y_pred)
recall_score(y_test,y_pred)
f1_score(y_test,y_pred)
fpr,tpr,thresholds =  roc_curve(y_test,y_pred)

import matplotlib.pyplot as plt
plt.plot(fpr,tpr,"b")
plt.plot([0,1],[0,1],"r-")
roc_auc_score(y_test,y_pred)



from sklearn.model_selection import RandomizedSearchCV

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

gamma = [int(x) for x in np.linspace(-5.0, 10.0,num=16)]

model2.get_params()
random_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'gamma':gamma
               }



cv_random=RandomizedSearchCV(estimator = model2, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

cv_random.fit(x_train,y_train)

cv_random.best_estimator_

cv_random.best_score_
