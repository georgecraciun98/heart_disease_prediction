# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 18:33:28 2021

@author: George Craciun
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


from tensorflow.keras.models import load_model

from keras import Sequential
from tensorflow.keras.layers import Dense
df = pd.read_csv("./heart.csv")
df.head()



df.isna().sum()

categorical_val = []
continous_val = []

def exploratory_analysis():
    pass
    df.columns
    corr = df.corr()
    df['thalach'].plot(kind='hist')
    plt.hist(df['thalach'],bins=10)
    df['thalach'].mean()
    df['thalach'].var()
    coef=df['thalach'].std()/df['thalach'].mean() *100
    #stacked bar char
    
    #describe the data
    describe=df.describe()
    df.target.value_counts().plot(kind="bar", color=["salmon", "lightblue"])

    
#append continous variables 

for column in df.columns:
    print('==============================')
    print(f"{column} : {df[column].unique()}")
    if len(df[column].unique()) <= 10:
        categorical_val.append(column)
    else:
        continous_val.append(column)
        

plt.figure(figsize=(15, 15))

# 9 plots for statistics

for i, column in enumerate(categorical_val, 1):
    plt.subplot(3, 3, i)
    df[df["target"] == 0][column].hist(bins=35, color='blue', label='Have Heart Disease = NO', alpha=0.6)
    df[df["target"] == 1][column].hist(bins=35, color='red', label='Have Heart Disease = YES', alpha=0.6)
    plt.legend()
    plt.xlabel(column)

def plots():
    # Create another figure
    plt.figure(figsize=(10, 8))
    
    # Scatter with postivie examples
    plt.scatter(df.age[df.target==1],
                df.thalach[df.target==1],
                c="salmon")
    
    # Scatter with negative examples
    plt.scatter(df.age[df.target==0],
                df.thalach[df.target==0],
                c="lightblue")
    
    # Add some helpful info
    plt.title("Heart Disease in function of Age and Max Heart Rate")
    plt.xlabel("Age")
    plt.ylabel("Max Heart Rate")
    plt.legend(["Disease", "No Disease"]);
    
def correlation_matrix():
    corr_matrix = df.corr()
    fix,ax=plt.subplots(figsize=(15,15))
    ax = sns.heatmap(corr_matrix,
                 annot=True,
                 linewidths=0.5,
                 fmt=".2f",
                 cmap="YlGnBu");
    bottom,top=ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)

