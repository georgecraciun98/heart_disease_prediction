# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 11:57:01 2021

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
df = pd.read_csv("../heart.csv")
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

    

def pie_chart():
    df.columns
    size0 = df[df["target"] == 0].shape[0]
    size1 = df[df["target"] == 1].shape[0]
    sum=size0+ size1
    size_0_perc=(size0 * 100) / sum
    size_1_perc=(size1 * 100) / sum

    sizes = [size_0_perc, size_1_perc]
    explode = (0, 0.1)
    labels=["Have heart disease","Don't have heart disease"]
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')

#append continous variables 
pie_chart()
for column in df.columns:
    print('==============================')
    print(f"{column} : {df[column].unique()}")
    if len(df[column].unique()) <= 10:
        categorical_val.append(column)
    else:
        continous_val.append(column)
        

plt.figure(figsize=(15, 15))


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

correlation_matrix()


plt.figure(figsize=(15, 15))

for i, column in enumerate(categorical_val, 1):
    plt.subplot(3, 3, i)
    df[df["target"] == 0][column].hist(bins=35, color='blue', label='Have Heart Disease = NO', alpha=0.6)
    df[df["target"] == 1][column].hist(bins=35, color='red', label='Have Heart Disease = YES', alpha=0.6)
    plt.legend()
    plt.xlabel(column)