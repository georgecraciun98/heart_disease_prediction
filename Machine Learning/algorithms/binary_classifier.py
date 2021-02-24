# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 11:57:07 2021

@author: George Craciun
"""

import pandas as pd


from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
import tensorflow as tf
import numpy as np
from algorithms.data_processing import load_encoder
from tensorflow.keras.models import Sequential

def model_loading(input_shape=30):
    
    
    model = Sequential()
    model.add(Dense(input_shape, input_shape=(input_shape,)))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam',
                           metrics=[
                               tf.keras.metrics.BinaryAccuracy(
                          name="accuracy", dtype=None, threshold=0.5),
                               tf.keras.metrics.Precision(
                          name='precision',thresholds=0.5),
                               tf.keras.metrics.Recall(
                          name='recall',thresholds=0.5),
                               
]
    )
    return model