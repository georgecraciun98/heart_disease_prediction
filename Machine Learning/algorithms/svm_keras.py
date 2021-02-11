# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 15:22:20 2021

@author: George Craciun
"""

#path = "G:/Freelancing/Machine Learning Videos/Module 4/Data/"
import pandas as pd


from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import RandomFourierFeatures
import tensorflow as tf
import numpy as np
from algorithms.data_processing import load_encoder

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


def model_loading(input_shape):
    

    model = keras.Sequential(
        [
            layers.Dense(20, input_shape=(input_shape,)),
            RandomFourierFeatures(
                output_dim=4096, kernel_initializer="gaussian"
            ),
            layers.Dense(units=1,activation='sigmoid'),
        ]
        )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.hinge,
        metrics=[tf.keras.metrics.BinaryAccuracy(
        name="binary_accuracy", dtype=None, threshold=0.5
    )],
    )
    return model






def save_model(model,name='my_model03.h'):
    model.save(name)


if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
    df = pd.read_csv("./heart.csv")

    input_shape=30
    model=model_loading(input_shape)
    x_train,y_train,x_test,y_test,x_val,y_val=split_data(df)
    
    model.fit(x_train,y_train,epochs=15,validation_data=(x_val, y_val))
    #keras evalutation function
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    predict=model.predict(x_test)
    
    
    y_pred=np.where(predict >= 0.5,1,0)
    
    
    encoder=load_encoder('../models/encoder_30.h5')
    # encode the train data
    X_train_encode = encoder.predict(x_train)
    # encode the test data
    X_test_encode = encoder.predict(x_test)
    
    model=model_loading(15)
    
    # fit the model on the training set
    x_val = X_train_encode[-30:]
    model.fit(X_train_encode,y_train,epochs=15,validation_data=(x_val, y_val))
    
    
    # make predictions on the test set
    y_pred_1 = model.predict(X_test_encode)
    
