# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 15:22:20 2021

@author: George Craciun
"""

#path = "G:/Freelancing/Machine Learning Videos/Module 4/Data/"
import pandas as pd

df1 = pd.read_csv("heart.csv")
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

x_val = x_train[-30:]
y_val = y_train[-30:]
x_train = x_train[:-30]
y_train = y_train[:-30]
model = keras.Sequential(
    [
        layers.Dense(20, input_shape=(30,)),
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

model.fit(x_train,y_train,epochs=15,validation_data=(x_val, y_val))
model.save("my_model03.h5")
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
predict=model.predict(x_test)
m1 = tf.keras.metrics.MeanSquaredError()
m1.update_state(predict,y_test)
m1.result().numpy()







