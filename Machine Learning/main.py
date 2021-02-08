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
    
# Let's make our correlation matrix a little prettier
def remove_cat_value():
    
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
        
dataset = remove_cat_value()
        
X = dataset.drop('target', axis=1)
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

x_val = X_train[-30:]
y_val = y_train[-30:]
X_train = X_train[:-30]
y_train = y_train[:-30]
def linear_regresion():
    
    model=keras.Sequential([
        Dense(32,activation=tf.nn.relu,input_shape=[30]),
        Dense(32,activation=tf.nn.relu),
        Dense(32,activation=tf.nn.relu),
        Dense(1),
        ])
    
    optimizer = tf.keras.optimizers.RMSprop(0.0099)
    model.compile(loss='mean_squared_error',optimizer=optimizer)
    model.fit(X_train,y_train,epochs=500)
 
    model.save("my_model01.h5")
    
    model1=keras.models.load_model("my_model01.h5")
    y_pred=model1.predict(X_test)
          
    m1 = tf.keras.metrics.MeanSquaredError()
    m1.update_state(y_pred,y_test)
    m1.result().numpy()
    #0.13 MSE


#Logistic Regression
def logistic_regretion():
    
    from keras.regularizers import L1L2
    
    reg = L1L2(l1=0.01, l2=0.01)
    
    model = keras.Sequential()
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=reg,input_shape=[30]))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy',metrics=[keras.metrics.CategoricalAccuracy(name="acc")])
    model.fit(X_train, y_train, epochs=500, validation_data=(X_test, y_test))
    model.save("my_model02.h5")
    y_pred=model.predict(X_test)
    
    m1 = tf.keras.metrics.MeanSquaredError()
    m1.update_state(y_pred,y_test)
    m1.result().numpy()
    
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    predict=model.predict(X_test)
    #0.27 MSE
    
from tensorflow.keras import layers
from tensorflow.python.keras.layers.kernelized import RandomFourierFeatures    
from keras.regularizers import l2
from tensorflow.keras import activations
def support_vector_machine():
   
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
        metrics=['binary_accuracy'],
    )
    
    model.fit(X_train,y_train,epochs=15,validation_data=(x_val, y_val))
    model.save("my_model03.h5")
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    predict=model.predict(X_test)
    m1 = tf.keras.metrics.MeanSquaredError()
    m1.update_state(predict,y_test)
    m1.result().numpy()
    
    
    
                                                                                                                                        
#support_vector_machine()   

def decision_tree_classifier():
    pass    

def random_forest_classifier():
    pass
def xg_boost_classifier():
    pass

#model=load_model("my_model03.h5",{"RandomFourierFeatures":RandomFourierFeatures},False)
#y_pred=model.predict(X_test)

#print_score(model,X_train,y_train,X_test,y_test,train=False)


    
    
    
    
    
    