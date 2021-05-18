import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from joblib import dump, load

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.layers.kernelized import RandomFourierFeatures    
from tensorflow.keras import activations
from tensorflow.keras import layers

from keras import Sequential
from keras.regularizers import l2
from keras.wrappers.scikit_learn import KerasClassifier
#model loading libraries
from algorithms.data_processing import accuracy_metrics,load_encoder,split_data,split_data_drop
from algorithms.svm_keras import model_loading as svm_loading
from algorithms.binary_classifier import model_loading as loading_binary
from algorithms.xg_boost import model_loading as loading_xg,load_from_file
from algorithms.random_forest import model_loading as load_forest
import joblib

import sys
from datetime import datetime

date = datetime.now().strftime("%Y_%m_%d")
# Let's make our correlation matrix a little prettier
# def print_score(clf, X_train, y_train, X_test, y_test, train=True):

#     if train:
#         pred = clf.predict(X_train)
#         clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
#         print("Train Result:\n================================================")
#         print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
#         print("_______________________________________________")
#         print(f"CLASSIFICATION REPORT:\n{clf_report}")
#         print("_______________________________________________")
#         print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")
        
#     elif train==False:
#         pred = clf.predict(X_test)
#         clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
#         print("Test Result:\n================================================")        
#         print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
#         print("_______________________________________________")
#         print(f"CLASSIFICATION REPORT:\n{clf_report}")
#         print("_______________________________________________")
#         print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")
        

"""
sklearn model
"""
def load_from_file1(path):
    return load(path)
def xg_boost_classifier():
    #sys.stdout=open(f"xg_boost_classifier_{date}.txt","w")

    input_shape=30
    #model=load_from_file1("./algorithms/xg_boost_sklearn.joblib")
    model=loading_xg(30)
    print("Xg Boost Classifier")

    df = pd.read_csv("./heart.csv")
    X=df.drop('target',axis=1)
    y=df.target
    # input_shape=30
    # model=loading_binary(input_shape)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
   
    model.fit(X_train,y_train)
    #keras evalutation function
    # score = print_score(model, X_train, y_train, X_test, y_test,train=False)

    # y_pred=model.predict(X_test)
    estimator =model
    x_total=pd.concat([X_train,X_test],axis=0,ignore_index=True,verify_integrity=True)
    y_total=pd.concat([y_train,y_test],axis=0,ignore_index=True,verify_integrity=True)
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    results = cross_val_score(estimator, x_total, y_total, cv=kfold)
 
    # accuracy_metrics(y_pred,y_test)
    # save_sklearn_model(model,'xg_boost.sav')
"""
keras model
"""
def svm_prediction():
    df = pd.read_csv("./heart.csv")
    #sys.stdout=open(f"svm_prediction_{date}.txt","w")

    input_shape=30
    model=svm_loading(input_shape)
    """
    we are splitting featured data , input shape-ul va deveni 30 
    """
    X_train,y_train,X_test,y_test,x_val,y_val=split_data(df)
    print("SVM Prediction")

    model.fit(X_train,y_train,epochs=15,validation_data=(x_val, y_val))
    #keras evalutation function
    # score = model.evaluate(X_test, y_test, verbose=0)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])
    # predict=model.predict(X_test)
    # model.save('svm_model.h5')
    # y_pred=np.where(predict >= 0.5,1,0)
    # encoder=load_encoder('./models/encoder_30.h5')
    # # encode the train data
    # X_train_encode = encoder.predict(X_train)
    # # encode the test data
    # X_test_encode = encoder.predict(X_test)
    
    # model=svm_loading(15)
    
    # # fit the model on the training set
    # x_val = X_train_encode[-30:]
    # model.fit(X_train_encode,y_train,epochs=15,validation_data=(x_val, y_val))
    
    # make predictions on the test set
    x_total=pd.concat([X_train,X_test],axis=0,ignore_index=True,verify_integrity=True)
    y_total=pd.concat([y_train,y_test],axis=0,ignore_index=True,verify_integrity=True)
    estimator = KerasClassifier(build_fn=model, epochs=100, batch_size=5, verbose=0)   
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    results = cross_val_score(estimator, x_total, y_total, cv=kfold)
    print("we are done")
"""
keras model
"""
"""

"""
def binary_classifier():
    df = pd.read_csv("./heart.csv")
    print("Binary Classifier")
    #sys.stdout=open(f"binary_classifier_{date}.txt","w")
    input_shape=30
    model=loading_binary(input_shape)
    model.save("binary_classifier")
    """
    we are splitting featured data , input shape-ul va deveni 30 
    """
    X_train,y_train,X_test,y_test,x_val,y_val=split_data(df)
    
    model.fit(X_train,y_train,epochs=15,validation_data=(x_val, y_val))
    #model.save(f"binary_classifier_{date}.sav")
    #keras evalutation function
    #score = model.evaluate(X_test, y_test, verbose=0)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])
    # predict=model.predict(X_test)
    estimator = KerasClassifier(build_fn=model, epochs=100, batch_size=5, verbose=0)
    x_total=pd.concat([X_train,X_test],axis=0,ignore_index=True,verify_integrity=True)
    y_total=pd.concat([y_train,y_test],axis=0,ignore_index=True,verify_integrity=True)
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    results = cross_val_score(estimator, x_total, y_total, cv=kfold)
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
    
from sklearn.preprocessing import LabelEncoder

def test_classifier():
    dataframe = pd.read_csv("sonar.csv", header=None)
    dataset = dataframe.values
    # split into input (X) and output (Y) variables
    X = dataset[:,0:60].astype(float)
    Y = dataset[:,60]
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    # baseline model
    def create_baseline():
    	# create model
    	model = Sequential()
    	model.add(Dense(60, input_dim=60, activation='relu'))
    	model.add(Dense(1, activation='sigmoid'))
    	# Compile model
    	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    	return model
    # evaluate model with standardized dataset
    estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
   
# """
# sklearn model
# """
def random_forest():
    df = pd.read_csv("./heart.csv")
    
    #sys.stdout=open(f"random_forest_{date}.txt","w")
    print("Random Forest")
    input_shape=30
    model=load_forest()
    X_train,y_train,X_test,y_test,x_val,y_val=split_data(df)
    model.fit(X_train,y_train)
    predict=model.predict(X_test)
    y_pred=np.where(predict >= 0.5,1,0)
    accuracy_metrics(y_pred,y_test)
    save_sklearn_model(model,'random_forest.sav')
    
    estimator = model
    x_total=pd.concat([X_train,X_test],axis=0,ignore_index=True,verify_integrity=True)
    y_total=pd.concat([y_train,y_test],axis=0,ignore_index=True,verify_integrity=True)
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    results = cross_val_score(estimator, x_total, y_total, cv=kfold)
def save_sklearn_model(model,filename = 'finalized_model.sav'):
    
    joblib.dump(model, filename)


#model=load_model("my_model03.h5",{"RandomFourierFeatures":RandomFourierFeatures},False)
#y_pred=model.predict(X_test)
#print_score(model,X_train,y_train,X_test,y_test,train=False)

def load_model_example():
    model_1=load_model("models/my_model02.h5")
    return model_1
def model_loading_example():
    
    pass
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    svm_prediction()
    #xg_boost_classifier()
    #binary_classifier()
    #random_forest()