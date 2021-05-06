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
from algorithms.data_processing import accuracy_metrics,load_encoder,split_data,split_data_drop
from sklearn.linear_model import LogisticRegression
from tensorflow.keras import layers
from tensorflow.python.keras.layers.kernelized import RandomFourierFeatures    
from keras.regularizers import l2
from tensorflow.keras import activations


from algorithms.svm_keras import model_loading as svm_loading
from algorithms.binary_classifier import model_loading as loading_binary
from algorithms.xg_boost import model_loading as loading_xg,load_from_file
from algorithms.random_forest import model_loading as load_forest
import joblib
from sklearn.model_selection import StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
# Let's make our correlation matrix a little prettier


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
        

def decision_tree_classifier():
    pass    

def random_forest_classifier():
    pass
"""
sklearn model
"""
def xg_boost_classifier():
    input_shape=30
    model=load_from_file("./algorithms/xg_boost_sklearn.joblib")

    df = pd.read_csv("./heart.csv")
    X=df.drop('target',axis=1)
    y=df.target
    # input_shape=30
    # model=loading_binary(input_shape)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
   
    model.fit(X_train,y_train)
    #keras evalutation function
    score = print_score(model, X_train, y_train, X_test, y_test,train=False)

    y_pred=model.predict(X_test)
    accuracy_metrics(y_pred,y_test)
    save_sklearn_model(model,'xg_boost.sav')
"""
keras model
"""
def svm_prediction():
    df = pd.read_csv("./heart.csv")

    input_shape=30
    model=svm_loading(input_shape)
    x_train,y_train,x_test,y_test,x_val,y_val=split_data(df)
    
    model.fit(x_train,y_train,epochs=15,validation_data=(x_val, y_val))
    #keras evalutation function
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    predict=model.predict(x_test)
    model.save('svm_model.h5')
    y_pred=np.where(predict >= 0.5,1,0)
    
    encoder=load_encoder('./models/encoder_30.h5')
    # encode the train data
    X_train_encode = encoder.predict(x_train)
    # encode the test data
    X_test_encode = encoder.predict(x_test)
    
    model=svm_loading(15)
    
    # fit the model on the training set
    x_val = X_train_encode[-30:]
    model.fit(X_train_encode,y_train,epochs=15,validation_data=(x_val, y_val))
    
    # make predictions on the test set
    y_pred_1 = model.predict(X_test_encode)
"""
keras model
"""
def binary_classifier():
    df = pd.read_csv("./heart.csv")

    input_shape=30
    model=loading_binary(input_shape)
    x_train,y_train,x_test,y_test,x_val,y_val=split_data(df)
    
    model.fit(x_train,y_train,epochs=15,validation_data=(x_val, y_val))
    model.save('binary_classifier.sav')
    #keras evalutation function
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    predict=model.predict(x_test)
    
    y_pred=np.where(predict >= 0.5,1,0)
    accuracy_metrics(y_pred,y_test)
"""
sklearn model
"""
def random_forest():
    df = pd.read_csv("./heart.csv")
    input_shape=30
    model=load_forest()
    x_train,y_train,x_test,y_test,x_val,y_val=split_data(df)
    model.fit(x_train,y_train)
    predict=model.predict(x_test)
    y_pred=np.where(predict >= 0.5,1,0)
    accuracy_metrics(y_pred,y_test)
    save_sklearn_model(model,'random_forest.sav')
def save_sklearn_model(model,filename = 'finalized_model.sav'):
    
    joblib.dump(model, filename)

#model=load_model("my_model03.h5",{"RandomFourierFeatures":RandomFourierFeatures},False)
#y_pred=model.predict(X_test)

#print_score(model,X_train,y_train,X_test,y_test,train=False)

def load_model_example():
    model_1=load_model("models/my_model02.h5")
    return model_1

from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    
    #accuracy_metrics(y_pred,y_test)
    svm_prediction()
    #xg_boost_classifier()
    #binary_classifier()
    #random_forest()