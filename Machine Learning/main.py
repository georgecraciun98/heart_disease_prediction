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
from algorithms.xg_boost import model_loading as loading_xg
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
    


def decision_tree_classifier():
    pass    

def random_forest_classifier():
    pass
def xg_boost_classifier():
    pass
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
        
    model=load_model_example()
    
    
    df = pd.read_csv("./heart.csv")
    X=df.drop('target',axis=1)
    y=df.target
    # input_shape=30
    # model=loading_binary(input_shape)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
    # corr=df.corr()    
    # #encoder=load_encoder('models/encoder_30.h5')   

    # #X_train_encode = encoder.predict(X_train)
    # # encode the test data
    # #X_test_encode = encoder.predict(X_test)                                                                                             
    # #support_vector_machine()   
    
    # # fit model on training set
    # model.fit(X_train, y_train)
    # #save_sklearn_model(model,'./models/sklearn.sav')
    # # make prediction on test set
    # binary_class = lambda x : 1 if (x>=0.5) else 0 
    lr_clf = LogisticRegression(solver='liblinear')
    lr_clf.fit(X_train, y_train)
    
    y_pred = lr_clf.predict(X_test)
    # y_pred=np.array([binary_class(i) for i in y_pred])
    
    # x_total=pd.concat([X_train,X_test],axis=0,ignore_index=True,verify_integrity=True)
    # y_total=pd.concat([y_train,y_test],axis=0,ignore_index=True,verify_integrity=True)
    # estimator = KerasClassifier(build_fn=loading_binary,input_shape=input_shape, epochs=100, batch_size=5, verbose=0)
    # kfold = StratifiedKFold(n_splits=10, shuffle=True)
    # results = cross_val_score(estimator, x_total, y_total, cv=kfold)
    # print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
    
    
    #accuracy_metrics(y_pred,y_test)
    
    