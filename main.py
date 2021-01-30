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
    
    
    
# Let's make our correlation matrix a little prettier
def remove_cat_value():
    
    corr_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(15, 15))
    ax = sns.heatmap(corr_matrix,
                     annot=True,
                     linewidths=0.5,
                     fmt=".2f",
                     cmap="YlGnBu");
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    
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


def linear_regretion():
    
    model=keras.Sequential([
        Dense(32,activation=tf.nn.relu,input_shape=[30]),
        Dense(32,activation=tf.nn.relu),
        Dense(32,activation=tf.nn.relu),
        Dense(1),
        ])
    
    optimizer = tf.keras.optimizers.RMSprop(0.0099)
    model.compile(loss='mean_squared_error',optimizer=optimizer)
    model.fit(X_train,y_train,epochs=500)
    plt.scatter(X['chol'],y)  
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
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    model.fit(X_train, y_train, epochs=500, validation_data=(X_test, y_test))
    model.save("my_model02.h5")
    y_pred=model.predict(X_test)
    
    m1 = tf.keras.metrics.MeanSquaredError()
    m1.update_state(y_pred,y_test)
    m1.result().numpy()
    #0.27 MSE
    
from tensorflow.keras import layers
from tensorflow.python.keras.layers.kernelized import RandomFourierFeatures    

def support_vector_machine():
    

    
    model = keras.Sequential(
    [
        keras.Input(shape=(30,)),
        RandomFourierFeatures(
            output_dim=4096, scale=10.0, kernel_initializer="gaussian"
        ),
        layers.Dense(units=1),
    ]
    )
    model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.hinge,
    metrics=[keras.metrics.CategoricalAccuracy(name="acc")],
    )
    model.fit(X_train, y_train, epochs=20, batch_size=128, validation_split=0.2)
    model.save("my_model03.h5")
    y_pred=model.predict(X_test)
    
    m1 = tf.keras.metrics.MeanSquaredError()
    m1.update_state(y_pred,y_test)
    m1.result().numpy()
                                                                                                                                        
support_vector_machine()   
    



model=load_model("my_model03.h5",{"RandomFourierFeatures":RandomFourierFeatures},False)
y_pred=model.predict(X_test)
    
m1 = tf.keras.metrics.MeanSquaredError()
m1.update_state(y_pred,y_test)
m1.result().numpy()
    
    
    
    
    
    