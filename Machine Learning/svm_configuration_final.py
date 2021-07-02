# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 17:57:46 2021

@author: George Craciun
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 17:21:38 2021

@author: George Craciun
"""

from sklearn.model_selection import GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# %matplotlib inline
# sns.set_style("whitegrid")
# plt.style.use("fivethirtyeight")

# params = {"C": np.logspace(-4, 4, 20),
#           "solver": ["liblinear"]}

# lr_clf = LogisticRegression()

# lr_cv = GridSearchCV(lr_clf, params, scoring="accuracy", n_jobs=-1, verbose=1, cv=5, iid=True)
# lr_cv.fit(X_train, y_train)
# best_params = lr_cv.best_params_
# print(f"Best parameters: {best_params}")
# lr_clf = LogisticRegression(**best_params)

# lr_clf.fit(X_train, y_train)

# print_score(lr_clf, X_train, y_train, X_test, y_test, train=True)
# print_score(lr_clf, X_train, y_train, X_test, y_test, train=False)



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

df = pd.read_csv("./heart.csv")
df.head()

df.info()

categorical_val = []
continous_val = []
for column in df.columns:
    print('==============================')
    print(f"{column} : {df[column].unique()}")
    if len(df[column].unique()) <= 10:
        categorical_val.append(column)
    else:
        continous_val.append(column)
categorical_val.remove('target')
dataset = pd.get_dummies(df, columns = categorical_val)   
X = dataset.drop('target', axis=1)
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)        


# lr_clf = LogisticRegression(solver='liblinear')
# lr_clf.fit(X_train, y_train)

# print_score(lr_clf, X_train, y_train, X_test, y_test, train=True)
# print_score(lr_clf, X_train, y_train, X_test, y_test, train=False)



# test_score = accuracy_score(y_test, lr_clf.predict(X_test)) * 100
# train_score = accuracy_score(y_train, lr_clf.predict(X_train)) * 100

# results_df = pd.DataFrame(data=[["Logistic Regression", train_score, test_score]], 
#                           columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])
# results_df




from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

svm_clf = SVC(kernel='rbf', gamma=0.1, C=1.0)

params = {"C":(0.1, 0.5, 1, 2, 5, 10, 20), 
          "gamma":(0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1), 
          "kernel":('linear', 'poly', 'rbf')}

svm_cv = GridSearchCV(svm_clf, params, n_jobs=-1, cv=5, verbose=1, scoring="accuracy")
svm_cv.fit(X_train, y_train)
best_params = svm_cv.best_params_
print(f"Best params: {best_params}")

svm_clf = SVC(**best_params)
svm_clf.fit(X_train, y_train)

print_score(svm_clf, X_train, y_train, X_test, y_test, train=True)
print_score(svm_clf, X_train, y_train, X_test, y_test, train=False)
