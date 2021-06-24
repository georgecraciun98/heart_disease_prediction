from sklearn.model_selection import train_test_split
from sklearn.metrics import  classification_report
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score,\
    confusion_matrix,roc_curve, f1_score,roc_auc_score,recall_score
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

from ml_app.submodels.model_configuration import ModelConfiguration


class ModelSaving:
    def __init__(self):
        self.categorical_val=['sex','cp','fbs','restecg','exang','slope','ca','thal']


    def save_model(self,model,name):
        full_path = "./models/"
        dump(model,full_path+ name)

    def print_score(self, clf, X_train, y_train, X_test, y_test, train=True):
        if train:
            pred = clf.predict(X_train)
            clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
            print("Train Result:\n================================================")
            print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
            print("_______________________________________________")
            print(f"CLASSIFICATION REPORT:\n{clf_report}")
            print("_______________________________________________")
            print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")

        elif train == False:
            pred = clf.predict(X_test)
            clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
            print("Test Result:\n================================================")
            print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
            print("_______________________________________________")
            print(f"CLASSIFICATION REPORT:\n{clf_report}")
            print("_______________________________________________")
            print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")

    def remove_cat_value(self,df):
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
        dataset = pd.get_dummies(df, columns=categorical_val)

        s_sc = StandardScaler()
        col_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']


        dataset[col_to_scale]= s_sc.fit_transform(dataset[col_to_scale])
        dump(s_sc, 'std_scaler.bin',compress=True)

        return dataset
    def activate_model(self,name,data,researcher_id):
        pass
        if name=='Random Forest Classifier':
            df = pd.read_csv("./ml_app/services/heart.csv")
            df = self.remove_cat_value(df)

            X = df.drop('target', axis=1)
            y = df.target

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
            # filename = f"../saved_models/xg_boost_2021_05_12_14_32_20.sav"
            # model=joblib.load(filename)
            model_conf=ModelConfiguration.objects.filter(alg_name=name).order_by("-created_date").first()
            rf_clf = RandomForestClassifier(n_estimators=model_conf.n_estimators,
                                            max_features=model_conf.max_features,max_depth=model_conf.max_depth,
                                            min_samples_split=model_conf.min_samples_split,
                                            min_samples_leaf=model_conf.min_samples_leaf,bootstrap=model_conf.bootstrap
                                            )
            rf_clf.fit(X_train, y_train)

            pred = rf_clf.predict(X_test)
            accuracy_score1=accuracy_score(y_test, pred) * 100
            precision_score1=precision_score(y_test,pred)*100
            f1_score1=f1_score(y_test,pred)*100
            recall_score1 = recall_score(y_test, pred) * 100
            roc_auc_score1=roc_auc_score(y_test,pred)*100

            model_conf.accuracy=accuracy_score1
            model_conf.precision=precision_score1
            model_conf.f1_score=f1_score1
            model_conf.recall_score=recall_score1
            model_conf.roc_auc_score=roc_auc_score1
            model_conf.alg_description = name + " Precision " + str(precision_score1) + " Accuracy " + str(
                accuracy_score1)
            model_conf.save()

            print(X.loc[:1])
            res = rf_clf.predict(X.loc[:1])

            create_date=model_conf.created_date
            date=create_date.strftime("%m%d%Y_%H_%M_%S")

            self.save_model(rf_clf,name=f"random_forest_{date}.joblib")
            self.print_score(rf_clf, X_train, y_train, X_test, y_test, train=True)
            self.print_score(rf_clf, X_train, y_train, X_test, y_test, train=False)
        if name=='Binary Classifier':
            df = pd.read_csv("./ml_app/services/heart.csv")
            df = self.remove_cat_value(df)

            X = df.drop('target', axis=1)
            y = df.target

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
            # filename = f"../saved_models/xg_boost_2021_05_12_14_32_20.sav"
            # model=joblib.load(filename)
            model_conf=ModelConfiguration.objects.filter(alg_name=name).order_by("-created_date").first()
            lr_clf = LogisticRegression(solver=model_conf.solver,C=model_conf.c,penalty=model_conf.penalty)

            lr_clf.fit(X_train, y_train)

            pred = lr_clf.predict(X_test)
            accuracy_score1=accuracy_score(y_test, pred) * 100
            precision_score1=precision_score(y_test,pred)*100
            f1_score1 = f1_score(y_test, pred) * 100
            recall_score1 = recall_score(y_test, pred) * 100
            roc_auc_score1 = roc_auc_score(y_test, pred) * 100

            model_conf.accuracy = accuracy_score1
            model_conf.precision = precision_score1
            model_conf.f1_score = f1_score1
            model_conf.recall_score = recall_score1
            model_conf.roc_auc_score = roc_auc_score1
            model_conf.save()


            print(X.loc[:1])
            res = lr_clf.predict(X.loc[:1])
            print('res is',res)

            create_date = model_conf.created_date
            date=create_date.strftime("%m%d%Y_%H_%M_%S")

            self.save_model(lr_clf,name=f"logistic_regression_{date}.joblib")
            self.print_score(lr_clf, X_train, y_train, X_test, y_test, train=True)
            self.print_score(lr_clf, X_train, y_train, X_test, y_test, train=False)
        if name=='Support Vector Machine':
            df = pd.read_csv("./ml_app/services/heart.csv")
            df = self.remove_cat_value(df)

            X = df.drop('target', axis=1)
            y = df.target

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
            # filename = f"../saved_models/xg_boost_2021_05_12_14_32_20.sav"
            # model=joblib.load(filename)
            model_conf=ModelConfiguration.objects.filter(alg_name=name).order_by("-created_date").first()
            svm_clf = SVC(C=model_conf.c,gamma=model_conf.gamma,kernel=model_conf.kernel)

            svm_clf.fit(X_train, y_train)


            pred = svm_clf.predict(X_test)
            accuracy_score1=accuracy_score(y_test, pred) * 100
            precision_score1=precision_score(y_test,pred)*100
            f1_score1 = f1_score(y_test, pred) * 100
            recall_score1 = recall_score(y_test, pred) * 100
            roc_auc_score1 = roc_auc_score(y_test, pred) * 100

            model_conf.accuracy = accuracy_score1
            model_conf.precision = precision_score1
            model_conf.f1_score = f1_score1
            model_conf.recall_score = recall_score1
            model_conf.roc_auc_score = roc_auc_score1
            print('roc score is',roc_auc_score1)
            model_conf.save()

            print(X.loc[:1])
            res = svm_clf.predict(X.loc[:1])
            create_date = model_conf.created_date
            date=create_date.strftime("%m%d%Y_%H_%M_%S")

            self.save_model(svm_clf,name=f"support_vector_machine_{date}.joblib")
            self.print_score(svm_clf, X_train, y_train, X_test, y_test, train=True)
            self.print_score(svm_clf, X_train, y_train, X_test, y_test, train=False)

        if name=='XGB Classifier':
            df = pd.read_csv("./ml_app/services/heart.csv")
            df = self.remove_cat_value(df)

            X = df.drop('target', axis=1)
            y = df.target

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
            # filename = f"../saved_models/xg_boost_2021_05_12_14_32_20.sav"
            # model=joblib.load(filename)
            model_conf=ModelConfiguration.objects.filter(alg_name=name).order_by("-created_date").first()
            print('leafs are',model_conf.min_samples_leaf)
            rf_clf = XGBClassifier(n_estimators=model_conf.n_estimators,
                                            max_depth=model_conf.max_depth,booster=model_conf.booster,
                                            base_score=model_conf.base_score,
                                            learning_rate=model_conf.learning_rate,bootstrap=model_conf.min_child_weight
                                            )
            rf_clf.fit(X_train, y_train)
            print(X.loc[:1])

            pred = rf_clf.predict(X_test)
            accuracy_score1 = accuracy_score(y_test, pred) * 100
            precision_score1 = precision_score(y_test, pred) * 100
            f1_score1 = f1_score(y_test, pred) * 100
            recall_score1 = recall_score(y_test, pred) * 100
            roc_auc_score1 = roc_auc_score(y_test, pred) * 100
            print('roc score and auc score is', roc_auc_score1)

            model_conf.accuracy = accuracy_score1
            model_conf.precision = precision_score1
            model_conf.f1_score = f1_score1
            model_conf.recall_score = recall_score1
            model_conf.roc_auc_score = roc_auc_score1
            model_conf.save()

            res = rf_clf.predict(X.loc[:1])
            create_date = model_conf.created_date
            date=create_date.strftime("%m%d%Y_%H_%M_%S")

            self.save_model(rf_clf,name=f"xg_boost_{date}.joblib")
            self.print_score(rf_clf, X_train, y_train, X_test, y_test, train=True)
            self.print_score(rf_clf, X_train, y_train, X_test, y_test, train=False)

    def train_model(self,name,data,researcher_id):
        pass
        if name=='Random Forest Classifier':
            df = pd.read_csv("./ml_app/services/heart.csv")
            df = self.remove_cat_value(df)

            X = df.drop('target', axis=1)
            y = df.target

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
            # filename = f"../saved_models/xg_boost_2021_05_12_14_32_20.sav"
            # model=joblib.load(filename)
            model_conf=ModelConfiguration.objects.filter(alg_name=name).order_by("-created_date").first()
            rf_clf = RandomForestClassifier(n_estimators=model_conf.n_estimators,
                                            max_features=model_conf.max_features,max_depth=model_conf.max_depth,
                                            min_samples_split=model_conf.min_samples_split,
                                            min_samples_leaf=model_conf.min_samples_leaf,bootstrap=model_conf.bootstrap
                                            )
            rf_clf.fit(X_train, y_train)

            pred = rf_clf.predict(X_test)
            accuracy_score1=accuracy_score(y_test, pred) * 100
            precision_score1=precision_score(y_test,pred)*100
            f1_score1=f1_score(y_test,pred)*100
            recall_score1 = recall_score(y_test, pred) * 100
            roc_auc_score1=roc_auc_score(y_test,pred)*100
            print('roc score and auc score is',roc_auc_score1)
            model_conf.accuracy=accuracy_score1
            model_conf.precision=precision_score1
            model_conf.f1_score=f1_score1
            model_conf.recall_score=recall_score1
            model_conf.roc_auc_score=roc_auc_score1
            model_conf.save()

            print(X.loc[:1])
            res = rf_clf.predict(X.loc[:1])

            create_date=model_conf.created_date
            date=create_date.strftime("%m%d%Y_%H_%M_%S")

            self.save_model(rf_clf,name=f"random_forest_{date}.joblib")
            self.print_score(rf_clf, X_train, y_train, X_test, y_test, train=True)
            self.print_score(rf_clf, X_train, y_train, X_test, y_test, train=False)
        if name=='Binary Classifier':
            df = pd.read_csv("./ml_app/services/heart.csv")
            df = self.remove_cat_value(df)

            X = df.drop('target', axis=1)
            y = df.target

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
            # filename = f"../saved_models/xg_boost_2021_05_12_14_32_20.sav"
            # model=joblib.load(filename)
            model_conf=ModelConfiguration.objects.filter(alg_name=name).order_by("-created_date").first()
            lr_clf = LogisticRegression(solver=model_conf.solver,C=model_conf.c,penalty=model_conf.penalty)

            lr_clf.fit(X_train, y_train)

            pred = lr_clf.predict(X_test)
            accuracy_score1 = accuracy_score(y_test, pred) * 100
            precision_score1 = precision_score(y_test, pred) * 100
            f1_score1 = f1_score(y_test, pred) * 100
            recall_score1 = recall_score(y_test, pred) * 100
            roc_auc_score1 = roc_auc_score(y_test, pred) * 100

            print('roc score and auc score is', roc_auc_score1)

            model_conf.accuracy = accuracy_score1
            model_conf.precision = precision_score1
            model_conf.f1_score = f1_score1
            model_conf.recall_score = recall_score1
            model_conf.roc_auc_score = roc_auc_score1
            model_conf.save()


            print(X.loc[:1])
            res = lr_clf.predict(X.loc[:1])
            print('res is',res)

            create_date = model_conf.created_date
            date=create_date.strftime("%m%d%Y_%H_%M_%S")

            self.save_model(lr_clf,name=f"logistic_regression_{date}.joblib")
            self.print_score(lr_clf, X_train, y_train, X_test, y_test, train=True)
            self.print_score(lr_clf, X_train, y_train, X_test, y_test, train=False)
        if name=='Support Vector Machine':
            df = pd.read_csv("./ml_app/services/heart.csv")
            df = self.remove_cat_value(df)

            X = df.drop('target', axis=1)
            y = df.target

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
            # filename = f"../saved_models/xg_boost_2021_05_12_14_32_20.sav"
            # model=joblib.load(filename)
            model_conf=ModelConfiguration.objects.filter(alg_name=name).order_by("-created_date").first()
            svm_clf = SVC(C=model_conf.c,gamma=model_conf.gamma,kernel=model_conf.kernel)

            svm_clf.fit(X_train, y_train)


            pred = svm_clf.predict(X_test)
            accuracy_score1 = accuracy_score(y_test, pred) * 100
            precision_score1 = precision_score(y_test, pred) * 100
            f1_score1 = f1_score(y_test, pred) * 100
            recall_score1 = recall_score(y_test, pred) * 100
            roc_auc_score1 = roc_auc_score(y_test, pred) * 100

            print('roc score and auc score is', roc_auc_score1)

            model_conf.accuracy = accuracy_score1
            model_conf.precision = precision_score1
            model_conf.f1_score = f1_score1
            model_conf.recall_score = recall_score1
            model_conf.roc_auc_score = roc_auc_score1
            model_conf.save()

            print(X.loc[:1])
            res = svm_clf.predict(X.loc[:1])
            create_date = model_conf.created_date
            date=create_date.strftime("%m%d%Y_%H_%M_%S")

            self.save_model(svm_clf,name=f"support_vector_machine_{date}.joblib")
            self.print_score(svm_clf, X_train, y_train, X_test, y_test, train=True)
            self.print_score(svm_clf, X_train, y_train, X_test, y_test, train=False)

        if name=='XGB Classifier':
            df = pd.read_csv("./ml_app/services/heart.csv")
            df = self.remove_cat_value(df)

            X = df.drop('target', axis=1)
            y = df.target

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
            # filename = f"../saved_models/xg_boost_2021_05_12_14_32_20.sav"
            # model=joblib.load(filename)
            model_conf=ModelConfiguration.objects.filter(alg_name=name).order_by("-created_date").first()
            print('leafs are',model_conf.min_samples_leaf)
            rf_clf = XGBClassifier(n_estimators=model_conf.n_estimators,
                                            max_depth=model_conf.max_depth,booster=model_conf.booster,
                                            base_score=model_conf.base_score,
                                            learning_rate=model_conf.learning_rate,bootstrap=model_conf.min_child_weight
                                            )
            rf_clf.fit(X_train, y_train)
            print(X.loc[:1])

            pred = rf_clf.predict(X_test)
            accuracy_score1 = accuracy_score(y_test, pred) * 100
            precision_score1 = precision_score(y_test, pred) * 100
            f1_score1 = f1_score(y_test, pred) * 100
            recall_score1 = recall_score(y_test, pred) * 100
            roc_auc_score1 = roc_auc_score(y_test, pred) * 100

            print('roc score and auc score is', roc_auc_score1)

            model_conf.accuracy = accuracy_score1
            model_conf.precision = precision_score1
            model_conf.f1_score = f1_score1
            model_conf.recall_score = recall_score1
            model_conf.roc_auc_score = roc_auc_score1
            model_conf.save()

            res = rf_clf.predict(X.loc[:1])
            create_date = model_conf.created_date
            date=create_date.strftime("%m%d%Y_%H_%M_%S")

            self.save_model(rf_clf,name=f"xg_boost_{date}.joblib")
            self.print_score(rf_clf, X_train, y_train, X_test, y_test, train=True)
            self.print_score(rf_clf, X_train, y_train, X_test, y_test, train=False)
        return accuracy_score1,precision_score1,f1_score1,roc_auc_score1