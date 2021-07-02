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
        self.continous_val=['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

    def save_model(self,model,name):
        full_path = "./models/"
        dump(model,full_path+ name)
    """
    Print model performance on the screen
    """
    def get_performance(self, model, x_train, y_train, x_test, y_test, train=True):
        if train:
            pred = model.predict(x_train)
            print("Train Result:\n================================================")
            print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
            print(f"Precision Score: {accuracy_score(y_train, pred) * 100:.2f}%")
            print(f"F1-Score: {accuracy_score(y_train, pred) * 100:.2f}%")
            print(f"Roc Auc Score: {accuracy_score(y_train, pred) * 100:.2f}%")
            print(f"Recall Score: {accuracy_score(y_train, pred) * 100:.2f}%")

        elif train == False:
            pred = model.predict(x_test)
            print("Train Result:\n================================================")
            print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
            print(f"Precision Score: {accuracy_score(y_test, pred) * 100:.2f}%")
            print(f"F1-Score: {accuracy_score(y_test, pred) * 100:.2f}%")
            print(f"Roc Auc Score: {accuracy_score(y_test, pred) * 100:.2f}%")
            print(f"Recall Score: {accuracy_score(y_test, pred) * 100:.2f}%")
    """
    Remove categorical values
    """
    def remove_cat_value(self,dataframe):
        cat_values = []
        con_values = []
        for col in dataframe.columns:
            if len(dataframe[col].unique()) <= 6:
                cat_values.append(col)
            else:
                con_values.append(col)
        cat_values.remove('target')
        ds = pd.get_dummies(dataframe, columns=cat_values)
        s_sc = StandardScaler()
        to_scale = self.continous_val
        ds[to_scale]= s_sc.fit_transform(ds[to_scale])
        dump(s_sc, 'std_scaler.bin',compress=True)
        return ds

    def train_model(self,name,data,researcher_id):
        pass
        if name=='Random Forest Classifier':
            dataframe = pd.read_csv("./ml_app/services/heart.csv")
            dataframe = self.remove_cat_value(dataframe)
            X = dataframe.drop('target', axis=1)
            y = dataframe.target
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            model_conf=ModelConfiguration.objects.filter(alg_name=name).order_by("-created_date").first()
            random_forest = RandomForestClassifier(n_estimators=model_conf.n_estimators,
                                            max_features=model_conf.max_features,max_depth=model_conf.max_depth,
                                            min_samples_split=model_conf.min_samples_split,
                                            min_samples_leaf=model_conf.min_samples_leaf,bootstrap=model_conf.bootstrap
                                            )
            random_forest.fit(X_train, y_train)

            pred = random_forest.predict(X_test)
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
            model_conf.save()
            res = random_forest.predict(X.loc[:1])
            create_date=model_conf.created_date
            date=create_date.strftime("%m%d%Y_%H_%M_%S")

            self.save_model(random_forest,name=f"random_forest_{date}.joblib")
            self.get_performance(random_forest, X_train, y_train, X_test, y_test, train=True)
            self.get_performance(random_forest, X_train, y_train, X_test, y_test, train=False)
        if name=='Binary Classifier':
            dataframe = pd.read_csv("./ml_app/services/heart.csv")
            dataframe = self.remove_cat_value(dataframe)

            X = dataframe.drop('target', axis=1)
            y = dataframe.target

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            # filename = f"../saved_models/xg_boost_2021_05_12_14_32_20.sav"
            # model=joblib.load(filename)
            model_conf=ModelConfiguration.objects.filter(alg_name=name).order_by("-created_date").first()
            logistic_regression = LogisticRegression(solver=model_conf.solver,C=model_conf.c,penalty=model_conf.penalty)

            logistic_regression.fit(X_train, y_train)

            pred = logistic_regression.predict(X_test)
            accuracy_score1 = accuracy_score(y_test, pred) * 100
            precision_score1 = precision_score(y_test, pred) * 100
            f1_score1 = f1_score(y_test, pred) * 100
            recall_score1 = recall_score(y_test, pred) * 100
            roc_auc_score1 = roc_auc_score(y_test, pred) * 100

            model_conf.accuracy = accuracy_score1
            model_conf.precision = precision_score1
            model_conf.f1_score = f1_score1
            model_conf.recall_score = recall_score1
            model_conf.roc_auc_score = roc_auc_score1
            model_conf.save()

            res = logistic_regression.predict(X.loc[:1])

            create_date = model_conf.created_date
            date=create_date.strftime("%m%d%Y_%H_%M_%S")

            self.save_model(logistic_regression,name=f"logistic_regression_{date}.joblib")
            self.get_performance(logistic_regression, X_train, y_train, X_test, y_test, train=True)
            self.get_performance(logistic_regression, X_train, y_train, X_test, y_test, train=False)
        if name=='Support Vector Machine':
            dataframe = pd.read_csv("./ml_app/services/heart.csv")
            dataframe = self.remove_cat_value(dataframe)

            X = dataframe.drop('target', axis=1)
            y = dataframe.target

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            # filename = f"../saved_models/xg_boost_2021_05_12_14_32_20.sav"
            # model=joblib.load(filename)
            model_conf=ModelConfiguration.objects.filter(alg_name=name).order_by("-created_date").first()
            svm = SVC(C=model_conf.c,gamma=model_conf.gamma,kernel=model_conf.kernel)

            svm.fit(X_train, y_train)


            pred = svm.predict(X_test)
            accuracy_score1 = accuracy_score(y_test, pred) * 100
            precision_score1 = precision_score(y_test, pred) * 100
            f1_score1 = f1_score(y_test, pred) * 100
            recall_score1 = recall_score(y_test, pred) * 100
            roc_auc_score1 = roc_auc_score(y_test, pred) * 100

            model_conf.accuracy = accuracy_score1
            model_conf.precision = precision_score1
            model_conf.f1_score = f1_score1
            model_conf.recall_score = recall_score1
            model_conf.roc_auc_score = roc_auc_score1
            model_conf.save()

            res = svm.predict(X.loc[:1])
            create_date = model_conf.created_date
            date=create_date.strftime("%m%d%Y_%H_%M_%S")

            self.save_model(svm,name=f"support_vector_machine_{date}.joblib")
            self.get_performance(svm, X_train, y_train, X_test, y_test, train=True)
            self.get_performance(svm, X_train, y_train, X_test, y_test, train=False)

        if name=='XGB Classifier':
            dataframe = pd.read_csv("./ml_app/services/heart.csv")
            dataframe = self.remove_cat_value(dataframe)

            X = dataframe.drop('target', axis=1)
            y = dataframe.target

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            model_conf=ModelConfiguration.objects.filter(alg_name=name).order_by("-created_date").first()
            xgb = XGBClassifier(n_estimators=model_conf.n_estimators,
                                            max_depth=model_conf.max_depth,booster=model_conf.booster,
                                            base_score=model_conf.base_score,
                                            learning_rate=model_conf.learning_rate,bootstrap=model_conf.min_child_weight
                                            )
            xgb.fit(X_train, y_train)

            pred = xgb.predict(X_test)
            accuracy_score1 = accuracy_score(y_test, pred) * 100
            precision_score1 = precision_score(y_test, pred) * 100
            f1_score1 = f1_score(y_test, pred) * 100
            recall_score1 = recall_score(y_test, pred) * 100
            roc_auc_score1 = roc_auc_score(y_test, pred) * 100

            model_conf.accuracy = accuracy_score1
            model_conf.precision = precision_score1
            model_conf.f1_score = f1_score1
            model_conf.recall_score = recall_score1
            model_conf.roc_auc_score = roc_auc_score1
            model_conf.save()

            res = xgb.predict(X.loc[:1])
            create_date = model_conf.created_date
            date=create_date.strftime("%m%d%Y_%H_%M_%S")

            self.save_model(xgb,name=f"xg_boost_{date}.joblib")
            self.get_performance(xgb, X_train, y_train, X_test, y_test, train=True)
            self.get_performance(xgb, X_train, y_train, X_test, y_test, train=False)
        return accuracy_score1,precision_score1,f1_score1,roc_auc_score1