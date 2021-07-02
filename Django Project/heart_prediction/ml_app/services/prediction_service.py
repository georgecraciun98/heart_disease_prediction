from django.http import Http404
from ml_app.submodels.health_record import HealthRecordModel

from ml_app.submodels.model_configuration import ModelConfiguration
import pandas as pd
from os import path
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers.kernelized import RandomFourierFeatures
import tensorflow as tf

import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load

full_path = r"S:\School\Licenta\Github Code\licence_machine_learning\Django Project\heart_prediction\models"
full_path_svm=r'S:\School\Licenta\Github Code\licence_machine_learning\Django Project\heart_prediction\models\svm_model'

class PredictionService:
    def __init__(self):
        self.categorical_val=['sex','cp','fbs','restecg','exang','slope','ca','thal']
    def get_record(self, pk):
        try:
            return HealthRecordModel.objects.order_by('created_data').get(id=pk)
        except self.queryset.model.DoesNotExist:
            raise Http404
    def svm_loading_1(self):
        input_shape=13
        model = keras.Sequential(
            [
                layers.Dense(20, input_shape=(input_shape,)),
                RandomFourierFeatures(
                    output_dim=4096, kernel_initializer="gaussian"
                ),
                layers.Dense(units=1, activation='sigmoid'),
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
    def make_prediction(self,record_id,model_id):
        name = 'ml_app'
        # load your models and model weights here and these are linked      "MODELS" variable that we mentioned in "settings.py"
        #path = os.path.join(settings.MODELS, "xg_boost_sklearn.joblib")
        # path= r"S:\School\Licenta\Github Code\licence_machine_learning\Django Project\heart_prediction\models\xg_boost_sklearn.joblib"
        # loaded_model = load(path)
        model_name=ModelConfiguration.objects.get(id=model_id)
        input_shape=30
        created_date=model_name.created_date
        model=""
        if(model_name.alg_name=='SVM_KERAS'):
            model = self.svm_loading_1(created_date)
        elif (model_name.alg_name == 'Support Vector Machine'):
            model = self.svm_loading_2(created_date)
        elif(model_name.alg_name=='Random Forest Classifier'):
            model = self.random_forest_2(created_date)
        elif (model_name.alg_name == 'XGB Classifier'):
            model = self.xg_loading_2(created_date)
        elif (model_name.alg_name == 'Binary_Classifier'):
            model = self.binary_loading_2()
        health_model=HealthRecordModel.objects.filter(pk=record_id)
        #split the values
        # values=health_model.values('age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
        #                            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal')
        values = health_model.values('age', 'trestbps', 'chol',
                                      'thalach', 'oldpeak', 'sex_0',
                                     'sex_1', 'cp_0', 'cp_1', 'cp_2', 'cp_3', 'fbs_0', 'fbs_1', 'restecg_0',
                                     'restecg_1', 'restecg_2', 'exang_0', 'exang_1', 'slope_0', 'slope_1',
                                     'slope_2', 'ca_0', 'ca_1', 'ca_2', 'ca_3', 'ca_4', 'thal_0', 'thal_1',
                                     'thal_2', 'thal_3')
        # print(list(values))
        # list_initial=list(values)
        # print(type(list_initial))
        # list_initial[0].append({"a":"0"})

        df = pd.DataFrame(list(values))
        scaler = load(open('scaler.pkl', 'rb'))
        col_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        df[col_to_scale] = scaler.transform(df[col_to_scale])
        print('df is',df[col_to_scale])
        # df_data = pd.get_dummies(df, columns=self.categorical_val)
        y_pred=model.predict(df)
        if(isinstance(y_pred, (np.ndarray, np.generic) )):
            print('original is',y_pred)
            print(y_pred[0])
            y_pred=y_pred[0]
        print('prediction is done using ',model_name.alg_name,'record id is',record_id,'result is',y_pred,type(y_pred))
        return y_pred

    def svm_loading(self):

        # model = keras.models.load_model(path.join(full_path,'svm_model'))
        model = keras.models.load_model(full_path_svm)

        return model

    def xg_boost_loading(self,input_shape):

        clf = load(path.join(full_path,'xg_boost.sav'))
        return clf

    def random_forest_loading(self,input_shape):
        clf = load(path.join(full_path,'random_forest.sav'))
        return clf
    def binary_classifier(self,input_shape):
        clf = load(path.join(full_path,'random_forest.sav'))
        return clf
    def xg_loading(self):
        model = XGBClassifier(n_estimators=1600, learning_rate=0.01, max_depth=10, gamma=0, use_label_encoder=False)
        return model


    def random_loading(self):
        clf = RandomForestClassifier(bootstrap=False, max_depth=20, max_features='sqrt',
                                     min_samples_leaf=4, min_samples_split=5,
                                     n_estimators=800)

        return clf
    def binary_loading(self):
        input_shape = 13
        model = Sequential()
        model.add(Dense(input_shape, input_shape=(input_shape,)))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam',
                      metrics=[
                          tf.keras.metrics.BinaryAccuracy(
                              name="accuracy", dtype=None, threshold=0.5),
                          tf.keras.metrics.Precision(
                              name='precision', thresholds=0.5),
                          tf.keras.metrics.Recall(
                              name='recall', thresholds=0.5),

                      ]
                      )
        return model



    def svm_loading_2(self,created_date):
        input_shape=13
        date = created_date.strftime("%m%d%Y_%H_%M_%S")
        clf = load(path.join(full_path,f"support_vector_machine_{date}.joblib"))
        return clf

    def xg_loading_2(self,created_date):
        input_shape=13
        date = created_date.strftime("%m%d%Y_%H_%M_%S")
        clf = load(path.join(full_path,f"random_forest_{date}.joblib"))
        clf = load(path.join(full_path,'xg_boost_sklearn.joblib'))
        return clf

    def random_forest_2(self,created_date):
        input_shape=13
        date = created_date.strftime("%m%d%Y_%H_%M_%S")
        clf = load(path.join(full_path,f"random_forest_{date}.joblib"))
        return clf

    def binary_loading_2(self,created_date):
        input_shape=13
        date = created_date.strftime("%m%d%Y_%H_%M_%S")
        clf = load(path.join(full_path,f"binary_classifier_{date}.joblib"))
        return clf