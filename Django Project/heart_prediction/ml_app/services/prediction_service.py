import os

from django.http import Http404

from django.conf import settings
from ml_app.submodels.health_record import HealthRecordModel
from mlflow.sklearn import load_model
from joblib import dump, load
import pandas as pd

from ml_app.submodels.model_configuration import ModelConfiguration
import pandas as pd
from os import path


from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers.kernelized import RandomFourierFeatures
import tensorflow as tf
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load

full_path = r"S:\School\Licenta\Github Code\licence_machine_learning\Django Project\heart_prediction\models"


class PredictionService:
    def get_record(self, pk):
        try:
            return HealthRecordModel.objects.order_by('created_data').get(id=pk)
        except self.queryset.model.DoesNotExist:
            raise Http404
    def make_prediction(self,record_id,model_id):
        name = 'ml_app'
        # load your models and model weights here and these are linked      "MODELS" variable that we mentioned in "settings.py"
        #path = os.path.join(settings.MODELS, "xg_boost_sklearn.joblib")
        path= r"S:\School\Licenta\Github Code\licence_machine_learning\Django Project\heart_prediction\models\xg_boost_sklearn.joblib"
        loaded_model = load(path)
        model_name=ModelConfiguration.objects.get(id=model_id)
        input_shape=30
        model=""
        if(model_name.alg_name=='SVM_KERAS'):
            model = model = keras.models.load_model('')
        elif(model_name.alg_name=='Random_Forest'):
            model = self.random_forest_loading(input_shape)
        elif (model_name.alg_name == 'XGB_Classifier'):
            model = self.xg_boost_loading(input_shape)

        health_model=HealthRecordModel.objects.filter(pk=record_id)

        values=health_model.values('age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                                   'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal')
        df = pd.DataFrame(list(values))
        y_pred=model.predict(df)
        print('prediction is done using ',model_name.alg_name)
        return y_pred

    def svm_loading(self,input_shape):

        model = keras.models.load_model(path.join(full_path,'svm_model.h5'))
        return model

    def xg_boost_loading(self,input_shape):

        clf = load(path.join(full_path,'xg_boost.sav'))
        return clf

    def random_forest_loading(self,input_shape):
        clf = load(path.join(full_path,'random_forest.sav'))
        return clf