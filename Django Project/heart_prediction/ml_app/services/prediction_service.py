import os

from django.http import Http404

from django.conf import settings
from ml_app.submodels.health_record import HealthRecordModel
from mlflow.sklearn import load_model
from joblib import dump, load
import pandas as pd
class PredictionService:
    def get_record(self, pk):
        try:
            return HealthRecordModel.objects.order_by('created_data').get(id=pk)
        except self.queryset.model.DoesNotExist:
            raise Http404
    def make_prediction(self,record_id):
        name = 'ml_app'
        # load your models and model weights here and these are linked      "MODELS" variable that we mentioned in "settings.py"
        path = os.path.join(settings.MODELS, "xg_boost_sklearn.joblib")
        loaded_model = load(r"S:\School\Licenta\Github Code\licence_machine_learning\Django Project\heart_prediction\models\xg_boost_sklearn.joblib")
        health_model=HealthRecordModel.objects.filter(pk=record_id)

        values=health_model.values('age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                                   'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal')
        df = pd.DataFrame(list(values))
        y_pred=loaded_model.predict(df)
        print('prediction is done using ',y_pred)
        # #loaded_model.load_weights(path)
        return loaded_model

