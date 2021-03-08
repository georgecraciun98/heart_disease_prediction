
from django.contrib.auth.models import User, Group
from rest_framework.authtoken.models import Token
from ml_app.submodels.model_configuration import ModelConfiguration


from ml_app.submodels.doctor_patients import *
from ml_app.submodels.health_record import *
from ml_app.submodels.model_configuration import *
from ml_app.submodels.monitored_data import *
from ml_app.submodels.predicted_data import *
from ml_app.submodels.user_details import *


class PredictedData(models.Model):
    model_id = models.ForeignKey(ModelConfiguration, related_name='models',
                                 on_delete=models.DO_NOTHING)
    record_id = models.ForeignKey(User, related_name='records', on_delete=models.CASCADE)
    average_thalach=models.IntegerField()
    start_time=models.DateField()
    end_time=models.DateField()
    target=models.IntegerField(choices=BinaryChoices.choices)

    def __str__(self):

        return 'Patient with average thalach {} '.format(self.average_thalach)
