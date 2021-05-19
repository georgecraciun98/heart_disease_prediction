
from django.contrib.auth.models import User, Group
from rest_framework.authtoken.models import Token
from ml_app.submodels.model_configuration import ModelConfiguration


from ml_app.submodels.doctor_patients import *
from ml_app.submodels.health_record import *
from ml_app.submodels.model_configuration import *
from ml_app.submodels.monitored_data import *
from ml_app.submodels.patient_model import *
from ml_app.submodels.auth_user import *
from ml_app.submodels.appointments import *

class PredictedData(models.Model):
    model = models.ForeignKey(ModelConfiguration, related_name='models',
                                 on_delete=models.DO_NOTHING,null=True)
    record = models.ForeignKey(HealthRecordModel, related_name='records', on_delete=models.CASCADE)
    average_thalach=models.IntegerField(null=True)
    start_time=models.DateField(null=True)
    end_time=models.DateField(null=True)
    target=models.IntegerField(choices=BinaryChoices.choices)
    photo = models.ImageField(null=True)
    created_time=models.DateTimeField(null=True,auto_now_add=True)
    def __str__(self):

        return 'Patient with average thalach {} '.format(self.average_thalach)
