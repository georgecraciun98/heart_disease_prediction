from django.db import models

from ml_app.submodels.patient_model import Patient
from django.contrib.auth.models import User

class DoctorPatients(models.Model):


    patient = models.ForeignKey(Patient,to_field='id', related_name='patients', on_delete=models.CASCADE)
    doctor=models.ForeignKey(User,to_field='id',related_name='doctors',on_delete=models.CASCADE)

    def __str__(self):
        user=Patient.objects.get(pk=self.patient_id)
        auth_user=User.objects.get(pk=user.user_id)
        return 'Patient with name {} and doctor id {}  '.format(auth_user.first_name,self.doctor_id)

