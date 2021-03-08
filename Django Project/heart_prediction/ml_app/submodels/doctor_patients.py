from django.db import models

from ml_app.submodels.user_details import UserDetailModel
from django.contrib.auth.models import User

class DoctorPatients(models.Model):


    patient_id = models.ForeignKey(UserDetailModel, related_name='patients', on_delete=models.CASCADE)
    doctor_id=models.ForeignKey(User,related_name='doctors',on_delete=models.CASCADE)

    def __str__(self):
        user=UserDetailModel.objects.get(pk=self.patient_id)
        auth_user=User.objects.get(pk=user.user_id)
        return 'Patient with name {} and doctor id {}  '.format(auth_user.first_name,self.doctor_id)
