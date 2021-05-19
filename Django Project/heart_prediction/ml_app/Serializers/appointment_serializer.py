from rest_framework import serializers
from ml_app.models import Appointments
from django.contrib.auth.models import User

from ml_app.serializers.patient_serializers import PatientUserSerializer


class AppointmentSerializer(serializers.ModelSerializer):
    doctor=serializers.IntegerField(required=True)
    patient=serializers.IntegerField(required=True)
    class Meta:
        model = Appointments
        fields = ['patient', 'doctor','time']

class DoctorSerializer(serializers.ModelSerializer):


    class Meta:
        model=User
        fields=['username','first_name','last_name','']