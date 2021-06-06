from rest_framework import serializers
from ml_app.models import Appointments
from django.contrib.auth.models import User

from ml_app.serializers.patient_serializers import PatientUserSerializer, PatientSerializer
from ml_app.serializers.user_serializers import UserSerializer


class AppointmentSerializer(serializers.ModelSerializer):
    doctor_id=serializers.IntegerField(required=True)
    patient_id=serializers.IntegerField(required=True)
    class Meta:
        model = Appointments
        fields = ['patient_id', 'doctor_id','time']

class AppointmentDoctors(serializers.ModelSerializer):
    doctor=UserSerializer()
    patient = PatientSerializer()
    # patient_id=serializers.IntegerField(required=True)
    class Meta:
        model = Appointments
        fields = ['id','patient', 'doctor','time']

class AppointmentDetailedSerializer(serializers.ModelSerializer):
    doctor_id=serializers.IntegerField(required=True)
    patient = PatientSerializer()
    # patient_id=serializers.IntegerField(required=True)
    class Meta:
        model = Appointments
        fields = ['id','patient', 'doctor_id','time']

class DoctorSerializer(serializers.ModelSerializer):


    class Meta:
        model=User
        fields=['username','first_name','last_name','']