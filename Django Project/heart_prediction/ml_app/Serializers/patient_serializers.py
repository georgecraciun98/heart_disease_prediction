from django.contrib.auth.models import User, Group
from rest_framework import serializers

from ml_app.models import HealthRecordModel, Patient



class PatientUserSerializer(serializers.ModelSerializer):

    class Meta:
        model = User
        fields = ['id', 'username','first_name','last_name','last_login']



class PatientSerializer(serializers.ModelSerializer):

    user=PatientUserSerializer()

    class Meta:
        model = Patient
        fields = ['id','user', 'sex', 'birth_date','description']

class PatientDetailSerializer(serializers.ModelSerializer):
    user = PatientUserSerializer()

    class Meta:
        model = Patient
        fields= ['user','sex','birth_date','description']