from django.contrib.auth.models import User, Group
from rest_framework import serializers

from ml_app.models import HealthRecordModel, Patient
from ml_app.submodels.monitored_data import MonitoredData


class MonitoredDataSerializer(serializers.ModelSerializer):


    class Meta:
        model = MonitoredData
        fields = ['id', 'api_value','start_time','end_time','activity_description',
                  'activity_source','created_date','patient_id','data_type']