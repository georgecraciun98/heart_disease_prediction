from django.contrib.auth.models import User
from rest_framework import serializers
from ml_app.models import HealthRecordModel

class UserSerializer(serializers.ModelSerializer):
    records = serializers.PrimaryKeyRelatedField(many=True, queryset=HealthRecordModel.objects.all())

    class Meta:
        model = User
        fields = ['id', 'username', 'records']