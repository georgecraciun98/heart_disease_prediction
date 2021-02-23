from django.contrib.auth.models import User
from rest_framework import serializers

from ml_app.models import HealthRecordModel


class UserSerializer(serializers.ModelSerializer):
    records = serializers.PrimaryKeyRelatedField(many=True, queryset=HealthRecordModel.objects.all())

    class Meta:
        model = User
        fields = ['id', 'username', 'records']
        extra_kwargs = {'password': {'write_only': True}}

    def create(self, validated_data):
        password = validated_data.pop('password')
        user = User(**validated_data)
        user.set_password(password)
        user.save()
        return user