from ml_app.Serializers.user_serializers import UserSerializer
from ml_app.models import HealthRecord
from django.contrib.auth.models import User
from rest_framework import serializers

class UserFilteredPrimaryKeyRelatedField(serializers.PrimaryKeyRelatedField):
    def get_queryset(self):
        request = self.context.get('request', None)
        queryset = super(UserFilteredPrimaryKeyRelatedField, self).get_queryset()
        if not request or not queryset:
            return None
        return queryset.filter(user=request.user)

class HealthRecordSerializer(serializers.ModelSerializer):
    user_id=serializers.PrimaryKeyRelatedField(many=True,read_only=True)
    print(user_id)
    class Meta:
        model=HealthRecord
        fields=('age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang',
                'oldpeak','slope','ca','thal','target','user_id')

class GetRecordSerializer(serializers.ModelSerializer):

    class Meta:
        model=HealthRecord
        fields=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang',
                'oldpeak','slope','ca','thal','target']