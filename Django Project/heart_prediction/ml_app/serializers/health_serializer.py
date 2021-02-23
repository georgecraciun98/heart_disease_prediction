from ml_app.models import HealthRecordModel
from rest_framework import serializers
from rest_framework import serializers

from ml_app.models import HealthRecordModel


class UserFilteredPrimaryKeyRelatedField(serializers.PrimaryKeyRelatedField):
    def get_queryset(self):
        request = self.context.get('request', None)
        queryset = super(UserFilteredPrimaryKeyRelatedField, self).get_queryset()
        if not request or not queryset:
            return None
        return queryset.filter(user=request.user)

class HealthRecordSerializer(serializers.ModelSerializer):
    #user_id=serializers.PrimaryKeyRelatedField(many=True,read_only=True)
    user_id = serializers.IntegerField()

    class Meta:
        model=HealthRecordModel
        fields=('age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang',
                'oldpeak','slope','ca','thal','target','user_id')

class GetRecordSerializer(serializers.ModelSerializer):

    class Meta:
        model=HealthRecordModel
        fields=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang',
                'oldpeak','slope','ca','thal','target']