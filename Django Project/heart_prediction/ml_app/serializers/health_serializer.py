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
    doctor_patients_id = serializers.IntegerField(required=False)
    age=serializers.IntegerField()
    sex=serializers.IntegerField()
    def create(self, validated_data):
        print('validated data is',validated_data)
        return HealthRecordModel.objects.create(**validated_data)
    class Meta:
        model=HealthRecordModel
        fields=('cp','trestbps','chol','fbs','restecg','thalach','exang',
                'oldpeak','slope','ca','thal','doctor_patients_id','age','sex')



