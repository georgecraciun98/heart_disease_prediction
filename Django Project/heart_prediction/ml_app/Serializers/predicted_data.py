
from rest_framework import serializers

from ml_app.models import HealthRecordModel, PredictedData


class PredictedDataSerializer(serializers.ModelSerializer):
    #user_id=serializers.PrimaryKeyRelatedField(many=True,read_only=True)
    record_id = serializers.IntegerField(required=False)
    average_thalach=serializers.IntegerField(required=False)
    start_time=serializers.DateTimeField(required=False)
    end_time=serializers.DateTimeField(required=False)
    photo = serializers.ImageField(required=False)
    target=serializers.IntegerField(required=False)

    def create(self, validated_data):
        print('validated data is',validated_data)
        return PredictedData.objects.create(**validated_data)
    class Meta:
        model=HealthRecordModel
        fields=('model_id','record_id','average_thalach','start_time','end_time','photo','target')