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
    doctor_patients_id = serializers.IntegerField(required=False)
    def create(self, validated_data):
        print('validated data is',validated_data)
        return HealthRecordModel.objects.create(**validated_data)
    class Meta:
        model=HealthRecordModel
        fields=('id','sex_0','sex_1','cp_0','cp_1','cp_2','cp_3',
                'fbs_0','fbs_1','restecg_0','restecg_1','restecg_2','exang_0',
                'exang_1','slope_0','slope_1','slope_2',
                'ca_0','ca_1','ca_2','ca_3','ca_4','thal_0','thal_1','thal_2','thal_3',
                'trestbps','created_data','chol','thalach',
                'oldpeak','doctor_patients_id','age')



