
from rest_framework import serializers
from ml_app.models import  ModelConfiguration

class ModelSerializer(serializers.ModelSerializer):

    class Meta:
        model= ModelConfiguration
        fields=('id','alg_name','alg_description','researcher_id')