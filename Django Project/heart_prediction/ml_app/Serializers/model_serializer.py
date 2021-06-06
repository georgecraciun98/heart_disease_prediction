
from rest_framework import serializers
from ml_app.models import  ModelConfiguration

class ModelSerializer(serializers.ModelSerializer):
    researcher_id = serializers.IntegerField(required=False)

    class Meta:
        model= ModelConfiguration
        fields=('id','alg_name','alg_description','researcher_id',
                'n_estimators','max_depth','booster','base_score',
                'learning_rate','min_child_weight','max_features','min_samples_split',
                'min_samples_leaf','bootstrap')