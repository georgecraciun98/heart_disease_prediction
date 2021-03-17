from django.contrib.auth.models import User, Group
from rest_framework import serializers

from ml_app.models import HealthRecordModel, UserDetailModel


class GroupSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Group
        fields = ('name',)

# class AuthGroupSerializer(serializers.ModelSerializer):
#     class Meta:
#         model = AuthGroup
#         fields = ('name',)

class UserSerializer(serializers.ModelSerializer):

    records = serializers.PrimaryKeyRelatedField(many=True,required=False, queryset=HealthRecordModel.objects.all())
    groups = GroupSerializer(many=True,required=False)
    class Meta:
        model = User
        fields = ['id', 'username', 'records','groups']
        extra_kwargs = {'password': {'write_only': True}}

    def create(self, validated_data):
        password = validated_data.pop('password')
        user = User(**validated_data)
        user.set_password(password)
        user.save()
        return user
class UserDetailSerializer(serializers.ModelSerializer):
    user_id = serializers.IntegerField(required=False)
    class Meta:
        model = UserDetailModel
        fields= ['sex','birth_date','user_id','description']