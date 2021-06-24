from django.contrib.auth.models import User, Group
from rest_framework import serializers

from ml_app.models import HealthRecordModel, Patient


class GroupSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Group
        fields = ('name',)

class UserSerializer(serializers.ModelSerializer):
    groups = GroupSerializer(many=True, required=False)

    class Meta:
        model = User
        fields = ['id', 'username', 'groups']
        extra_kwargs = {'password': {'write_only': True}}

    def create(self, validated_data):
        password = validated_data.pop('password')
        user = User(**validated_data)
        user.set_password(password)
        user.save()
        return user

class HelperUserSerializer(serializers.ModelSerializer):

    class Meta:
        model = User
        fields = [ 'first_name','last_name','username']


class UserDetailSerializer(serializers.ModelSerializer):
    user_id = serializers.IntegerField(required=False)
    user = HelperUserSerializer(required=False)
    class Meta:
        model = Patient
        fields= ['user','sex','birth_date','user_id','description']

    def to_representation(self, obj):
        """Move fields from user to user representation."""
        representation = super().to_representation(obj)
        user_representation = representation.pop('user')
        for key in user_representation:
            representation[key] = user_representation[key]

        return representation

    def to_internal_value(self, data):
        """Move fields related to user to their own user dictionary."""
        user_internal = {}
        for key in HelperUserSerializer.Meta.fields:
            if key in data:
                user_internal[key] = data.pop(key)

        internal = super().to_internal_value(data)
        internal['user'] = user_internal
        return internal

    def update(self, instance, validated_data):
        """Update user and user. Assumes there is a user for every user."""
        user_data = validated_data.pop('user')
        super().update(instance, validated_data)

        user = instance.user
        for attr, value in user_data.items():
            setattr(user, attr, value)
        user.save()

        return instance