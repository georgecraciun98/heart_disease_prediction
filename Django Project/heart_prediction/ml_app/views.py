from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from .apps import MlAppConfig
from rest_framework.decorators import api_view
import os
import numpy as np
import pandas as pd
import time
import glob
import requests
from scipy import ndimage
from scipy.ndimage import zoom
from django.apps import apps
from django.contrib.auth.models import User, Group
from rest_framework import viewsets,status
from rest_framework import permissions
from ml_app.Serializers.user_serializers import UserSerializer
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import authentication, permissions
from django.contrib.auth.models import User
from rest_framework.authtoken.views import ObtainAuthToken
from rest_framework.authtoken.models import Token
from rest_framework.response import Response
from ml_app.Serializers.health_record_serializer import HealthRecordSerializer
@api_view(["POST"])
def check_result(request):
    #Get video file url
    url = request.POST.get('url')
    print(url)

    emo_jso={'Url':url,'Feeling':'Good'}
    model=apps.get_app_config('ml_app').loaded_model
    print(model.summary())
    return JsonResponse(emo_jso, safe=False)




class UserViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """
    queryset = User.objects.all().order_by('-date_joined')
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticated]




class ListUsers(APIView):
    """
    View to list all users in the system.

    * Requires token authentication.
    * Only admin users are able to access this view.
    """
    authentication_classes = [authentication.TokenAuthentication]
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, format=None):
        """
        Return a list of all users.
        """
        usernames = [user.username for user in User.objects.all()]
        return Response(usernames)

class CustomAuthToken(ObtainAuthToken):

    def post(self, request, *args, **kwargs):

        serializer = self.serializer_class(data=request.data,
                                           context={'request': request})
        serializer.is_valid(raise_exception=True)
        user = serializer.validated_data['user']
        token, created = Token.objects.get_or_create(user=user)
        return Response({
            'token': token.key,
            'user_id': user.pk,
            'email': user.email
        })
import io
from rest_framework.parsers import JSONParser
class HealthRecord(APIView):
    authentication_classes = [authentication.TokenAuthentication]
    permission_classes = [permissions.IsAuthenticated]
    serializer_class=HealthRecordSerializer

    def post(self, request, *args, **kwargs):
        user=request.user

        json=request.data
        json['user_id']=request.user.pk
        print(type(json))
        print('request data',json)
        serializer = self.serializer_class(data=json,
                                           context={'request': request})
        if serializer.is_valid():
           print('is done')
           serializer.create(json)
           return Response(serializer.validated_data,status=status.HTTP_201_CREATED)
        print('not good')
        return Response(json, status=status.HTTP_400_BAD_REQUEST)