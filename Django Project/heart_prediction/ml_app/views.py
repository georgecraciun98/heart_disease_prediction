from django.shortcuts import render
from django.http import JsonResponse, Http404
from django.conf import settings
from rest_framework.reverse import reverse

from . import models
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
from ml_app.models import HealthRecordModel
from rest_framework import viewsets, status, mixins, generics, renderers
from rest_framework import permissions
from ml_app.Serializers.user_serializers import UserSerializer
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import authentication, permissions
from django.contrib.auth.models import User
from rest_framework.authtoken.views import ObtainAuthToken
from rest_framework.authtoken.models import Token
from rest_framework.response import Response
from ml_app.Serializers.health_serializer import HealthRecordSerializer, GetRecordSerializer
from .permissions import IsOwnerOrReadOnly


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
        # serializer = UserSerializer
        # serializer = UserSerializer(data=request.data,
        #                                    context={'request': request})
        # serializer.is_valid(raise_exception=True)
        # user = serializer.validated_data['user']
        serializer = self.serializer_class(data=request.data,
                                           context={'request': request})
        serializer.is_valid(raise_exception=True)
        user = serializer.validated_data['user']
        print("username is {}".format(user.username))
        token, created = Token.objects.get_or_create(user)
        print('hello')
        return Response({
            'token': token.key,
            'user_id': user.pk,
        })
import io
from rest_framework.parsers import JSONParser
class HealthRecord(APIView,mixins.ListModelMixin):
    authentication_classes = [authentication.TokenAuthentication]
    permission_classes = [permissions.IsAuthenticated]
    serializer_class=HealthRecordSerializer
    from ml_app import  models
    def get_object(self, pk):
        try:
            return models.HealthRecordModel.objects.get(pk=pk)
        except models.HealthRecordModel.DoesNotExist:
            raise Http404
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
    # def get(self, request, *args, **kwargs):
    #     all=models.HealthRecord.objects.all()
    #     print('records',[all])
    #     serializer=GetRecordSerializer(data=[all],many=True)
    #     if serializer.is_valid():
    #        print('is done')
    #        serializer.create(request.data)
    #
    #        return Response(serializer.validated_data,status=status.HTTP_200_OK)
    #     return Response({"message":"SORRY"}, status=status.HTTP_400_BAD_REQUEST)


class HealthRecordList(generics.ListCreateAPIView):
    permission_classes = [permissions.IsAuthenticatedOrReadOnly,
                          IsOwnerOrReadOnly]
    queryset = HealthRecordModel.objects.all()
    serializer_class = HealthRecordSerializer

    def perform_create(self, serializer):
        serializer.save(user_id=self.request.user)

class HealthRecordDetail(generics.CreateAPIView):
    permission_classes = [permissions.IsAuthenticatedOrReadOnly,
                          IsOwnerOrReadOnly]
    queryset = HealthRecordModel.objects.all()
    serializer_class = HealthRecordSerializer

# class Recordhighlight(generics.GenericAPIView):
#     queryset = HealthRecordModel.objects.all()
#     renderer_classes = [renderers.StaticHTMLRenderer]
#
#     def get(self, request, *args, **kwargs):
#         record = self.get_object()
#         return Response(record.highlighted)


class UserList(generics.ListAPIView):
    queryset = User.objects.all()
    serializer_class = UserSerializer


class UserDetail(generics.RetrieveAPIView):
    queryset = User.objects.all()
    serializer_class = UserSerializer


@api_view(['GET'])
def api_root(request, format=None):
    return Response({
        'users': reverse('user-list', request=request, format=format),
        'records': reverse('record-list', request=request, format=format)
    })