from django.shortcuts import render
from django.http import JsonResponse, Http404
from django.conf import settings
from rest_framework.reverse import reverse
from django.contrib.auth import authenticate, get_user_model

from . import models
from .apps import MlAppConfig
from rest_framework.decorators import api_view

from django.apps import apps
from django.contrib.auth.models import User, Group
from ml_app.models import HealthRecordModel
from rest_framework import viewsets, status, mixins, generics, renderers
from rest_framework import permissions
from ml_app.serializers.user_serializers import UserSerializer
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import authentication, permissions
from django.contrib.auth.models import User
from rest_framework.authtoken.views import ObtainAuthToken
from rest_framework.authtoken.models import Token
from rest_framework.response import Response
from ml_app.serializers.health_serializer import HealthRecordSerializer, GetRecordSerializer
from .permissions import IsOwnerOrReadOnly



class HealthRecordList(generics.ListCreateAPIView):
    permission_classes = [permissions.IsAuthenticatedOrReadOnly,
                          IsOwnerOrReadOnly]
    queryset = HealthRecordModel.objects.all()
    serializer_class = HealthRecordSerializer

    def perform_create(self, serializer):
        serializer.save(user_id=self.request.user.pk)

class HealthRecordDetail(generics.RetrieveUpdateDestroyAPIView):
    permission_classes = [permissions.IsAuthenticatedOrReadOnly,
                          IsOwnerOrReadOnly]
    queryset = HealthRecordModel.objects.all()
    serializer_class = HealthRecordSerializer



class UserList(generics.ListAPIView):
    queryset = User.objects.order_by('id').all()
    serializer_class = UserSerializer
    def list(self, request, *args, **kwargs):
        queryset = self.filter_queryset(self.get_queryset())

        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)


class UserDetail(generics.RetrieveAPIView):
    queryset = User.objects.all()
    serializer_class = UserSerializer


@api_view(['GET'])
def api_root(request, format=None):
    return Response({
        'users': reverse('user-list', request=request, format=format),
        'records': reverse('record-list', request=request, format=format)
    })