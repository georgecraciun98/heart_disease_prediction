from django.contrib.auth.models import User
from rest_framework import generics, status
from rest_framework import permissions
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.reverse import reverse

from ml_app.submodels.user_details import  UserDetailModel
from ml_app.serializers.health_serializer import HealthRecordSerializer
from ml_app.serializers.user_serializers import UserSerializer, UserDetailSerializer

from django.http import Http404




class UserDetailView(generics.RetrieveUpdateAPIView,generics.CreateAPIView):
    model = UserDetailModel
    queryset = UserDetailModel.objects.all()
    serializer_class = UserDetailSerializer

    def get_object(self, pk):
        try:
            return UserDetailModel.objects.get(user_id=pk)
        except self.queryset.model.DoesNotExist:
            raise Http404

    def get(self, request, pk, format=None):
        user = self.get_object(request.user.pk)
        serializer = UserDetailSerializer(user)
        return Response(serializer.data)

    def put(self, request, pk, format=None):
        try:
            user = self.get_object(request.user.pk)
            serializer = UserDetailSerializer(user, data=request.data)
            print('1')
        except Http404:
            data=request.data
            data['user_id']=request.user.pk
            print(data)
            serializer=UserDetailSerializer(data=data)
            print('2')

        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)