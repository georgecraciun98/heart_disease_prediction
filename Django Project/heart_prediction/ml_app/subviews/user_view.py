from django.contrib.auth.models import User
from rest_framework import generics, status
from rest_framework import permissions
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.reverse import reverse
from ml_app.sub_permissions.group_permissions import IsPatient,IsDoctor

from ml_app.submodels.user_details import  UserDetailModel
from ml_app.serializers.health_serializer import HealthRecordSerializer
from ml_app.serializers.user_serializers import UserSerializer, UserDetailSerializer

from django.http import Http404




class UserDetailView(generics.RetrieveUpdateAPIView,generics.CreateAPIView):
    permission_classes = [permissions.IsAuthenticatedOrReadOnly,
                         IsPatient]
    model = UserDetailModel
    serializer_class = UserDetailSerializer

    def get_queryset(self):
        user=self.request.user
        return UserDetailModel.objects.filter(user_id=user.pk)

    def get_object(self, pk):
        try:
            return UserDetailModel.objects.get(user_id=pk)
        except UserDetailModel.DoesNotExist:
            raise Http404

    def get(self, request, format=None):
        #get user detail based on his user_id field
        try:
            user_detail = self.get_object(request.user.pk)
        except Http404:
            return Response(status=status.HTTP_400_BAD_REQUEST)
        serializer = UserDetailSerializer(user_detail)
        return Response(serializer.data)


    def put(self, request, format=None):
        try:
            user = self.get_object(request.user.pk)
            serializer = UserDetailSerializer(user, data=request.data)
        except Http404:
            data=request.data
            UserDetailModel.objects.create(user_id=request.user.pk)
            user = self.get_object(request.user.pk)

            serializer=UserDetailSerializer(user,data=data)

        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)