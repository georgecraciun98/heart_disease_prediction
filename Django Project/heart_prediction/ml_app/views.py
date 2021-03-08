from django.contrib.auth.models import User
from rest_framework import generics
from rest_framework import permissions
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.reverse import reverse

from ml_app.models import HealthRecordModel
from ml_app.serializers.health_serializer import HealthRecordSerializer
from ml_app.serializers.user_serializers import UserSerializer
from .permissions import IsOwnerOrReadOnly
from rest_framework.permissions import AllowAny

class HealthRecordList(generics.ListCreateAPIView):
    model=HealthRecordModel
    permission_classes = [permissions.IsAuthenticatedOrReadOnly,
                          IsOwnerOrReadOnly]
    queryset = HealthRecordModel.objects.order_by('id').all()
    serializer_class = HealthRecordSerializer

    def perform_create(self, serializer):
        serializer.save(user_id=self.request.user.pk)

    def get_queryset(self):
        return self.queryset.filter(user_id=self.request.user.pk)



class HealthRecordDetail(generics.RetrieveUpdateDestroyAPIView):
    permission_classes = [permissions.IsAuthenticatedOrReadOnly,
                          IsOwnerOrReadOnly]
    queryset = HealthRecordModel.objects.all()
    serializer_class = HealthRecordSerializer



class UserList(generics.ListCreateAPIView):
    permission_classes = [AllowAny]
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
