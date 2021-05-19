from django.contrib.auth.models import User
from django.http import Http404
from rest_framework import generics, status
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
        serializer.save()

    def get_queryset(self):
        return self.queryset.filter(user_id=self.request.user.pk)



class HealthRecordDetail(generics.RetrieveAPIView):
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


class UserDetail(generics.GenericAPIView):
    queryset = User.objects.all()
    serializer_class = UserSerializer

    def get_object(self, pk):
        try:
            return User.objects.get(id=pk)
        except self.queryset.model.DoesNotExist:
            raise Http404

    def get(self, request, format=None):
        # get user detail based on his user_id field
        user = self.get_object(request.user.pk)

        serializer = UserSerializer(user)

        return Response(serializer.data)



@api_view(['GET'])
def api_root(request, format=None):
    return Response({
        'users': reverse('user-list', request=request, format=format),
        'records': reverse('record-list', request=request, format=format)
    })
