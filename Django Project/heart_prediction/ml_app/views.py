from django.contrib.auth.models import User
from django.http import Http404
from rest_framework import generics, status
from rest_framework import permissions
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.reverse import reverse

from ml_app.models import HealthRecordModel
from ml_app.serializers.health_record_serializer import HealthRecordSerializer
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
    def get(self, request,pk, format=None):
        #get user detail based on his user_id field

        health_models = HealthRecordModel.objects.get(pk=pk)
        try:
            serializer = HealthRecordSerializer(data=health_models.__dict__)

            if serializer.is_valid():
                data1 = serializer.data
                print('data is',data1)

                data = {}

                data['trestbps'] = data1['trestbps']
                data['chol'] = data1['chol']
                data['thalach'] = data1['thalach']
                data['oldpeak'] = data1['oldpeak']
                if data1['sex_0'] == 1:
                    data['sex'] = 0
                elif data1['sex_1'] == 1 :
                    data['sex'] = 1

                if data1['cp_0'] == 1:
                    data['cp'] = 0
                elif data1['cp_1'] == 1:
                    data['cp'] = 1
                elif data1['cp_2'] == 1:
                    data['cp'] = 2
                elif data1['cp_3'] == 1:
                    data['cp'] = 3

                if data1['fbs_0'] == 1:
                    data['fbs'] = 0
                elif data1['fbs_1'] == 1:
                    data['fbs'] = 1

                if data1['restecg_0'] == 1:
                    data['restecg'] = 0
                elif data1['restecg_1'] == 1:
                    data['restecg'] = 1
                elif data1['restecg_2'] == 1:
                    data['restecg'] = 2

                elif data1['exang_0'] == 1:
                    data['exang'] = 0
                elif data1['exang_1'] == 1:
                    data['exang'] = 1

                if data1['slope_0'] == 1:
                    data['slope'] = 0
                elif data1['slope_1'] == 1:
                    data['slope'] = 1
                elif data1['slope_2'] == 1:
                    data['slope'] = 2

                if data1['ca_0'] == 1:
                    data['ca'] = 0
                elif data1['ca_1'] == 1:
                    data['ca'] = 1
                elif data1['ca_2'] == 1:
                    data['ca'] = 2
                elif data1['ca_3'] == 1:
                    data['ca'] = 3
                elif data1['ca_4'] == 1:
                    data['ca'] = 4

                if data1['thal_0'] == 1:
                    data['thal'] = 1
                elif data1['thal_1'] == 1:
                    data['thal'] = 3
                elif data1['thal_2'] == 1:
                    data['thal'] = 6
                elif data1['thal_3'] == 1:
                    data['thal'] = 7
                return Response(data)
            else:
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        except Http404:
            return Response(serializer.errors, status=status.HTTP_406_NOT_ACCEPTABLE)



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
