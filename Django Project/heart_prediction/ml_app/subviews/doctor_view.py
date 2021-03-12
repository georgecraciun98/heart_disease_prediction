from django.contrib.auth.models import User
from django.http import Http404
from rest_framework import generics, status
from rest_framework import permissions
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.reverse import reverse

from ml_app.models import HealthRecordModel, DoctorPatients
from ml_app.serializers.health_serializer import HealthRecordSerializer
from ml_app.serializers.patient_serializers import PatientSerializer
from ml_app.serializers.user_serializers import UserSerializer
from ml_app.permissions import IsOwnerOrReadOnly
from rest_framework.permissions import AllowAny

from ml_app.sub_permissions.group_permissions import IsDoctor
from ml_app.submodels.user_details import UserDetailModel


class PatientList(generics.ListAPIView):
    model= UserDetailModel
    permission_classes = [permissions.IsAuthenticatedOrReadOnly,
                          IsDoctor]

    serializer_class = PatientSerializer

    def get_queryset(self):
        doctor_id=self.request.user.pk
        queryset = UserDetailModel.objects.order_by('id').filter(patients__doctor_id=doctor_id)
        return queryset
    def perform_create(self, serializer):
        serializer.save(user_id=self.request.user.pk)

    def list(self, request, *args, **kwargs):
        queryset = self.filter_queryset(self.get_queryset())

        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)





