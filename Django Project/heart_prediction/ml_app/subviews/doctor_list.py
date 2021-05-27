from django.http import Http404
from rest_framework import generics, status
from django.contrib.auth.models import User

from ml_app.serializers.appointment_serializer import AppointmentSerializer
from ml_app.serializers.patient_serializers import PatientUserSerializer
from ml_app.sub_permissions.group_permissions import IsPatient,IsDoctor
from rest_framework.response import Response
from rest_framework import permissions

from ml_app.submodels.appointments import Appointments


class DoctorList(generics.ListCreateAPIView):
    model= User
    permission_classes = [permissions.IsAuthenticatedOrReadOnly,
                          IsPatient]

    serializer_class = PatientUserSerializer

    def get_queryset(self):
        patient_id=self.request.user.user.id
        queryset = User.objects.order_by('id').filter(groups__name='doctor')
        return queryset
    #
    # def perform_create(self, serializer):
    #     serializer.save(patient_id=self.request.user.pk)

    def list(self, request, *args, **kwargs):
        queryset = self.filter_queryset(self.get_queryset())

        serializer = self.get_serializer(queryset, many=True)

        return Response(serializer.data,status=status.HTTP_200_OK)

    # def post(self, request, pk, format=None):
    #     data = {}
    #     try:
    #         # patient_id=self.request.user.user.id
    #         data = request.data
    #         serializer = self.get_serializer(data=data)
    #     except Http404:
    #         Response(serializer.errors, status=status.HTTP_406_NOT_ACCEPTABLE)
    #
    #     if serializer.is_valid():
    #         serializer.save()
    #         print('all data is', data)
    #         return Response(serializer.data, status=status.HTTP_201_CREATED)
    #     return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class AppointmentGet(generics.ListCreateAPIView):
    model = Appointments
    permission_classes = [permissions.IsAuthenticatedOrReadOnly,
                          IsPatient]

    serializer_class = AppointmentSerializer

    def get_queryset(self,pk):
        queryset = Appointments.objects.filter(doctor_id=pk)
        return queryset


    # def perform_create(self, serializer):
    #     serializer.save(patient_id=self.request.user.pk)

    def list(self, request,pk, *args, **kwargs):
        queryset = self.filter_queryset(self.get_queryset(pk))

        serializer = self.get_serializer(queryset, many=True)

        return Response(serializer.data, status=status.HTTP_200_OK)

    def post(self, request, pk, format=None):
        data = {}
        try:
            patient = self.request.user.pk

            data = request.data
            data['doctor_id']=pk
            data['patient_id']=patient
            print('patient is',patient)
            serializer = self.get_serializer(data=data)
        except Http404:
            Response(serializer.errors, status=status.HTTP_406_NOT_ACCEPTABLE)

        if serializer.is_valid():
            serializer.save()
            print('all data is', data)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    ""
class AppointmentHours(generics.ListCreateAPIView):
    model = Appointments
    permission_classes = [permissions.IsAuthenticatedOrReadOnly,
                          IsPatient]

    serializer_class = AppointmentSerializer

    def get_queryset(self,pk):
        queryset = Appointments.objects.filter(doctor_id=pk)
        return queryset


    # def perform_create(self, serializer):
    #     serializer.save(patient_id=self.request.user.pk)

    def list(self, request,pk, *args, **kwargs):
        queryset = self.filter_queryset(self.get_queryset(pk))

        serializer = self.get_serializer(queryset, many=True)

        return Response(serializer.data, status=status.HTTP_200_OK)

    ""
class AppointmentPatient(generics.ListCreateAPIView):
    model = Appointments
    permission_classes = [permissions.IsAuthenticatedOrReadOnly,
                          IsPatient]

    serializer_class = AppointmentSerializer

    def get_queryset(self,pk):
        queryset = Appointments.objects.filter(patient_id=pk)
        return queryset

    def list(self, request,pk, *args, **kwargs):
        queryset = self.filter_queryset(self.get_queryset(pk))
        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

