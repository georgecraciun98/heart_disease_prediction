from django.http import Http404
from rest_framework import generics, status
from django.contrib.auth.models import User

from ml_app.serializers.appointment_serializer import AppointmentSerializer,\
    AppointmentDetailedSerializer,AppointmentDoctors
from ml_app.serializers.patient_serializers import PatientUserSerializer
from ml_app.sub_permissions.group_permissions import IsPatient,IsDoctor
from rest_framework.response import Response
from rest_framework import permissions

from ml_app.submodels.appointments import Appointments
from ml_app.submodels.patient_model import Patient


class DoctorList(generics.ListCreateAPIView):
    model= User
    permission_classes = [permissions.IsAuthenticatedOrReadOnly,
                          IsPatient]

    serializer_class = PatientUserSerializer

    def get_queryset(self):
        # patient_id=self.request.user.user.id
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

class AppointmentDoctor(generics.ListCreateAPIView):
    model=Appointments
    permission_classes = [permissions.IsAuthenticatedOrReadOnly,IsDoctor]

    serializer_class = AppointmentDetailedSerializer

    def get_queryset(self, pk):

        queryset = Appointments.objects.filter(doctor_id=pk)
        return queryset

    def list(self, request, pk, *args, **kwargs):
        queryset = self.filter_queryset(self.get_queryset(pk))

        serializer = self.get_serializer(queryset, many=True)

        return Response(serializer.data, status=status.HTTP_200_OK)



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
        print('all data initial', data)
        try:
            id=self.request.user.pk
            patient=Patient.objects.get(user_id=id)

            data = request.data
            print('all data is', data)
            data['doctor_id']=pk
            data['patient_id']=patient.id
            print('patient is',patient)
            serializer = self.get_serializer(data=data)
        except Http404:
            Response(serializer.errors, status=status.HTTP_406_NOT_ACCEPTABLE)

        if serializer.is_valid():
            serializer.save()
            print('all data is', data)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class AppointmentByUser(generics.ListCreateAPIView):
        model = Appointments
        permission_classes = [permissions.IsAuthenticatedOrReadOnly,
                              IsPatient]

        serializer_class = AppointmentDoctors

        def get_queryset(self, pk):
            queryset = Appointments.objects.filter(patient_id=pk)
            print('list is',queryset,pk)
            return queryset

        def list(self, request, pk, *args, **kwargs):
            id = self.request.user.pk
            patient = Patient.objects.get(user_id=id)
            queryset = self.filter_queryset(self.get_queryset(pk=patient.id))
            serializer = self.get_serializer(queryset, many=True)
            return Response(serializer.data, status=status.HTTP_200_OK)

        def post(self, request, pk, format=None):
            data = {}
            try:
                id = self.request.user.pk
                patient = Patient.objects.get(user_id=id)

                data = request.data
                data['doctor_id'] = pk
                data['patient_id'] = patient
                serializer = self.get_serializer(data=data)
            except Http404:
                Response(serializer.errors, status=status.HTTP_406_NOT_ACCEPTABLE)

            if serializer.is_valid():
                serializer.save()
                return Response(serializer.data, status=status.HTTP_201_CREATED)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

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

