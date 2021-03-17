from django.contrib.auth.models import User
from django.http import Http404
from rest_framework import generics, status
from rest_framework import permissions
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.reverse import reverse

from ml_app.models import HealthRecordModel, DoctorPatients, PredictedData
from ml_app.serializers.health_serializer import HealthRecordSerializer
from ml_app.serializers.patient_serializers import PatientSerializer
from ml_app.serializers.user_serializers import UserSerializer
from ml_app.permissions import IsOwnerOrReadOnly
from rest_framework.permissions import AllowAny

from ml_app.services.prediction_service import PredictionService
from ml_app.sub_permissions.group_permissions import IsDoctor
from ml_app.submodels.user_details import UserDetailModel
from datetime import date

def calculate_age(born):
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))
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

class PatientAddRecord(generics.GenericAPIView):
    model= HealthRecordModel
    permission_classes = [permissions.IsAuthenticatedOrReadOnly,
                          IsDoctor]

    serializer_class = HealthRecordSerializer

    def get_object(self, pk1,pk2):
        try:
            print('primary keys are',pk1,pk2)
            return DoctorPatients.objects.get(patient_id=pk1,doctor_id=pk2)
        except self.queryset.model.DoesNotExist:
            raise Http404

    """
        def perform_create(self, serializer):
        serializer.save(user_id=self.request.user.pk)
    """
    def get_user_details(self,pk):
        try:
            return UserDetailModel.objects.get(id=pk)
        except self.queryset.model.DoesNotExist:
            raise Http404
    def get_queryset(self,pk):
        return HealthRecordModel.objects.filter(doctor_patients_id=pk).order_by('created_data').first()
    def get(self, request,pk, format=None):
        #get user detail based on his user_id field

        doctor_patients = self.get_object(pk, request.user.pk)
        health_models = self.get_queryset(doctor_patients.id)
        try:
            user_model = self.get_queryset(pk)
            serializer = HealthRecordSerializer(data=health_models.__dict__)
            if serializer.is_valid():
                return Response(serializer.data)
            else:
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        except Http404:
            return Response(serializer.errors, status=status.HTTP_406_NOT_ACCEPTABLE)

    def post(self, request,pk, format=None):
        data={}
        try:
            doctor_patients = self.get_object(pk,request.user.pk)
            user_model=self.get_user_details(pk)
            age=calculate_age(user_model.birth_date)
            sex=user_model.sex
            print('record id',doctor_patients.id)
            data=request.data
            data['doctor_patients_id']=doctor_patients.id
            data['sex']=sex
            data['age']=age

            serializer = HealthRecordSerializer( data=data)
        except Http404:
            Response(serializer.errors, status=status.HTTP_406_NOT_ACCEPTABLE)

        if serializer.is_valid():
            serializer.save()
            print('all data is',data)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class PatientPrediction(generics.ListAPIView):
    model= PredictedData
    permission_classes = [permissions.IsAuthenticatedOrReadOnly,
                          IsDoctor]

    serializer_class = PatientSerializer

    def __init__(self):
        self.pred_service = PredictionService()
    #get last predicted data
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
    def get_doctor_patients(self, pk1,pk2):
        try:
            print('primary keys are',pk1,pk2)
            return DoctorPatients.objects.get(patient_id=pk1,doctor_id=pk2)
        except self.queryset.model.DoesNotExist:
            raise Http404
    def get_last_record(self, pk):
        try:
            return HealthRecordModel.objects.order_by('-created_data').filter(doctor_patients_id=pk).first()
        except self.queryset.model.DoesNotExist:
            raise Http404
    def post(self, request,pk, format=None):

        data={}
        try:
            doctor_patients = self.get_doctor_patients(pk,request.user.pk)
            #get last record
            last_record=self.get_last_record(doctor_patients.pk)
            model=self.pred_service.make_prediction(last_record.id)

            #We need to make the prediction

            return Response({"hi":"good"}, status=status.HTTP_201_CREATED)
        except Http404:
            Response({"hi":"bad"}, status=status.HTTP_406_NOT_ACCEPTABLE)


        return Response({"hi":"bad"}, status=status.HTTP_400_BAD_REQUEST)

