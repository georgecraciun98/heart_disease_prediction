from django.http import Http404
from rest_framework import generics, status
from rest_framework import permissions
from rest_framework.response import Response

from ml_app.models import HealthRecordModel, DoctorPatients, PredictedData
from ml_app.serializers.health_record_serializer import HealthRecordSerializer
from ml_app.serializers.patient_serializers import PatientSerializer, PatientDetailSerializer
from ml_app.services.prediction_service import PredictionService
from ml_app.sub_permissions.group_permissions import IsDoctor
from ml_app.submodels.patient_model import Patient
from datetime import date

def calculate_age(born):
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))
class PatientList(generics.ListAPIView):
    model= Patient
    permission_classes = [permissions.IsAuthenticatedOrReadOnly,
                          IsDoctor]

    serializer_class = PatientSerializer

    def get_queryset(self):
        doctor_id=self.request.user.pk
        queryset = Patient.objects.order_by('id').filter(patients__doctor_id=doctor_id)
        return queryset
    def perform_create(self, serializer):
        serializer.save(user_id=self.request.user.pk)

    def list(self, request, *args, **kwargs):
        queryset = self.filter_queryset(self.get_queryset())

        serializer = self.get_serializer(queryset, many=True)

        return Response(serializer.data)



class RecordList(generics.ListCreateAPIView):
    model = HealthRecordModel
    permission_classes = [permissions.IsAuthenticatedOrReadOnly,
                          IsDoctor]
    serializer_class = HealthRecordSerializer

    def get_object(self, pk1, pk2):
        try:
            print('primary keys are', pk1, pk2)
            return DoctorPatients.objects.get(patient_id=pk1, doctor_id=pk2)
        except self.queryset.model.DoesNotExist:
            raise Http404

    """
        def perform_create(self, serializer):
        serializer.save(user_id=self.request.user.pk)
    """

    def get_user_details(self, pk):
        try:
            return Patient.objects.get(id=pk)
        except self.queryset.model.DoesNotExist:
            raise Http404

    def get_queryset(self, pk):
        return HealthRecordModel.objects.filter(doctor_patients_id=pk).order_by('-created_data')

    def get(self, request, pk, format=None):
        doctor_patients = self.get_object(pk, request.user.pk)
        health_models = self.get_queryset(doctor_patients.id)
        serializer=self.get_serializer(health_models,many=True)


        return Response(serializer.data)
    def list(self, request, pk, format=None):
        doctor_patients = self.get_object(pk, request.user.pk)
        health_models = self.get_queryset(doctor_patients.id)
        serializer=self.get_serializer(health_models,many=True)
        return Response(serializer.data)

class RecordDetail(generics.GenericAPIView):
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
            return Patient.objects.get(id=pk)
        except self.queryset.model.DoesNotExist:
            raise Http404
    def get_queryset(self,pk):
        return HealthRecordModel.objects.filter(doctor_patients_id=pk).order_by('-created_data').first()
    def get(self, request,pk, format=None):
        #get user detail based on his user_id field

        doctor_patients = self.get_object(pk, request.user.pk)
        health_models = self.get_queryset(doctor_patients.id)
        try:
            serializer = HealthRecordSerializer(data=health_models.__dict__)


            if serializer.is_valid():
                data1 = serializer.data

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

                if data1['exang_0'] == 1:
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

    def post(self, request,pk, format=None):
        data={}
        try:
            doctor_patients = self.get_object(pk,request.user.pk)
            user_model=self.get_user_details(pk)
            age=calculate_age(user_model.birth_date)
            sex=user_model.sex
            print('record id',doctor_patients.id)

            data1=request.data
            data1['sex'] = sex

            data2={}
            data2['age']=age
            data2['trestbps'] = data1['trestbps']
            data2['chol'] = data1['chol']
            data2['thalach'] = data1['thalach']
            data2['oldpeak'] = data1['oldpeak']

            if data1['sex'] == 0:
                data2['sex_0'] = 1
            else:
                data2['sex_1'] = 1

            if data1['cp'] == '0':
                data2['cp_0'] = 1
            elif data1['cp'] == '1':
                data2['cp_1'] = 1
            elif data1['cp'] == '2':
                data2['cp_2'] = 1
            elif data1['cp'] == '3':
                data2['cp_3'] = 1

            if data1['fbs'] == '0':
                data2['fbs_0'] = 1
            elif data1['fbs'] == '1':
                data2['fbs_1'] = 1

            if data1['restecg'] == '0':
                data2['restecg_0'] = 1
            elif data1['restecg'] == '1':
                data2['restecg_1'] = 1
            elif data1['restecg'] == '2':
                data2['restecg_2'] = 1

            if data1['exang'] == '0':
                data2['exang_0'] = 1
            elif data1['exang'] == '1':
                data2['exang_1'] = 1

            if data1['slope'] == '0':
                data2['slope_0'] = 1
            elif data1['slope'] == '1':
                data2['slope_1'] = 1
            elif data1['slope'] == '2':
                data2['slope_2'] = 1

            if data1['ca'] == '0':
                data2['ca_0'] = 1
            elif data1['ca'] == '1':
                data2['ca_1'] = 1
            elif data1['ca'] == '2':
                data2['ca_2'] = 1
            elif data1['ca'] == '3':
                data2['ca_3'] = 1
            elif data1['ca'] == '4':
                data2['ca_4'] = 1

            if data1['thal'] == '1':
                data2['thal_0'] = 1
            elif data1['thal'] == '3':
                data2['thal_1'] = 1
            elif data1['thal'] == '6':
                data2['thal_2'] = 1
            elif data1['thal'] == '7':
                data2['thal_3'] = 1
            data2['doctor_patients_id']=doctor_patients.id
            serializer = HealthRecordSerializer( data=data2)
        except Http404:
            Response(serializer.errors, status=status.HTTP_406_NOT_ACCEPTABLE)

        if serializer.is_valid():
            serializer.save()
            print('all data is',data)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
"""
    Classed used for making patients predictions, as an input this class
    will take just the id of the patient and the record id . 
"""
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
        queryset = Patient.objects.order_by('id').filter(patients__doctor_id=doctor_id)
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
            data = request.data
            print('requested data is',data)
            doctor_patients = self.get_doctor_patients(pk,request.user.pk)
            #get last record
            last_record=self.get_last_record(doctor_patients.pk)
            returned_value=self.pred_service.make_prediction(last_record.id,model_id=data['model'])
            if returned_value >= 0.5:
                returned_value=1
            else:
                returned_value=0
            print('type of returned value is',type(returned_value))
            PredictedData.objects.create(model_id=data['model'],target=returned_value,record_id=last_record.id)
            #We need to make the prediction
            return Response({'target':returned_value}, status=status.HTTP_201_CREATED)
        except Http404:
            Response({"Your object was not found"}, status=status.HTTP_406_NOT_ACCEPTABLE)


        return Response({"Your object was not found"}, status=status.HTTP_400_BAD_REQUEST)

class PatientDetail(generics.RetrieveUpdateAPIView,generics.CreateAPIView):
    permission_classes = [permissions.IsAuthenticatedOrReadOnly,
                         IsDoctor]
    model = Patient
    serializer_class = PatientDetailSerializer


    def get_object(self, pk):
        try:
            return Patient.objects.filter(id=pk)
        except self.queryset.model.DoesNotExist:
            raise Http404
    def get_queryset(self,pk):
        queryset = Patient.objects.order_by('id').filter(id=pk).first()
        return queryset
    def get(self, request,pk, format=None):
        #get user detail based on his user_id field
        try:

            queryset = self.filter_queryset(self.get_queryset(pk))

            serializer = self.get_serializer(queryset)

            data=serializer.data
            data['age']=calculate_age(queryset.birth_date)
            return Response(data, status=status.HTTP_200_OK)
        except Http404:
            return Response({"hi":"bad"}, status=status.HTTP_200_OK)

