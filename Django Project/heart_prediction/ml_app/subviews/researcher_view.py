from django.contrib.auth.models import User
from django.core.files.storage import FileSystemStorage
from django.http import Http404
from rest_framework import generics, status
from rest_framework import permissions
from rest_framework.decorators import api_view
from rest_framework.parsers import FileUploadParser
from rest_framework.response import Response
from rest_framework.reverse import reverse

from ml_app.models import HealthRecordModel, DoctorPatients, PredictedData
from ml_app.serializers.health_record_serializer import HealthRecordSerializer
from ml_app.serializers.model_serializer import ModelSerializer, ModelMetricsSerializer, ModelFile
from ml_app.serializers.patient_serializers import PatientSerializer, PatientDetailSerializer
from ml_app.serializers.user_serializers import UserSerializer, UserDetailSerializer
from ml_app.permissions import IsOwnerOrReadOnly
from rest_framework.permissions import AllowAny

from ml_app.services.model_saving import ModelSaving
from ml_app.services.prediction_service import PredictionService
from ml_app.services.show_data_service import ShowData
from ml_app.sub_permissions.group_permissions import IsDoctor, IsResearcher, IsDoctororResearcher
from ml_app.submodels.model_configuration import ModelConfiguration
from ml_app.submodels.patient_model import Patient
from datetime import date
import io
from base64 import encodebytes
from PIL import Image


class ModelFileUpload(generics.ListCreateAPIView):
    model= ModelConfiguration
    permission_classes = [permissions.IsAuthenticatedOrReadOnly,
                          IsDoctororResearcher]

    serializer_class = ModelFile
    def get_response_image(self,pil_img):
        byte_arr = io.BytesIO()
        image2 = Image.fromarray(pil_img)
        image2.save(byte_arr, format='PNG')  # convert the PIL image to byte array
        encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii')  # encode as base64
        return encoded_img

    def list(self, request, *args, **kwargs):
        queryset = ModelConfiguration.objects.all()

        serializer = ModelFile(queryset, many=True)
        return Response(serializer.data)

    def post(self, request, format=None):

        data={}


        try:
            researcher_id=self.request.user.pk
            data=request.data
            print('data is',data)
            data['researcher_id']=researcher_id
            data1 = {}
            data1['researcher_id'] = researcher_id
            data1['source_file'] = request.FILES['source_file']
            serializer = self.get_serializer(data=data1)
            print(data1)
        except Http404:
            Response({"hi":"bad"}, status=status.HTTP_406_NOT_ACCEPTABLE)
        if serializer.is_valid():
            serializer.save()
            print('all data is',data)
            showData=ShowData()
            model = ModelConfiguration.objects.order_by("-created_date").first()

            img,img1,img2,img3=showData.load_data(model.id)
            img=self.get_response_image(img)
            img1=self.get_response_image(img1)
            img2 = self.get_response_image(img2)
            img3 = self.get_response_image(img3)

            return Response({"model_id":model.id,"scatter":img,"heatmap":img1,"correlation":img2,"pie":img3}, status=status.HTTP_201_CREATED)

        return Response({"hi":"bad"}, status=status.HTTP_400_BAD_REQUEST)

class PredictionModelsTraining(generics.ListCreateAPIView):
    model = ModelConfiguration
    permission_classes = [permissions.IsAuthenticatedOrReadOnly,
                          IsDoctororResearcher]

    serializer_class = ModelMetricsSerializer

    def list(self, request, *args, **kwargs):
        queryset = ModelConfiguration.objects.all()

        serializer = ModelMetricsSerializer(queryset, many=True)
        return Response(serializer.data)

    def post(self, request, format=None):

        data = {}
        try:
            researcher_id = self.request.user.pk
            data = request.data
            data['researcher_id'] = researcher_id
            if data['alg_name'] == 'Support Vector Machine':
                serializer = ModelSerializer(data=data)
                pass
            data1 = {}
            data1['researcher_id'] = researcher_id
            data1['alg_name'] = data['alg_name']
            if data['alg_name'] == 'Random Forest Classifier':
                data1['n_estimators'] = data['n_estimators']
                data1['max_features'] = data['max_features']
                data1['max_depth'] = data['max_depth']
                data1['min_samples_split'] = data['min_samples_split']
                data1['min_samples_leaf'] = data['min_samples_leaf']
                data1['bootstrap'] = data['bootstrap']
                serializer = ModelSerializer(data=data1)
            if data['alg_name'] == "XGB Classifier":
                data1['n_estimators'] = data['n_estimators']
                data1['max_depth'] = data['max_depth']
                data1['booster'] = data['booster']
                data1['base_score'] = data['base_score']
                data1['learning_rate'] = data['learning_rate']
                data1['min_child_weight'] = data['min_child_weight']
                serializer = ModelSerializer(data=data1)
                print('data xgb classifier is', data1)
            if data['alg_name'] == "Support Vector Machine":
                data1['c'] = data['c']
                data1['gamma'] = data['gamma']
                data1['kernel'] = data['kernel']
                serializer = ModelSerializer(data=data1)
            if data['alg_name'] == "Binary Classifier":
                data1['c'] = data['c']
                data1['solver'] = data['solver']
                data1['penalty'] = data['penalty']
                serializer = ModelSerializer(data=data1)
                print(data1)
        except Http404:
            Response({"hi": "bad"}, status=status.HTTP_406_NOT_ACCEPTABLE)
        if serializer.is_valid():
            serializer.save()
            print('all data is', data)
            modelSaving = ModelSaving()
            accuracy_score1,precision_score1,f1_score1,roc_auc_score1=modelSaving.train_model(name=data['alg_name'], data=data1, researcher_id=researcher_id)
            data2=serializer.data
            data2['precision']=precision_score1
            data2['accuracy'] = accuracy_score1
            data2['f1_score'] = f1_score1
            data2['roc_auc_score'] = roc_auc_score1
            return Response(data2, status=status.HTTP_201_CREATED)

        return Response({"hi": "bad"}, status=status.HTTP_400_BAD_REQUEST)

class PredictionModelsSaving(generics.ListCreateAPIView):
    model = ModelConfiguration
    permission_classes = [permissions.IsAuthenticatedOrReadOnly,
                          IsDoctororResearcher]

    serializer_class = ModelSerializer

    def list(self, request, *args, **kwargs):
        queryset = ModelConfiguration.objects.filter(active=True)

        serializer = ModelMetricsSerializer(queryset, many=True)
        return Response(serializer.data)

    def post(self, request, format=None):

        data = {}
        try:
            researcher_id = self.request.user.pk
            data = request.data
            data['researcher_id'] = researcher_id
            if data['alg_name'] == 'Support Vector Machine':
                serializer = ModelSerializer(data=data)
                pass
            data1 = {}
            data1['researcher_id'] = researcher_id
            model_id=data['model_id']
            model=ModelConfiguration.objects.get(id=model_id)
            model.active=True
            model.alg_description=model.alg_name+" Precision " + str(model.precision)+" Accuracy " + str(model.accuracy)
            model.save()
        except Http404:
            Response({"hi": "bad"}, status=status.HTTP_406_NOT_ACCEPTABLE)

        return Response({"response":"model saved"}, status=status.HTTP_201_CREATED)

