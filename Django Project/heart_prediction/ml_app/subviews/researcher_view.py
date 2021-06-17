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
    # parser_classes = [FileUploadParser]


    #get last predicted data
    # def get_queryset(self):
    #     doctor_id=self.request.user.pk
    #     queryset = Patient.objects.order_by('id').filter(patients__doctor_id=doctor_id)
    #     return queryset
    # def perform_create(self, serializer):
    #     serializer.save(user_id=self.request.user.pk)
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

            img,img1,img2=showData.load_data(model.id)
            img=self.get_response_image(img)
            img1=self.get_response_image(img1)
            img2 = self.get_response_image(img2)
            return Response({"scatter":img,"heatmap":img1,"correlation":img2}, status=status.HTTP_201_CREATED)

        return Response({"hi":"bad"}, status=status.HTTP_400_BAD_REQUEST)