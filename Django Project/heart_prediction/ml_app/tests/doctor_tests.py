from django.contrib.auth.models import User, Group
from django.test import TestCase, Client
from rest_framework import status

from ml_app.serializers.user_serializers import UserSerializer
from rest_framework.test import APIRequestFactory, force_authenticate

from ml_app.submodels.doctor_patients import DoctorPatients
from ml_app.submodels.health_record import HealthRecordModel
from ml_app.submodels.model_configuration import ModelConfiguration
from ml_app.submodels.patient_model import Patient
from ml_app.subviews.doctor_list import DoctorList
from ml_app.subviews.doctor_view import PatientPrediction
from ml_app.subviews.researcher_view import PredictionModelsTraining


class DoctorTests(TestCase):

    def setUp(self):
        self.researcher = User.objects.create(
            username='researcher', is_active=True, password='*@jklas*lM32', email='researcher@yahoo.com')
        self.doctor123=User.objects.create(
            username='doctor123',is_active=True, password='alexandru32',email='ionel@yahoo.com')
        self.marian=User.objects.create(
            username='marian238',is_active=True, password='marian238',email='marian@yahoo.com')
        self.emanuel=User.objects.create(
            username='emanuel43',is_active=True, password='alexandru32',email='alexandru@yahoo.com')
        group_doctor=Group.objects.create(name='doctor')
        group_doctor.user_set.add(self.doctor123)
        group_researcher = Group.objects.create(name='researcher')
        group_researcher.user_set.add(self.researcher)
        group_patient = Group.objects.create(name='patient')
        group_patient.user_set.add(self.marian)
        group_patient.user_set.add(self.emanuel)
        self.patient=Patient.objects.create(birth_date='1998-05-05',description='My Health is pretty good',
                               sex=0,user_id=self.marian.pk)
        doctor_patients=DoctorPatients.objects.create(doctor_id=self.doctor123.pk,patient_id=self.patient.pk)
        self.health_record=HealthRecordModel.objects.create(trestbps=145, chol=233, thalach=150, oldpeak=2.3,
                                         age=22, sex_0=0,
                                         sex_1=1, cp_0=0, cp_1=0, cp_2=0, cp_3=1, fbs_0=0, fbs_1=1, restecg_0=0,
                                         restecg_1=1, restecg_2=0, exang_0=1, exang_1=0, slope_0=1, slope_1=0,
                                         slope_2=0, ca_0=0, ca_1=1, ca_2=0, ca_3=0, ca_4=0, thal_0=1, thal_1=0,
                                         thal_2=0, thal_3=0,doctor_patients_id=doctor_patients.pk
                                         )
        self.view=DoctorList.as_view()

    def train_models(self):
        factory = APIRequestFactory()
        data1 = {}
        data1['alg_name'] = 'Support Vector Machine'
        data1['researcher_id'] = self.researcher.pk
        data1['c'] = 0.1
        data1['gamma'] = 0.001
        data1['kernel'] = "linear"
        request = factory.post('http://127.0.0.1:8000/api/models/train', data1)
        request.user = self.researcher
        force_authenticate(request, user=User.objects.get(username='researcher'))
        view = PredictionModelsTraining.as_view()
        response = view(request)
        if response.status_code>=200 and response.status_code<=204:
            return True
        return False
    def test_predict_health(self):
        factory = APIRequestFactory()
        pk=self.patient.pk
        self.train_models()
        ml_model = ModelConfiguration.objects.order_by("-created_date").first()
        request = factory.post("http://127.0.0.1:8000/api/patients/{}/predict/".format(pk),
                               data={'model': ml_model.pk})
        force_authenticate(request, user=User.objects.get(username='doctor123'))
        view = PatientPrediction.as_view()
        response = view(request,pk=pk)

        if response.status_code >= 200 and response.status_code <= 204:
            self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        else:
            testValue = True
            self.assertFalse(testValue, "Test value is not false.")

    def test_get_all_users(self):
        client = Client()
        factory = APIRequestFactory()

        #client.login(username='alexandru232', password='superuser123')
        request = factory.get('http://127.0.0.1:8000/api/doctors/')
        force_authenticate(request, user=User.objects.get(username='marian238'))

        response = self.view(request)
        users= User.objects.order_by('id').filter(groups__name='doctor')
        serializer=UserSerializer(data=users, many=True)
        print(response.data)
        print('users are',users)
        if serializer.is_valid():
            self.assertEqual(response.data,serializer.data)
        self.assertEqual(response.status_code,status.HTTP_200_OK)

    def test_get_patient_details(self):
        client = Client()
        factory = APIRequestFactory()
        id=self.patient.pk
        #client.login(username='alexandru232', password='superuser123')
        request = factory.get('http://127.0.0.1:8000/api/patients/'+id)
        force_authenticate(request, user=User.objects.get(username='doctor123'))

        response = self.view(request)

        users= User.objects.order_by('id').filter(groups__name='doctor')
        serializer=UserSerializer(data=users, many=True)
        print(response.data)
        print('users are',users)
        if serializer.is_valid():
            self.assertEqual(response.data,serializer.data)
        self.assertEqual(response.status_code,status.HTTP_200_OK)


