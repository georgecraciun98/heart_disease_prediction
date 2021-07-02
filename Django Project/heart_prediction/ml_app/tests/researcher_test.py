from django.contrib.auth.models import User, Group
from django.test import TestCase, Client
from rest_framework import status
from rest_framework.test import APIRequestFactory, force_authenticate
from ml_app.submodels.model_configuration import ModelConfiguration
from ml_app.subviews.researcher_view import PredictionModelsTraining, PredictionModelsSaving

class GetAllResearcher(TestCase):


    def setUp(self):
        self.researcher=User.objects.create(
            username='popmarian',is_active=True, password='*@jklas*lM32',email='ionel@yahoo.com')
        marian=User.objects.create(
            username='marian238',is_active=True, password='marian238',email='marian@yahoo.com')
        User.objects.create(
            username='emanuel43',is_active=True, password='alexandru32',email='alexandru@yahoo.com')

        group_doctor=Group.objects.create(name='doctor')
        group_researcher = Group.objects.create(name='researcher')
        group_researcher.user_set.add(self.researcher)
        group_patient = Group.objects.create(name='patient')
        group_patient.user_set.add(marian)

        ModelConfiguration.objects.create(researcher_id=self.researcher.pk,alg_name='Support Vector Machine'
                                          ,c=0.2,gamma=0.001,kernel="linear")

        self.user = self.researcher



    def create_model(self):
        factory = APIRequestFactory()
        data1 = {}
        data1['alg_name'] = 'Support Vector Machine'
        data1['researcher_id'] = self.researcher.pk
        data1['c'] = 0.1
        data1['gamma'] = 0.001
        data1['kernel'] = "linear"
        request=factory.post('http://127.0.0.1:8000/api/models/train', data1)
        force_authenticate(request, user=User.objects.get(username='popmarian'))
        view=PredictionModelsTraining.as_view()
        response = view(request)
        model = ModelConfiguration.objects.order_by("-created_date").first()
        return model.pk

    #Test for saving trained models
    def test_train_status(self):
        factory = APIRequestFactory()
        data1={}
        data1['alg_name']='Support Vector Machine'
        data1['researcher_id'] = self.researcher.pk
        data1['c'] = 0.1
        data1['gamma'] = 0.001
        data1['kernel'] = "linear"
        request = factory.post('http://127.0.0.1:8000/api/models/train', data1)
        request.user=self.researcher
        force_authenticate(request, user=User.objects.get(username='popmarian'))
        view = PredictionModelsTraining.as_view()
        response = view(request)
        if response.status_code>=200 and response.status_code<=204:
            self.assertEqual(response.status_code,status.HTTP_201_CREATED)
        else:
            testValue = True
            self.assertFalse(testValue,"Test value is not false.")

    # Test for saving trained models
    def test_train_model(self):
        factory = APIRequestFactory()
        data1 = {}
        data1['alg_name'] = 'Support Vector Machine'
        data1['researcher_id'] = self.researcher.pk
        data1['c'] = 0.1
        data1['gamma'] = 0.001
        data1['kernel'] = "linear"
        request = factory.post('http://127.0.0.1:8000/api/models/train', data1)
        request.user = self.researcher
        force_authenticate(request, user=User.objects.get(username='popmarian'))
        view = PredictionModelsTraining.as_view()
        response = view(request)
        if response.status_code >= 200 and response.status_code <= 204:
            self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        else:
            testValue = True
            self.assertFalse(testValue, "Test value is not false.")
    #Test for activating models
    def test_save_models(self):
        factory = APIRequestFactory()

        model_id=self.create_model()
        request = factory.post('http://127.0.0.1:8000/api/models/save', {'model_id':model_id})
        request.user=self.researcher
        force_authenticate(request, user=User.objects.get(username='popmarian'))
        view = PredictionModelsSaving.as_view()
        response = view(request)
        if response.status_code >= 200 and response.status_code <= 204:
            self.assertEqual(response.status_code, status.HTTP_201_CREATED)

