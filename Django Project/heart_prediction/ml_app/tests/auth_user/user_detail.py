from django.contrib.auth.models import User,Group
from django.test import TestCase, Client
from rest_framework import status

from ml_app.serializers.user_serializers import UserSerializer, UserDetailSerializer
from rest_framework.test import force_authenticate, APIClient

from ml_app.submodels.patient_model import Patient
from ml_app.subviews.user_view import UserDetailView
from rest_framework.test import APIRequestFactory
from rest_framework.authtoken.models import Token

#Correct Authentication for a User
class GetUserDetails(TestCase):
    """ Test module for GET all puppies API """

    def setUp(self):
        self.alexandru=User.objects.create(
            username='alexandru32', password='alexandru32',email='ionel@yahoo.com')

        self.group=Group.objects.create(name='patient')
        self.alexandru.groups.add(self.group)
        Patient.objects.create(sex=1,birth_date='1997-10-19',user_id=self.alexandru.pk)
        self.view=UserDetailView.as_view()

    def test_get_user_details(self):
        factory = APIRequestFactory()

        client = APIClient()
        alexandru=User.objects.get(username='alexandru32')
        alexandru_details=Patient.objects.get(user_id=alexandru.pk)

        token = Token.objects.get(user__username='alexandru32')
        client.credentials(HTTP_AUTHORIZATION='Token ' + token.key)

        response=client.get('http://127.0.0.1:8000/api/patients/detail/')

        serializer=UserDetailSerializer(alexandru_details)

        self.assertEqual(response.data,serializer.data)
        self.assertEqual(response.status_code,status.HTTP_200_OK)