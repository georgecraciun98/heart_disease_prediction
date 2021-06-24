from django.contrib.auth.models import User, Group
from django.test import TestCase, Client
from rest_framework import status

from ml_app.serializers.user_serializers import UserSerializer
from rest_framework.test import APIRequestFactory, force_authenticate

from ml_app.subviews.doctor_list import DoctorList
from ml_app.views import UserList


class GetAllDoctor(TestCase):
    """ Test module for GET all puppies API """

    def setUp(self):
        doctor123=User.objects.create(
            username='doctor123', password='alexandru32',email='ionel@yahoo.com')
        marian=User.objects.create(
            username='marian238', password='marian238',email='marian@yahoo.com')
        User.objects.create(
            username='emanuel43', password='alexandru32',email='alexandru@yahoo.com')
        group_doctor=Group.objects.create(name='doctor')
        group_doctor.user_set.add(doctor123)
        group_patient = Group.objects.create(name='patient')
        group_patient.user_set.add(marian)

        self.view=DoctorList.as_view()

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