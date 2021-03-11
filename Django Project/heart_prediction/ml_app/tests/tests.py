from django.contrib.auth.models import User
from django.test import TestCase, Client
from rest_framework import status

from ml_app.serializers.user_serializers import UserSerializer
from rest_framework.test import APIRequestFactory

from ml_app.views import UserList


class GetAllUserTest(TestCase):
    """ Test module for GET all puppies API """

    def setUp(self):
        User.objects.create(
            username='alexandru32', password='alexandru32',email='ionel@yahoo.com')
        User.objects.create(
            username='marian238', password='marian238',email='marian@yahoo.com')
        User.objects.create(
            username='emanuel43', password='alexandru32',email='alexandru@yahoo.com')
        self.view=UserList.as_view()

    def test_get_all_users(self):
        client = Client()
        factory = APIRequestFactory()

        #client.login(username='alexandru232', password='superuser123')
        request = factory.get('http://127.0.0.1:8000/api/users/')
        response = self.view(request)

        users=User.objects.order_by('id').all()
        serializer=UserSerializer(users, many=True)

        self.assertEqual(response.data,serializer.data)
        self.assertEqual(response.status_code,status.HTTP_200_OK)





class ShortPasswordTestCase(TestCase):
    """ Test module for GET all puppies API """

    def setUp(self):
        self.alexandru = User.objects.create(
            username='alexandru32', password='a',email='ionel@yahoo.com')
        self.marian = User.objects.create(
            username='marian238', password='ma',email='marian@yahoo.com')
        self.emanuel = User.objects.create(
            username='emanuel43', password='ale',email='alexandru@yahoo.com')

    def test_get_all_users(self):
        client = Client()

        pk=self.alexandru.pk

        response = client.get('http://127.0.0.1:8000/api/users/{}/'.format(pk))

        user = User.objects.get(pk=self.alexandru.pk)

        serializer = UserSerializer(user)

        self.assertEqual(response.data, serializer.data)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

