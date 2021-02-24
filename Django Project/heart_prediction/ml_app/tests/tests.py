from django.contrib.auth.models import User
from django.test import TestCase, Client
from rest_framework import status

from ml_app.serializers.user_serializers import UserSerializer


class GetAllUserTest(TestCase):
    """ Test module for GET all puppies API """

    def setUp(self):
        User.objects.create(
            username='alexandru32', password='alexandru32')
        User.objects.create(
            username='marian238', password='marian238')
        User.objects.create(
            username='emanuel43', password='alexandru32')

    def test_get_all_users(self):
        client = Client()

        #client.login(username='alexandru232', password='superuser123')
        response = client.get('http://127.0.0.1:8000/api/users/')

        users=User.objects.order_by('id').all()
        serializer=UserSerializer(users, many=True)

        self.assertEqual(response.data,serializer.data)
        self.assertEqual(response.status_code,status.HTTP_200_OK)



class GetSingleUserTest(TestCase):
    """ Test module for GET all puppies API """

    def setUp(self):
        self.alexandru = User.objects.create(
            username='alexandru32', password='alexandru32')
        self.marian = User.objects.create(
            username='marian238', password='marian238')
        self.emanuel = User.objects.create(
            username='emanuel43', password='alexandru32')

    def test_get_all_users(self):
        client = Client()
        pk=self.alexandru.pk

        response = client.get('http://127.0.0.1:8000/api/users/{}/'.format(pk))

        user = User.objects.get(pk=self.alexandru.pk)

        serializer = UserSerializer(user)

        self.assertEqual(response.data, serializer.data)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

class ShortPasswordTestCase(TestCase):
    """ Test module for GET all puppies API """

    def setUp(self):
        self.alexandru = User.objects.create(
            username='alexandru32', password='a')
        self.marian = User.objects.create(
            username='marian238', password='ma')
        self.emanuel = User.objects.create(
            username='emanuel43', password='ale')

    def test_get_all_users(self):
        client = Client()
        pk=self.alexandru.pk

        response = client.get('http://127.0.0.1:8000/api/users/{}/'.format(pk))

        user = User.objects.get(pk=self.alexandru.pk)

        serializer = UserSerializer(user)

        self.assertEqual(response.data, serializer.data)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

# class CreateNewUserTest(TestCase):
#     """ Test module for GET all puppies API """
#
#     def setUp(self):
#         self.valid_payload = {
#             'name': 'Muffin',
#             'age': 4,
#             'breed': 'Pamerion',
#             'color': 'White'
#         }
#         self.invalid_payload = {
#             'name': '',
#             'age': 4,
#             'breed': 'Pamerion',
#             'color': 'White'
#         }
#
#     def test_get_all_users(self):
#         client = Client()
#         pk=self.alexandru.pk
#
#         response = client.get('http://127.0.0.1:8000/api/users/{}/'.format(pk))
#
#         user = User.objects.get(pk=self.alexandru.pk)
#
#         serializer = UserSerializer(user)
#
#         self.assertEqual(response.data, serializer.data)
#         self.assertEqual(response.status_code, status.HTTP_200_OK)