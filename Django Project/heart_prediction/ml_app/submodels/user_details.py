
import datetime

from django.conf import settings
from django.contrib.auth.models import User, Group
from django.db import models
from django.db.models.signals import post_save
from django.dispatch import receiver
from rest_framework.authtoken.models import Token

User._meta.get_field('email')._unique = True
User._meta.get_field('email').blank = False
User._meta.get_field('email').null = False
# Create your models here.

class SexClass(models.TextChoices):
    Male = 'M'
    Female = 'F'
class BinaryChoices(models.IntegerChoices):
    Type_0 =0
    Type_1 =1
#create a token for every newly created user
@receiver(post_save, sender=settings.AUTH_USER_MODEL)
def create_auth_token(sender, instance=None, created=False, **kwargs):
    if created:
        Token.objects.get_or_create(user=instance)

# @receiver(post_save, sender=User)
# def create_user_profile(sender, instance, created, **kwargs):
#     if created:
#         instance.groups.add(Group.objects.get(name='patient'))

class UserDetailModel(models.Model):


    user=models.OneToOneField('auth.User',related_name='user',on_delete=models.CASCADE)
    sex = models.IntegerField(max_length=2, choices=BinaryChoices.choices, default=BinaryChoices.Type_1)
    birth_date=models.DateField(auto_now=False,auto_now_add=False,default=datetime.date(1997, 10, 19))
    description=models.CharField(max_length=200)
    def __str__(self):
        return "user borned at {} and with the sex {}".format(self.birth_date,self.sex)