import datetime

from django.conf import settings
from django.contrib.auth.models import User, Group
from django.db import models
from django.db.models.signals import post_save
from django.dispatch import receiver
from rest_framework.authtoken.models import Token


class AlgName(models.TextChoices):
    Svm_keras = 'Support Vector Machine'
    Random_forest = 'Random Forest Classifier'
    Xgb_classifier='XGB Classifier'
    Binary_classifier='Binary Classifier'

class BinaryChoices(models.IntegerChoices):
    Type_0=0
    Type_1=1
class ModelConfiguration(models.Model):

    researcher=models.ForeignKey(User,related_name='researchers',on_delete=models.CASCADE)
    alg_name = models.CharField(max_length=30, choices=AlgName.choices, default=AlgName.Svm_keras)
    alg_description=models.CharField(max_length=100,null=True)
    precision=models.FloatField()
    accuracy=models.FloatField()
    target=models.IntegerField(choices=BinaryChoices.choices)
    parameter1=models.CharField(max_length=12,null=True)
    parameter2 = models.CharField(max_length=12, null=True)
    parameter3 = models.CharField(max_length=12, null=True)


    def __str__(self):
        researcher=User.objects.get(pk=self.researcher_id)
        return 'Model with the algorithm {} , precision {} , accuracy , ' \
               'created by researcher {} '.format(self.alg_name,self.precision,self.accuracy,
                                                  researcher.first_name)
