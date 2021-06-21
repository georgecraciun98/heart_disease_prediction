import datetime

from django.conf import settings
from django.contrib.auth.models import User, Group
from django.core.validators import MinValueValidator, MaxValueValidator
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

class BoosterChoices(models.TextChoices):
    Type_0="gbtree"
    Type_1="gblinear"

class MaxFeatures(models.TextChoices):
    Type_0="auto"
    Type_1="sqrt"

class KernelChoices(models.TextChoices):
    Type_0="linear"
    Type_1="poly"
    Type_2 = "rbf"

class SolverChoices(models.TextChoices):
    Type_0 = "newton-cg"
    Type_1 = "lbfgs"
    Type_2 = "liblinear"
    Type_3 = "sag"
    Type_4 = "saga"
class PenaltyChoices(models.TextChoices):
    Type_0 = "none"
    Type_1 = "l1"
    Type_2 = "l2"
    Type_3 = "elasticnet"

class ModelConfiguration(models.Model):

    researcher=models.ForeignKey(User,related_name='researchers',on_delete=models.CASCADE)
    alg_name = models.CharField(max_length=30, choices=AlgName.choices, default=AlgName.Svm_keras)
    alg_description=models.CharField(max_length=100,null=True)
    precision=models.FloatField(null=True)
    accuracy=models.FloatField(null=True)
    f1_score=models.FloatField(null=True)
    recall_score=models.FloatField(null=True)
    roc_auc_score=models.FloatField(null=True)

    target=models.IntegerField(choices=BinaryChoices.choices,null=True)

    n_estimators=models.IntegerField(null=True)
    max_depth = models.IntegerField( null=True)
    booster = models.CharField(max_length=12,choices=BoosterChoices.choices, null=True)
    base_score = models.FloatField( null=True)
    learning_rate = models.FloatField( null=True)
    min_child_weight = models.FloatField( null=True)

    max_features = models.CharField(max_length=20,null=True,choices=MaxFeatures.choices)
    min_samples_split = models.IntegerField(null=True)
    min_samples_leaf=models.IntegerField(null=True)
    bootstrap=models.BooleanField(null=True)
    created_date=models.DateTimeField(auto_now_add=True)
    source_file=models.FileField(null=True,blank=True,upload_to='media/documents/')
    #svm c,gamma,kernel

    c = models.FloatField(validators=[MinValueValidator(0.1),MaxValueValidator(20)],null=True)
    gamma = models.FloatField(validators=[MinValueValidator(0.0001),MaxValueValidator(1)],null=True)
    kernel = models.CharField(max_length=20,null=True,choices=KernelChoices.choices)

    #binary classifier c, solver , penalty

    solver=models.CharField(max_length=20,null=True, choices=SolverChoices.choices)
    penalty=models.CharField(max_length=20,null=True,choices=PenaltyChoices.choices)
    active=models.BooleanField(null=True,default=False)
    def __str__(self):
        # researcher=User.objects.get(pk=self.researcher_id)
        return 'Model with the algorithm {} , precision {} , accuracy {}, ' \
               'created by researcher {} '.format(self.alg_name,self.precision,self.accuracy,
                                                  self.researcher.first_name)
