
from django.db import models

class ModelParameters(models.Model):
    parameter1 = models.CharField(max_length=12,null=True)
    parameter2 = models.CharField(max_length=12, null=True)
    parameter3 = models.CharField(max_length=12, null=True)
    parameter4 = models.CharField(max_length=12, null=True)
    parameter5 = models.CharField(max_length=12, null=True)
    parameter6 = models.CharField(max_length=12, null=True)
    parameter7 = models.CharField(max_length=12, null=True)
    parameter8 = models.CharField(max_length=12, null=True)
    parameter9 = models.CharField(max_length=12, null=True)
    parameter10 = models.CharField(max_length=12, null=True)

    def __str__(self):
        return 'Model with the following parameters {}, {}, {} '.format(self.alg_name)
