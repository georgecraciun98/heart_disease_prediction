from django.db import models
from ml_app.validators.health_record_validator import validate_age,\
    validate_trebtps,validate_thalach,validate_oldpeak
from django.conf import settings
from django.db.models.signals import post_save
from django.dispatch import receiver
from rest_framework.authtoken.models import Token
from django.contrib.auth.models import User

# Create your models here.
class HealthRecordModel(models.Model):
    class SexClass(models.TextChoices):
        Male='M'
        Female='F'
    class ChestPainType(models.IntegerChoices):
        Pain_0=0
        Pain_1 = 1
        Pain_2 = 2
        Pain_3 = 3
    class BinaryChoices(models.IntegerChoices):
        Type_0=0
        Type_1=1
    class ThalChoices(models.IntegerChoices):
        Type_0=1
        Type_1=3
        Type_2=6
        Type_3=7
    class ZeroToTwo(models.IntegerChoices):
        Type_0=0
        Type_1=1
        Type_2=2
    #user=models.ForeignKey(User,on_delete=models.DO_NOTHING,related_name="user_id",blank=True,null=True)
    user = models.ForeignKey('auth.User', related_name='records', on_delete=models.CASCADE)

    age=models.FloatField(validators=[validate_age])
    sex=models.CharField(max_length=2,choices=SexClass.choices,default=SexClass.Male)
    """
    cp - chest pain type
        0: Typical angina: chest pain related decrease blood supply to the heart
        1: Atypical angina: chest pain not related to heart
        2: Non-anginal pain: typically esophageal spasms (non heart related)
        3: Asymptomatic: chest pain not showing signs of disease
    """
    cp=models.IntegerField(choices=ChestPainType.choices,
                           default=ChestPainType.Pain_0)
    """
    trestbps - resting blood pressure (in mm Hg on admission to the hospital) 
    anything above 130-140 is typically cause for concern
    """
    trestbps=models.IntegerField(validators=[validate_trebtps])
    """
    chol - serum cholestoral in mg/dl
    """
    chol =models.IntegerField()
    """
    fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
     '>126' mg/dL signals diabetes
    """
    fbs=models.IntegerField(choices=BinaryChoices.choices)
    """
    restecg - resting electrocardiographic results
        0: Nothing to note
        1: ST-T Wave abnormality
        can range from mild symptoms to severe problems
        signals non-normal heart beat
        2: Possible or definite left ventricular hypertrophy
        Enlarged heart's main pumping chamber
    """
    restecg=models.IntegerField(choices=ZeroToTwo.choices)
    """
    thalach - maximum heart rate achieved
    
    """
    thalach = models.IntegerField(validators=[validate_thalach])
    #exercise induced angina (1 = yes; 0 = no)
    exang = models.IntegerField(choices=BinaryChoices.choices)
    """
    oldpeak - ST depression induced by exercise relative to rest looks 
    at stress of heart during excercise unhealthy heart will stress more
    """
    oldpeak = models.FloatField(validators=[validate_oldpeak])
    """
    slope - the slope of the peak exercise ST segment
        0: Upsloping: better heart rate with excercise (uncommon)
        1: Flatsloping: minimal change (typical healthy heart)
        2: Downslopins: signs of unhealthy heart
    """
    slope = models.IntegerField(choices=ZeroToTwo.choices)
    """
    ca - number of major vessels (0-3) colored by flourosopy
    colored vessel means the doctor can see the blood passing through
    the more blood movement the better (no clots)
    
    """
    ca = models.IntegerField(choices=BinaryChoices.choices)
    """
    thal - thalium stress result
        1,3: normal
        6: fixed defect: used to be defect but ok now
        7: reversable defect: no proper blood movement when excercising
    """
    thal = models.IntegerField(choices=ThalChoices.choices)
    """
    target - have disease or not (1=yes, 0=no) (= the predicted attribute)
    """
    target=models.IntegerField(blank=True,choices=BinaryChoices.choices)

    def __str__(self):
        return 'Patient age: '+str(self.age)+' thal: '+str(self.thal)+' trestbps: '+str(self.trestbps)

    def save(self, *args, **kwargs):
        """
        Use the `pygments` library to create a highlighted HTML
        representation of the code snippet.
        """

        super(HealthRecordModel, self).save(*args, **kwargs)

    class Meta:
        verbose_name_plural = "HeathRecords"

@receiver(post_save, sender=settings.AUTH_USER_MODEL)
def create_auth_token(sender, instance=None, created=False, **kwargs):
    if created:
        Token.objects.get_or_create(user=instance)