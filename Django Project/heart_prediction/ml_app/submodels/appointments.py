import pytz
from django.contrib.auth.models import User
from ml_app.submodels.patient_model import Patient
import django.db.models as models
from django.core.validators import MinValueValidator,MaxValueValidator
from datetime import datetime
from dateutil.relativedelta import relativedelta

utc = pytz.UTC
class Appointments(models.Model):

    patient = models.ForeignKey(Patient, related_name='appoint_patients', on_delete=models.CASCADE)
    doctor=models.ForeignKey(User,related_name='appoint_doctors',on_delete=models.CASCADE)
    time=models.DateTimeField(blank=False,validators=[MinValueValidator(datetime.now(tz=utc)),
                                                             MaxValueValidator(datetime.now(tz=utc)+relativedelta(years=1))])

    def __str__(self):
        user=Patient.objects.get(pk=self.patient_id)
        auth_user=User.objects.get(pk=user.user_id)
        return 'Patient with name {} has an appointment on {} at the  doctor id {}  '.format(auth_user.first_name,self.time,self.doctor_id)

