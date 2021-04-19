

from ml_app.submodels.health_record import *
from ml_app.submodels.model_configuration import *
from ml_app.submodels.user_details import *


class MonitoredData(models.Model):

    patient=models.ForeignKey(UserDetailModel,related_name='monitored_patient',on_delete=models.CASCADE)
    api_value=models.IntegerField(default=0,null=True)
    start_time=models.DateTimeField(null=True)
    end_time=models.DateTimeField(null=True)
    activity_description=models.CharField(max_length=200,blank=True)
    activity_source=models.CharField(max_length=300,blank=True)
    data_type=models.CharField(max_length=100,blank=False)
    created_date=models.DateTimeField(auto_now_add=True)


    def __str__(self):
        return 'We have a record with the value {} ,with start ' \
               'time {} and end time '.format(self.api_value, self.start_time,self.end_time)

    def save(self, *args, **kwargs):
        super(MonitoredData, self).save(*args, **kwargs)