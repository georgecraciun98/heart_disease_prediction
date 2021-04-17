from django.contrib.auth.models import User
from rest_framework import generics, status
from rest_framework import permissions
from rest_framework.decorators import api_view
from rest_framework.response import Response


from ml_app.services.ExtractDataService import ExtractDataService
from ml_app.sub_permissions.group_permissions import IsPatient,IsDoctor


from datetime import datetime,tzinfo
from django.http import Http404
import pytz
from ml_app.submodels.monitored_data import MonitoredData


def current_timestamp():
    return round(datetime.now().timestamp())*1000
class ExtractData(generics.GenericAPIView):
    permission_classes = [permissions.IsAuthenticatedOrReadOnly,
                         IsPatient]

    def post(self,request):
        token=request.data['token']
        print('extracted token is',token)
        end_time=current_timestamp()
        if(MonitoredData.objects.filter(patient_id=2).exists()):
            data_obj = MonitoredData.objects.filter(patient_id=2).order_by('-end_time').first()
            date=data_obj.end_time
            date_time=datetime.combine(date.today(), datetime.min.time())
            data_milis=round(date_time.timestamp())
            start_time=data_milis+1
        else:
            start_time=1612130400000

        extractDataInstance=ExtractDataService(start_time,end_time)
        extractDataInstance.extract_data(token=token)
        return Response("Extracted Data", status=status.HTTP_200_OK)
