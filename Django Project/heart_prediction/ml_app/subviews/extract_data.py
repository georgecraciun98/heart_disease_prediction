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
def calculate_intervals():
    """
    Here we calculate how many time intervals of one month we will need
    :return:
    """
    #February
    #start_time=1612415200000
    start_time = 1606793460000
    len=1
    #one month miliseconds
    month_milis=2592000000
    end_time = current_timestamp()
    interval=round((end_time-start_time)/month_milis)+1
    time_intervals=[]
    for i in range(interval):
        time_intervals.append([start_time,start_time+month_milis])
        start_time+=2592000000
    return interval,time_intervals

"""
This service extracts google fit data 
You need to provide a google token
"""
class ExtractData(generics.GenericAPIView):
    permission_classes = [permissions.IsAuthenticatedOrReadOnly,
                         IsPatient]

    def post(self,request):
        token=request.data['token']
        print('google token is',token)
        patient=request.user.user
        end_time=current_timestamp()
        if(MonitoredData.objects.filter(patient_id=patient.id).exists()):
            data_obj = MonitoredData.objects.filter(patient_id=patient.id).order_by('-end_time').first()
            date=data_obj.end_time
            # date_time=datetime.combine(date.today(), datetime.min.time())
            data_milis=round(date.timestamp())*1000
            start_time=data_milis+1
            #create from last extraction
            extractDataInstance = ExtractDataService(start_time, end_time,patient_id=patient.id)
            extractDataInstance.extract_data(token=token)
        else:
            interval,time_intervals=calculate_intervals()
            for i in range(interval):
                start_time,end_time=time_intervals[i]
                extractDataInstance = ExtractDataService(start_time, end_time,patient_id=patient.id)
                extractDataInstance.extract_data(token=token)
        return Response("Extracted Data", status=status.HTTP_200_OK)
