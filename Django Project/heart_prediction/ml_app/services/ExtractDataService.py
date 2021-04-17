import requests
from django.conf import settings
import json
import httplib2
from datetime import datetime

import json

from ml_app.submodels.monitored_data import MonitoredData


class ExtractDataService:

    def __init__(self,start_time,end_time):
        self.start_time=start_time
        self.end_time=end_time
        print('start time is',start_time,'end_time is', end_time)
    def extract_data(self,token):
        requestBody = {"aggregateBy":[{"dataTypeName":"com.google.heart_rate.bpm"},
                                      {"dataTypeName":"com.google.step_count.delta",
                                       "dataSourceId": "derived:com.google.step_count.delta:com.google.android.gms:estimated_steps"},
                                      {"dataTypeName":"com.google.heart_minutes"},
                                      {"dataTypeName":"com.google.sleep.segment"}],
                        "bucketByTime": {"durationMillis": 86400000},
                        "endTimeMillis": self.end_time,"startTimeMillis": self.start_time

                        }
        request_body = json.dumps(requestBody)

        headers = {'Content-type': 'application/json',
                   'Authorization': "Bearer "+token}
        print('json object is',request_body)
        response=requests.post('https://www.googleapis.com/fitness/v1/users/me/dataset:aggregate',data=request_body,headers=headers)
        try:
            data=response.json()['bucket']
            for item in data:
                dataset=item['dataset']
                if(len(dataset)!=0):
                    for data_item in dataset:
                        if(len(data_item['point'])!=0):
                            for data_points in data_item['point']:
                                avg = 0
                                n=0
                                try:
                                    data_source=data_points['originDataSourceId']
                                except KeyError:
                                    data_source='empty'

                                for data_value in data_points['value']:

                                    try:
                                        if(data_value['fpVal']):
                                            fp_val=data_value['fpVal']
                                            data_record=fp_val
                                    except KeyError:
                                        pass
                                    try:

                                        if (data_value['intVal']):
                                            int_val = data_value['intVal']
                                            data_record=int_val

                                    except KeyError:
                                        pass

                                    avg+=data_record
                                    n+=1
                                recorded_value=avg/n

                print("start time: %d" % ( int(item['startTimeMillis'])))
                start_time=datetime.fromtimestamp(int(item['startTimeMillis'])/1000.0)
                end_time=datetime.fromtimestamp(int(item['endTimeMillis'])/1000.0)
                monitored_data=MonitoredData(patient_id=2,api_value=recorded_value,start_time=start_time,end_time=end_time,
                                             activity_measure_type=data_source,activity_description='')
                monitored_data.save()
        except KeyError:
            pass
        print('Response data is ',response)


