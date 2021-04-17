import requests
from django.conf import settings
import json
import httplib2
from datetime import datetime

import json
import pytz
from ml_app.submodels.monitored_data import MonitoredData
import operator
from functools import reduce
class ExtractDataService:

    def __init__(self,start_time,end_time):
        self.start_time=start_time
        self.end_time=end_time
    def extract_data(self,token):
        requestBody = {"aggregateBy":[{"dataTypeName":"com.google.heart_rate.bpm"},
                                      {"dataTypeName":"com.google.step_count.delta",
                                       # "dataSourceId": "derived:com.google.step_count.delta:com.google.android.gms:estimated_steps"
                                       },
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
        data_source='empty'
        try:
            data=response.json()['bucket']
            val_list = []
            for item in data:
                startTimeNanos=int(item['startTimeMillis'])*pow(10,6)
                endTimeNanos=int(item['endTimeMillis']) * pow(10, 6)
                dataset=item['dataset']

                if(len(dataset)!=0):
                    for data_item in dataset:
                        if(len(data_item['point'])!=0):
                            data_source=data_item['dataSourceId']
                            for data_point in data_item['point']:
                                data_type=data_point['dataTypeName']
                                avg = 0
                                n=0

                                startTimeNanos=int(data_point['startTimeNanos'])
                                endTimeNanos=int(data_point['endTimeNanos'])
                                for data_value in data_point['value']:

                                    try:
                                        if(data_value['fpVal']):
                                            fp_val=data_value['fpVal']
                                            val_list.append(fp_val)

                                    except KeyError:
                                        pass
                                    try:

                                        if (data_value['intVal']):
                                            int_val = data_value['intVal']
                                            val_list.append(int_val)


                                    except KeyError:
                                        pass


                                    n+=1


                print("start time: %d" % ( startTimeNanos))
                start_time=datetime.fromtimestamp(startTimeNanos/pow(10,9),tz=pytz.UTC)

                end_time=datetime.fromtimestamp(endTimeNanos/pow(10,9),tz=pytz.UTC)
                end_time=datetime(end_time.year,end_time.month,end_time.day,end_time.hour,tzinfo=pytz.UTC)
                print("Good Start time: ",start_time)

                number_of_points=len(val_list)+1
                if data_source=="derived:com.google.heart_minutes.summary:com.google.android.gms:aggregated" and len(val_list) > 1:
                    recorded_value=reduce(operator.add,val_list)/number_of_points
                    monitored_data = MonitoredData(patient_id=2, api_value=recorded_value, start_time=start_time,
                                                   end_time=end_time,
                                                   activity_measure_type=data_source, data_type=data_type,
                                                   activity_description='')
                    monitored_data.save()
                else:
                    for i in range(len(val_list)):
                        recorded_value=val_list[i]
                        monitored_data=MonitoredData(patient_id=2,api_value=recorded_value,start_time=start_time,end_time=end_time,
                                                     activity_measure_type=data_source,data_type=data_type,activity_description='')
                        monitored_data.save()
        except KeyError:
            pass
        print('Response data is ',response)


