import requests
from datetime import datetime
import json
import pytz
from ml_app.submodels.monitored_data import MonitoredData

class ExtractDataService:

    def __init__(self,start_time,end_time,patient_id):
        self.start_time=start_time
        self.end_time=end_time
        self.patient_id=patient_id

    def get_body(self,token):
        requestBody = {"aggregateBy": [{"dataTypeName": "com.google.heart_rate.bpm",
                                        "dataSourceId":"derived:com.google.heart_rate.bpm:com.google.android.gms:merge_heart_rate_bpm"},
                                       {"dataTypeName": "com.google.step_count.delta",},
                                       {"dataTypeName": "com.google.height"
                                        },
                                       {"dataTypeName":"com.google.height.summary"
                                        ,"dataSourceId":"derived:com.google.heart_rate.bpm:com.google.android.gms:merge_heart_rate_bpm"},
                                       {"dataTypeName": "com.google.heart_minutes"},
                                       {"dataTypeName": "com.google.sleep.segment"},
                                       {"dataTypeName": "com.google.step_count.delta"},
                                       {"dataTypeName": "com.google.active_minutes"},
                                       {"dataTypeName": "com.google.active_minutes",
                                        "dataSourceId":"derived:com.google.active_minutes:com.google.android.gms:merge_active_minutes"},
                                       {"dataTypeName": "com.google.weight"},
                                       {"dataTypeName": "com.google.speed"},
                                       {"dataTypeName": "com.google.calories.bmr",
                                        "dataSourceId":"derived:com.google.calories.bmr:com.google.android.gms:from_height&weight"},
                                       {"dataTypeName": "com.google.calories.bmr",
                                        "dataSourceId": "derived:com.google.calories.bmr:com.google.android.gms:merged"},
                                       {"dataTypeName": "com.google.calories.bmr",
                                        "dataSourceId": "derived:com.google.calories.bmr:com.google.android.gms:from_height&weight"},
                                       {"dataTypeName": "com.google.calories.expended",
                                        "dataSourceId": "derived:com.google.calories.expended:com.google.android.gms:merge_calories_expended"
                                        },
                                       {"dataTypeName": "com.google.activity.summary",
                                        "dataSourceId":"derived:com.google.activity.segment:nl.appyhapps.healthsync:session_activity_segment"},
                                       {"dataTypeName": "com.google.distance.delta"},
                                       {"dataTypeName": "com.google.sleep.segment"},
                                       ],
                       "bucketByTime": {"durationMillis": 86400000},
                       "endTimeMillis": self.end_time, "startTimeMillis": self.start_time

                       }
        request_body = json.dumps(requestBody)
        headers = {'Content-type': 'application/json',
                   'Authorization': "Bearer " + token}
        print('json object is', request_body)
        return  request_body,headers


    def extract_data(self,token):

        request_body,headers=self.get_body(token)

        response=requests.post('https://www.googleapis.com/fitness/v1/users/me/dataset:aggregate',data=request_body,
                               headers=headers)
        data_source='empty'
        activity_description=''
        try:
            data=response.json()['bucket']
            activity_value = 0
            for item in data:
                startTimeNanos=int(item['startTimeMillis'])*pow(10,6)
                endTimeNanos=int(item['endTimeMillis']) * pow(10, 6)
                dataset=item['dataset']

                if(len(dataset)!=0):
                    # we iterate to dataset elements , basically each value recorded
                    for data_item in dataset:
                        nr_points = 0
                        if(len(data_item['point'])!=0):

                            data_source=data_item['dataSourceId']
                            for data_point in data_item['point']:
                                data_type=data_point['dataTypeName']
                                nr_points+=1
                                startTimeNanos=int(data_point['startTimeNanos'])
                                endTimeNanos=int(data_point['endTimeNanos'])

                                if data_type=='com.google.heart_rate.summary':
                                    avg=data_point['value'][0]['fpVal']
                                    activity_value=avg
                                    activity_description='Average heart rate'
                                elif data_type=='com.google.heart_minutes.summary':
                                    activity_value=data_point['value'][0]['fpVal']
                                    activity_description='Heart minutes'
                                elif data_type == 'com.google.step_count.delta':
                                    activity_value = data_point['value'][0]['intVal']
                                    activity_description = 'Daily step count'
                                elif data_type == 'com.google.active_minutes':
                                    try:

                                        activity_value = data_point['value'][0]['intVal']
                                    except KeyError:
                                        activity_value = data_point['value'][0]['fpVal']

                                    activity_description = 'Active Minutes'
                                elif data_type == 'com.google.calories.expanded':
                                    try:

                                        activity_value = data_point['value'][0]['intVal']
                                    except KeyError:
                                        activity_value = data_point['value'][0]['fpVal']

                                    activity_description = 'Minimum required Calories'
                                elif data_type == 'com.google.calories.expended':
                                    try:

                                        activity_value = data_point['value'][0]['intVal']
                                    except KeyError:
                                        activity_value = data_point['value'][0]['fpVal']

                                    activity_description = 'Expended Calories'
                                elif data_type == 'com.google.distance.delta':
                                    try:

                                        activity_value = data_point['value'][0]['intVal']
                                    except KeyError:
                                        activity_value = data_point['value'][0]['fpVal']

                                    activity_description = 'Distance Traveled in meters Since last Reading'
                                elif data_type == 'com.google.speed.summary':
                                    try:

                                        activity_value = data_point['value'][0]['intVal']
                                    except KeyError:
                                        activity_value = data_point['value'][0]['fpVal']

                                    activity_description = 'Average speed meters/second'

                                elif data_type == 'com.google.weight.summary':
                                    try:

                                        activity_value = data_point['value'][0]['intVal']
                                    except KeyError:
                                        activity_value = data_point['value'][0]['fpVal']

                                    activity_description = 'Weight in kgs'
                                elif data_type == 'com.google.height.summary':
                                    try:

                                        activity_value = data_point['value'][0]['fpVal']
                                    except KeyError:
                                        activity_value = data_point['value'][0]['intVal']
                                    print('height in meters is',activity_value)
                                    activity_description = 'Height in meters'
                                else:
                                    try:

                                        activity_value = data_point['value'][0]['intVal']
                                    except KeyError:
                                        activity_value = data_point['value'][0]['fpVal']
                                    activity_description='other'
                        else:
                            nr_points=0

                        start_time=datetime.fromtimestamp(startTimeNanos/pow(10,9),tz=pytz.UTC)

                        end_time=datetime.fromtimestamp(endTimeNanos/pow(10,9),tz=pytz.UTC)
                        
                        activity_value=round(activity_value,2)
                        if nr_points==1:
                            recorded_value=activity_value
                            monitored_data = MonitoredData(patient_id=self.patient_id, api_value=recorded_value,
                                                           start_time=start_time,end_time=end_time,
                                                           activity_source=data_source, data_type=data_type,
                                                           activity_description=activity_description)
                            monitored_data.save()
                        elif nr_points>1:
                            print('number of points is greater')
                            for i in range(len(activity_value)):
                                recorded_value=activity_value[i]
                                monitored_data=MonitoredData(patient_id=self.patient_id,api_value=recorded_value,
                                                             start_time=start_time,end_time=end_time,
                                                             activity_source=data_source,data_type=data_type,
                                                             activity_description='activity_description')
                                monitored_data.save()
        except KeyError:
            pass
        print('Response data is ',response)


