import pytz
from rest_framework import generics, status
from rest_framework import permissions
from rest_framework.response import Response
from ml_app.serializers.monitored_data_serializer import MonitoredDataSerializer
from ml_app.sub_permissions.group_permissions import IsDoctor, IsPatient
from ml_app.submodels.monitored_data import MonitoredData
from ml_app.submodels.patient_model import Patient
from datetime import date, datetime


class MonitoredDataList(generics.ListAPIView):
    model= Patient
    permission_classes = [permissions.IsAuthenticatedOrReadOnly,IsPatient
                          ]

    serializer_class = MonitoredDataSerializer

    def get_queryset(self):
        patient_id=self.request.user.user.id
        queryset = MonitoredData.objects.filter(patient_id=patient_id)
        return queryset
    def get_by_activity(self,data_type,start_time,end_time):
        patient_id=self.request.user.user.id
        queryset = MonitoredData.objects.filter(patient_id=patient_id,data_type=data_type,
                                                start_time__gte=start_time,end_time__lte=end_time)
        return queryset
    def perform_create(self, serializer):
        serializer.save(user_id=self.request.user.pk)

    """
    Get all monitored data by patient or by doctor
    """
    def list(self, request, *args, **kwargs):
        patient = request.user.user
        queryset = self.filter_queryset(self.get_queryset())

        serializer = self.get_serializer(queryset, many=True)

        return Response(serializer.data)
    """
    Get data by activity and prefered timestamp
    """
    def post(self, request, *args, **kwargs):
        patient = request.user.user
        try:
            data_type = request.data['data_type']
            start_time_string = request.data['start_time']
            try:
                start_time_obj = datetime.strptime(start_time_string, '%d/%m/%y %H:%M:%S')
            except KeyError:
                start_time_obj = datetime.strptime(start_time_string, '%d/%m/%y %H:%M:%f')
            except ValueError:
                start_time_obj = datetime.strptime('24/10/20 12:12:12', '%d/%m/%y %H:%M:%f')
            start_time_tz=start_time_obj.replace(tzinfo=pytz.UTC)
            end_time_string = request.data['end_time']
            try:
                end_time_obj = datetime.strptime(end_time_string, '%d/%m/%y %H:%M:%S')
            except KeyError:
                end_time_obj = datetime.strptime(end_time_string, '%d/%m/%y %H:%M:%f')
            except ValueError:
                end_time_obj = datetime.now()
            end_time_tz=end_time_obj.replace(tzinfo=pytz.UTC)
            queryset = self.filter_queryset(self.get_by_activity(data_type, start_time_tz, end_time_tz))

            serializer = self.get_serializer(queryset, many=True)
            print('everything is good',serializer.data)
            return Response(serializer.data,status=status.HTTP_200_OK)


        except KeyError:
            print('bad',queryset)

            print("wrong data is ",data_type,data_type)

            return Response({"Your parameters contain invalid values"}, status=status.HTTP_400_BAD_REQUEST)
        except ValueError:
            print('very bad',data_type)
            print('start_time_string', start_time_string)
            print('start_time_obj', start_time_obj)
            print('start_time_tz', start_time_tz)
            print('queryset', queryset)
            print("wrong data is ", data_type, data_type)
            return Response({"Your parameters contain invalid values"}, status=status.HTTP_400_BAD_REQUEST)

