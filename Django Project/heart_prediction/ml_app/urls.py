from django.urls import include, path,re_path
from rest_framework import routers
from django.views.generic import TemplateView
from ml_app import views
from ml_app.api import HealthRecordViewSet
from ml_app.subviews.doctor_view import PatientList,RecordList, RecordDetail,\
    PatientPrediction, PatientDetail
from ml_app.subviews.monitored_data_view import MonitoredDataList, MonitoredDataDoctor
from ml_app.subviews.researcher_view import ModelFileUpload,PredictionModelsTraining,PredictionModelsSaving
from ml_app.subviews.user_view import UserDetailView
from ml_app.subviews.extract_data import ExtractData
from ml_app.subviews.doctor_list import DoctorList, AppointmentGet,AppointmentByUser,AppointmentPatient,AppointmentDoctor

router=routers.DefaultRouter()
router.register('healthrecord',HealthRecordViewSet,'health')

urlpatterns = [
    path('', views.api_root),
    path('auth/', include('djoser.urls')),
    path('auth/', include('djoser.social.urls')),
    path('auth/', include('djoser.urls.authtoken')),
    #retrieve all health records
    path('health_record/', views.HealthRecordList.as_view(),name='record-list'),
    #retrieve specific health record
    path('health_record/<int:pk>/',views.HealthRecordDetail.as_view(),name='record-detail'),
    #All the users
    path('users/', views.UserList.as_view(),name='user-list'),
    #Current User
    path('users/me',views.UserDetail.as_view(),name='user-me'),

    #Doctor list
    path('doctors/',DoctorList.as_view(),name='doctor-list'),

    #Get appointments by doctor id
    path('appointments/<int:pk>/',AppointmentGet.as_view(),name='appointment-doctors'),
    path('appointments/by_user/<int:pk>/', AppointmentByUser.as_view(), name='appointment-users'),

    path('appointments/<int:pk>/hours', AppointmentDoctor.as_view(), name='appointment-hours'),

    # Get appointments by patient id
    path('appointments/by_patient/<int:pk>/', AppointmentPatient.as_view(), name='appointment-doctors'),

    #Show details for a patient,patient calls this route
    path('patients/detail/', UserDetailView.as_view(),name='user-detail-true'),
    #doctor can retrieve all patients
    path('patients/', PatientList.as_view(),name='user-detail-true'),
    #doctor can retrieve one patient
    path('patients/<int:pk>/', PatientDetail.as_view(), name='patient-detail'),

    #doctor can see patient medical health records
    path('patients/<int:pk>/health_record/list', RecordList.as_view(),name='list-create-record'),
    #retrieve patients't last record , or create a new one
    path('patients/<int:pk>/health_record/', RecordDetail.as_view(),name='retrieve-update-record'),
    #doctor can make a prediction on a specific patient
    path('patients/<int:pk>/predict/',PatientPrediction.as_view(),name='predict'),

    path('patients/<int:pk>/monitored_data/list', RecordList.as_view(), name='get-patient-info'),

    #patient can see his own monitored data
    path('patients/monitored_data/list', MonitoredDataList.as_view(), name='retrieve-patient-info'),

    #doctor can see patient monitored data by patient id
    path('patients/monitored_data/list/<int:pk>', MonitoredDataDoctor.as_view(), name='retrieve-patient-info'),

    #Researcher View
    #shows all models
    path('models/train', PredictionModelsTraining.as_view(), name='model-list'),
    path('models/save', PredictionModelsSaving.as_view(), name='model-save'),
    path('models/file_upload',ModelFileUpload.as_view(), name='model-file'),
    # This route is used for google fit data extraction
    path('extract_data/',ExtractData.as_view(),name='extract-data')
]

#urlpatterns += [re_path(r'^.*',TemplateView.as_view(template_name='index.html'))]
