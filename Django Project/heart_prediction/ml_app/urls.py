from django.urls import include, path,re_path
from rest_framework import routers
from django.views.generic import TemplateView
from ml_app import views
from ml_app.api import HealthRecordViewSet
from ml_app.subviews.doctor_view import PatientList,RecordList, RecordDetail, PatientPrediction, PatientDetail, Models
from ml_app.subviews.monitored_data_view import MonitoredDataList, MonitoredDataDoctor
from ml_app.subviews.user_view import UserDetailView
from ml_app.subviews.extract_data import ExtractData
router=routers.DefaultRouter()
router.register('healthrecord',HealthRecordViewSet,'health')

urlpatterns = [
    path('', views.api_root),
    path('auth/', include('djoser.urls')),
    path('auth/', include('djoser.social.urls')),
    path('auth/', include('djoser.urls.authtoken')),
    path('health_record/', views.HealthRecordList.as_view(),name='record-list'),
    path('health_record/<int:pk>/',views.HealthRecordDetail.as_view(),name='record-detail'),
    #All the users
    path('users/', views.UserList.as_view(),name='user-list'),
    #Current User
    path('users/me',views.UserDetail.as_view(),name='user-me'),
    #Show details for a patient,patient calls this route
    path('patients/detail/', UserDetailView.as_view(),name='user-detail-true'),
    #doctor perspective
    path('patients/', PatientList.as_view(),name='user-detail-true'),
    path('patients/<int:pk>/', PatientDetail.as_view(), name='patient-detail'),
    path('patients/<int:pk>/health_record/list', RecordList.as_view(),name='list-create-record'),
    path('patients/<int:pk>/health_record/', RecordDetail.as_view(),name='retrieve-update-record'),
    path('patients/<int:pk>/predict/',PatientPrediction.as_view(),name='predict'),
    path('patients/<int:pk>/monitored_data/list', RecordList.as_view(), name='get-patient-info'),

    path('patients/monitored_data/list', MonitoredDataList.as_view(), name='retrieve-patient-info'),
    path('patients/monitored_data/list/<int:pk>', MonitoredDataDoctor.as_view(), name='retrieve-patient-info'),

    path('models/', Models.as_view(),name='model-list'),

    path('extract_data/',ExtractData.as_view(),name='extract-data')
]

#urlpatterns += [re_path(r'^.*',TemplateView.as_view(template_name='index.html'))]
