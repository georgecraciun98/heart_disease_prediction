from django.urls import include, path
from rest_framework import routers

from ml_app import views
from ml_app.api import HealthRecordViewSet
from ml_app.subviews.doctor_view import PatientList, PatientAddRecord, PatientPrediction, PatientDetail, Models
from ml_app.subviews.user_view import UserDetailView

router=routers.DefaultRouter()
router.register('healthrecord',HealthRecordViewSet,'health')

urlpatterns = [
    path('', views.api_root),
    path('auth/',include('djoser.urls')),
    path('auth/',include('djoser.social.urls')),
    path('auth/',include('djoser.urls.authtoken')),
    path('health_record/', views.HealthRecordList.as_view(),name='record-list'),
    path('health_record/<int:pk>/',views.HealthRecordDetail.as_view(),name='record-detail'),
    #All the users
    path('users/', views.UserList.as_view(),name='user-list'),
    #Current User
    path('users/me',views.UserDetail.as_view(),name='user-me'),
    #Show details for a patient,patient calls this route
    path('patients/detail/', UserDetailView.as_view(),name='user-detail-true'),

    #path('account_info/',UserDetailModel.as_view(),name='user-detail'),

    # path('token-auth/', include('rest_framework.urls', namespace='rest_framework'))
    #doctor perspective
    path('patients/', PatientList.as_view(),name='user-detail-true'),
    path('patients/<int:pk>/', PatientDetail.as_view(),name='insert-record'),
    path('patients/<int:pk>/health_record/', PatientAddRecord.as_view(),name='insert-record'),
    path('patients/<int:pk>/predict/',PatientPrediction.as_view(),name='predict'),

    path('models/', Models.as_view(),name='model-list'),
]
