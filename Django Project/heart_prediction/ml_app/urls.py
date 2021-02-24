from django.urls import include, path
from rest_framework import routers

from ml_app import views
from .api import HealthRecordViewSet

router=routers.DefaultRouter()
router.register('healthrecord',HealthRecordViewSet,'health')

urlpatterns = [
    path('', views.api_root),
    path('health_record/', views.HealthRecordList.as_view(),name='record-list'),
    path('health_record/<int:pk>/',views.HealthRecordDetail.as_view(),name='record-detail'),
    path('users/', views.UserList.as_view(),name='user-list'),
    path('users/<int:pk>/', views.UserDetail.as_view(),name='user-detail'),
    path('auth/',include('djoser.urls')),
    path('auth/',include('djoser.urls.authtoken')),



    # path('token-auth/', include('rest_framework.urls', namespace='rest_framework'))
]
