from django.urls import include, path

from .api import HealthRecordViewSet
from rest_framework import routers

from ml_app import views



router=routers.DefaultRouter()
router.register('api/healthrecord',HealthRecordViewSet,'health')

urlpatterns = [
    path('', include(router.urls)),
    path('users/',views.ListUsers.as_view()),
    path('health_record/',views.HealthRecord.as_view()),
    path('token/auth/', views.ObtainAuthToken.as_view()),
    path('api-auth/', include('rest_framework.urls', namespace='rest_framework'))
]
