from rest_framework import routers
from .api import HealthRecordViewSet

router=routers.DefaultRouter()
router.register('api/healthrecord',HealthRecordViewSet,'health')

urlpatterns=router.urls
