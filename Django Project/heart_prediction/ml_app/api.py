from ml_app.models import HealthRecord
from rest_framework import viewsets,permissions
from ml_app.Serializers.health_record_serializer import HealthRecordSerializer

#viewsuts allows us to quicker cruds

class HealthRecordViewSet(viewsets.ModelViewSet):
    queryset = HealthRecord.objects.all()
    permission_classes = [permissions.AllowAny]
    serializer_class = HealthRecordSerializer