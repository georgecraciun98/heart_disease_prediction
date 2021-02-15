from ml_app.models import HealthRecord
from rest_framework import viewsets,permissions,authentication
from ml_app.Serializers.health_record_serializer import HealthRecordSerializer

#viewsuts allows us to quicker cruds

class HealthRecordViewSet(viewsets.ModelViewSet):
    queryset = HealthRecord.objects.all()
    authentication_classes = [authentication.TokenAuthentication]
    permission_classes = [permissions.IsAuthenticated]
    serializer_class = HealthRecordSerializer