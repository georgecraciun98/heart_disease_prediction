from ml_app.Serializers.health_serializer import HealthRecordSerializer
import os
from django.contrib.auth.models import User

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "heart_prediction.settings")
serializer=HealthRecordSerializer(User.objects.all()[0])

serializer.data