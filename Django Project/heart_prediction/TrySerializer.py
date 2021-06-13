import os

from django.contrib.auth.models import User

from ml_app.serializers.health_record_serializer import HealthRecordSerializer

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "heart_prediction.settings")
serializer=HealthRecordSerializer(User.objects.all()[0])

serializer.data