from django.contrib import admin
# Register your models here.
from rest_framework.authtoken.admin import TokenAdmin

from ml_app.models import HealthRecordModel

TokenAdmin.raw_id_fields = ['user']

@admin.register(HealthRecordModel)
class PersonAdmin(admin.ModelAdmin):
    pass