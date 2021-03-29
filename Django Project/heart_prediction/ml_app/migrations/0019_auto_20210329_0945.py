# Generated by Django 3.0.9 on 2021-03-29 06:45

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('ml_app', '0018_predicteddata_created_time'),
    ]

    operations = [
        migrations.AlterField(
            model_name='predicteddata',
            name='record',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='records', to=settings.AUTH_USER_MODEL),
        ),
    ]