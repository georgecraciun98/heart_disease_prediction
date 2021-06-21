# Generated by Django 3.0.9 on 2021-06-15 17:01

import datetime
import django.core.validators
from django.db import migrations, models
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('ml_app', '0043_auto_20210615_2000'),
    ]

    operations = [
        migrations.AlterField(
            model_name='appointments',
            name='time',
            field=models.DateTimeField(validators=[django.core.validators.MinValueValidator(datetime.datetime(2021, 6, 15, 17, 1, 29, 42178, tzinfo=utc)), django.core.validators.MaxValueValidator(datetime.datetime(2022, 6, 15, 17, 1, 29, 42178, tzinfo=utc))]),
        ),
    ]