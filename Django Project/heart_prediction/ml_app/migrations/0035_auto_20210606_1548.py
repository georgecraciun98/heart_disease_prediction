# Generated by Django 3.0.9 on 2021-06-06 12:48

import datetime
import django.core.validators
from django.db import migrations, models
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('ml_app', '0034_auto_20210606_1538'),
    ]

    operations = [
        migrations.AlterField(
            model_name='appointments',
            name='time',
            field=models.DateTimeField(validators=[django.core.validators.MinValueValidator(datetime.datetime(2021, 6, 6, 12, 48, 16, 505042, tzinfo=utc)), django.core.validators.MaxValueValidator(datetime.datetime(2022, 6, 6, 12, 48, 16, 505042, tzinfo=utc))]),
        ),
    ]