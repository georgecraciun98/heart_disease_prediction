# Generated by Django 3.0.9 on 2021-06-02 19:03

import datetime
import django.core.validators
from django.db import migrations, models
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('ml_app', '0029_auto_20210601_2124'),
    ]

    operations = [
        migrations.AlterField(
            model_name='appointments',
            name='time',
            field=models.DateTimeField(validators=[django.core.validators.MinValueValidator(datetime.datetime(2021, 6, 2, 19, 3, 27, 640700, tzinfo=utc)), django.core.validators.MaxValueValidator(datetime.datetime(2022, 6, 2, 19, 3, 27, 640700, tzinfo=utc))]),
        ),
    ]