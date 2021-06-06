# Generated by Django 3.0.9 on 2021-06-06 12:03

import datetime
import django.core.validators
from django.db import migrations, models
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('ml_app', '0030_auto_20210602_2203'),
    ]

    operations = [
        migrations.AlterField(
            model_name='appointments',
            name='time',
            field=models.DateTimeField(validators=[django.core.validators.MinValueValidator(datetime.datetime(2021, 6, 6, 12, 3, 17, 748655, tzinfo=utc)), django.core.validators.MaxValueValidator(datetime.datetime(2022, 6, 6, 12, 3, 17, 748655, tzinfo=utc))]),
        ),
    ]