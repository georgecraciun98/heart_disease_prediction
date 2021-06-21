# Generated by Django 3.0.9 on 2021-06-20 14:02

import datetime
import django.core.validators
from django.db import migrations, models
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('ml_app', '0045_auto_20210619_1254'),
    ]

    operations = [
        migrations.AddField(
            model_name='modelconfiguration',
            name='active',
            field=models.BooleanField(default=False, null=True),
        ),
        migrations.AlterField(
            model_name='appointments',
            name='time',
            field=models.DateTimeField(validators=[django.core.validators.MinValueValidator(datetime.datetime(2021, 6, 20, 14, 2, 33, 669749, tzinfo=utc)), django.core.validators.MaxValueValidator(datetime.datetime(2022, 6, 20, 14, 2, 33, 669749, tzinfo=utc))]),
        ),
    ]