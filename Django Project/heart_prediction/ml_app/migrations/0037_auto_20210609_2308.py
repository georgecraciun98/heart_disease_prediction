# Generated by Django 3.0.9 on 2021-06-09 20:08

import datetime
import django.core.validators
from django.db import migrations, models
import django.utils.timezone
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('ml_app', '0036_auto_20210606_1635'),
    ]

    operations = [
        migrations.AddField(
            model_name='modelconfiguration',
            name='created_date',
            field=models.DateTimeField(auto_now_add=True, default=django.utils.timezone.now),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='appointments',
            name='time',
            field=models.DateTimeField(validators=[django.core.validators.MinValueValidator(datetime.datetime(2021, 6, 9, 20, 7, 20, 1725, tzinfo=utc)), django.core.validators.MaxValueValidator(datetime.datetime(2022, 6, 9, 20, 7, 20, 1725, tzinfo=utc))]),
        ),
    ]
