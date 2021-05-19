# Generated by Django 3.0.9 on 2021-05-19 19:02

import datetime
from django.conf import settings
import django.core.validators
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('ml_app', '0027_auto_20210419_1134'),
    ]

    operations = [
        migrations.CreateModel(
            name='Appointments',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('time', models.DateTimeField(validators=[django.core.validators.MinValueValidator(datetime.datetime(2021, 5, 19, 22, 2, 12, 579848)), django.core.validators.MaxValueValidator(datetime.datetime(2022, 5, 19, 22, 2, 12, 579848))])),
                ('doctor', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='appoint_doctors', to=settings.AUTH_USER_MODEL)),
                ('patient', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='appoint_patients', to='ml_app.Patient')),
            ],
        ),
    ]
