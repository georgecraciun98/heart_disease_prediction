# Generated by Django 3.0.9 on 2021-02-17 10:37

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models

import ml_app.validators.health_record_validator


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='HealthRecordModel',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('age', models.FloatField(validators=[ml_app.validators.health_record_validator.validate_age])),
                ('sex', models.CharField(choices=[('M', 'Male'), ('F', 'Female')], default='M', max_length=2)),
                ('cp', models.IntegerField(choices=[(0, 'Pain 0'), (1, 'Pain 1'), (2, 'Pain 2'), (3, 'Pain 3')], default=0)),
                ('trestbps', models.IntegerField(validators=[ml_app.validators.health_record_validator.validate_trebtps])),
                ('chol', models.IntegerField()),
                ('fbs', models.IntegerField(choices=[(0, 'Type 0'), (1, 'Type 1')])),
                ('restecg', models.IntegerField(choices=[(0, 'Type 0'), (1, 'Type 1'), (2, 'Type 2')])),
                ('thalach', models.IntegerField(validators=[ml_app.validators.health_record_validator.validate_thalach])),
                ('exang', models.IntegerField(choices=[(0, 'Type 0'), (1, 'Type 1')])),
                ('oldpeak', models.FloatField(validators=[ml_app.validators.health_record_validator.validate_oldpeak])),
                ('slope', models.IntegerField(choices=[(0, 'Type 0'), (1, 'Type 1'), (2, 'Type 2')])),
                ('ca', models.IntegerField(choices=[(0, 'Type 0'), (1, 'Type 1')])),
                ('thal', models.IntegerField(choices=[(1, 'Type 0'), (3, 'Type 1'), (6, 'Type 2'), (7, 'Type 3')])),
                ('target', models.IntegerField(blank=True, choices=[(0, 'Type 0'), (1, 'Type 1')])),
                ('user_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='records', to=settings.AUTH_USER_MODEL)),
            ],
        ),
    ]
