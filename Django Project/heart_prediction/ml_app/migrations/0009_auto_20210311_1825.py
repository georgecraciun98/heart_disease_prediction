# Generated by Django 3.0.9 on 2021-03-11 16:25

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('ml_app', '0008_auto_20210311_1741'),
    ]

    operations = [
        migrations.RenameField(
            model_name='modelconfiguration',
            old_name='researcher_id',
            new_name='researcher',
        ),
        migrations.RenameField(
            model_name='monitoreddata',
            old_name='patient_id',
            new_name='patient',
        ),
        migrations.RenameField(
            model_name='predicteddata',
            old_name='model_id',
            new_name='model',
        ),
        migrations.RenameField(
            model_name='predicteddata',
            old_name='record_id',
            new_name='record',
        ),
    ]
