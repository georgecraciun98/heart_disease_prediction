# Generated by Django 3.0.9 on 2021-04-18 19:13

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('ml_app', '0024_auto_20210418_1551'),
    ]

    operations = [
        migrations.RenameField(
            model_name='monitoreddata',
            old_name='activity_measure_type',
            new_name='activity_source',
        ),
    ]
