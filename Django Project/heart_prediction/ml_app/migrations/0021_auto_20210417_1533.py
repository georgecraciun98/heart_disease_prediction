# Generated by Django 3.0.9 on 2021-04-17 12:33

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ml_app', '0020_auto_20210329_1012'),
    ]

    operations = [
        migrations.AlterField(
            model_name='monitoreddata',
            name='activity_measure_type',
            field=models.CharField(blank=True, max_length=300),
        ),
    ]
