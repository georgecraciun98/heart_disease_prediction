# Generated by Django 3.0.9 on 2021-04-17 17:46

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('ml_app', '0022_auto_20210417_2019'),
    ]

    operations = [
        migrations.AddField(
            model_name='monitoreddata',
            name='data_type',
            field=models.CharField(default=django.utils.timezone.now, max_length=100),
            preserve_default=False,
        ),
    ]
