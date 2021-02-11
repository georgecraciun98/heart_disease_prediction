# Generated by Django 3.0.9 on 2021-02-10 17:04

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ml_app', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='healthrecord',
            name='cp',
            field=models.IntegerField(choices=[(0, 'Pain 0'), (1, 'Pain 1'), (2, 'Pain 2'), (3, 'Pain 3')], default=0),
        ),
    ]
