# Generated by Django 3.0.9 on 2021-02-17 11:45

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('ml_app', '0001_initial'),
    ]

    operations = [
        migrations.RenameField(
            model_name='healthrecordmodel',
            old_name='user_id',
            new_name='user',
        ),
    ]