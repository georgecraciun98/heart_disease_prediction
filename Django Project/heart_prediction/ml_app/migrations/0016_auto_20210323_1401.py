# Generated by Django 3.0.9 on 2021-03-23 12:01

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('ml_app', '0015_auto_20210317_1507'),
    ]

    operations = [
        migrations.AlterField(
            model_name='healthrecordmodel',
            name='sex',
            field=models.IntegerField(choices=[(0, 'Type 0'), (1, 'Type 1')], default=1),
        ),
        migrations.AlterField(
            model_name='predicteddata',
            name='model',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.DO_NOTHING, related_name='models', to='ml_app.ModelConfiguration'),
        ),
        migrations.AlterField(
            model_name='userdetailmodel',
            name='sex',
            field=models.IntegerField(choices=[(0, 'Type 0'), (1, 'Type 1')], default=1),
        ),
    ]
