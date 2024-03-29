# Generated by Django 3.0.9 on 2021-06-12 10:43

import datetime
import django.core.validators
from django.db import migrations, models
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('ml_app', '0040_auto_20210612_1219'),
    ]

    operations = [
        migrations.AlterField(
            model_name='appointments',
            name='time',
            field=models.DateTimeField(validators=[django.core.validators.MinValueValidator(datetime.datetime(2021, 6, 12, 10, 43, 36, 482619, tzinfo=utc)), django.core.validators.MaxValueValidator(datetime.datetime(2022, 6, 12, 10, 43, 36, 482619, tzinfo=utc))]),
        ),
        migrations.AlterField(
            model_name='healthrecordmodel',
            name='ca_0',
            field=models.IntegerField(default=0, null=True),
        ),
        migrations.AlterField(
            model_name='healthrecordmodel',
            name='ca_1',
            field=models.IntegerField(default=0, null=True),
        ),
        migrations.AlterField(
            model_name='healthrecordmodel',
            name='ca_2',
            field=models.IntegerField(default=0, null=True),
        ),
        migrations.AlterField(
            model_name='healthrecordmodel',
            name='ca_3',
            field=models.IntegerField(default=0, null=True),
        ),
        migrations.AlterField(
            model_name='healthrecordmodel',
            name='ca_4',
            field=models.IntegerField(default=0, null=True),
        ),
        migrations.AlterField(
            model_name='healthrecordmodel',
            name='cp_0',
            field=models.IntegerField(default=0, null=True),
        ),
        migrations.AlterField(
            model_name='healthrecordmodel',
            name='cp_1',
            field=models.IntegerField(default=0, null=True),
        ),
        migrations.AlterField(
            model_name='healthrecordmodel',
            name='cp_2',
            field=models.IntegerField(default=0, null=True),
        ),
        migrations.AlterField(
            model_name='healthrecordmodel',
            name='cp_3',
            field=models.IntegerField(default=0, null=True),
        ),
        migrations.AlterField(
            model_name='healthrecordmodel',
            name='exang_0',
            field=models.IntegerField(default=0, null=True),
        ),
        migrations.AlterField(
            model_name='healthrecordmodel',
            name='exang_1',
            field=models.IntegerField(default=0, null=True),
        ),
        migrations.AlterField(
            model_name='healthrecordmodel',
            name='fbs_0',
            field=models.IntegerField(default=0, null=True),
        ),
        migrations.AlterField(
            model_name='healthrecordmodel',
            name='fbs_1',
            field=models.IntegerField(default=0, null=True),
        ),
        migrations.AlterField(
            model_name='healthrecordmodel',
            name='restecg_0',
            field=models.IntegerField(default=0, null=True),
        ),
        migrations.AlterField(
            model_name='healthrecordmodel',
            name='restecg_1',
            field=models.IntegerField(default=0, null=True),
        ),
        migrations.AlterField(
            model_name='healthrecordmodel',
            name='restecg_2',
            field=models.IntegerField(default=0, null=True),
        ),
        migrations.AlterField(
            model_name='healthrecordmodel',
            name='sex_0',
            field=models.IntegerField(default=0, null=True),
        ),
        migrations.AlterField(
            model_name='healthrecordmodel',
            name='sex_1',
            field=models.IntegerField(default=0, null=True),
        ),
        migrations.AlterField(
            model_name='healthrecordmodel',
            name='slope_0',
            field=models.IntegerField(default=0, null=True),
        ),
        migrations.AlterField(
            model_name='healthrecordmodel',
            name='slope_1',
            field=models.IntegerField(default=0, null=True),
        ),
        migrations.AlterField(
            model_name='healthrecordmodel',
            name='slope_2',
            field=models.IntegerField(default=0, null=True),
        ),
        migrations.AlterField(
            model_name='healthrecordmodel',
            name='thal_0',
            field=models.IntegerField(default=0, null=True),
        ),
        migrations.AlterField(
            model_name='healthrecordmodel',
            name='thal_1',
            field=models.IntegerField(default=0, null=True),
        ),
        migrations.AlterField(
            model_name='healthrecordmodel',
            name='thal_2',
            field=models.IntegerField(default=0, null=True),
        ),
        migrations.AlterField(
            model_name='healthrecordmodel',
            name='thal_3',
            field=models.IntegerField(default=0, null=True),
        ),
    ]
