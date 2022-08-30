# Generated by Django 3.0.7 on 2020-10-20 10:10

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Roman_Converter_Model',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('inputEnter', models.CharField(max_length=250)),
                ('detailStep', models.TextField()),
                ('finalAnswer', models.CharField(max_length=300)),
                ('slug', models.CharField(max_length=300)),
                ('solutionTitle', models.CharField(max_length=250)),
                ('generateDate', models.DateField()),
            ],
        ),
    ]
