# Generated by Django 3.2.12 on 2022-02-16 23:29

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='cat',
            name='flag',
            field=models.IntegerField(choices=[(0, '错误'), (1, '正确')], default=1),
        ),
    ]
