# Generated by Django 3.1.2 on 2020-12-14 12:15

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0003_person_plocation'),
    ]

    operations = [
        migrations.AddField(
            model_name='plocation',
            name='time',
            field=models.DateTimeField(auto_now_add=True, null=True),
        ),
    ]
