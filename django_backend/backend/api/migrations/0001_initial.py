# Generated by Django 3.1.2 on 2020-11-04 07:10

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='analytics',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('location', models.CharField(max_length=100)),
                ('time', models.DateField(auto_now_add=True)),
                ('peoplecount', models.IntegerField()),
                ('socialdistancing', models.IntegerField()),
                ('scenedetect', models.CharField(blank=True, max_length=60, null=True)),
                ('sceneimage', models.ImageField(blank=True, null=True, upload_to='sceneimage/')),
            ],
        ),
    ]
