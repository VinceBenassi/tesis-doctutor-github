# Generated by Django 4.2.5 on 2024-07-25 20:15

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('tutorApp', '0007_perfilusuario_fecha_inicio_and_more'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='perfilusuario',
            name='fecha_inicio',
        ),
        migrations.RemoveField(
            model_name='perfilusuario',
            name='nivel_inicial',
        ),
    ]
