# Generated by Django 4.2.5 on 2024-07-26 04:00

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('tutorApp', '0010_alter_perfilusuario_foto_perfil'),
    ]

    operations = [
        migrations.AddField(
            model_name='perfilusuario',
            name='estado_actividad',
            field=models.CharField(choices=[('activo', 'Activo'), ('inactivo', 'Inactivo'), ('invisible', 'Invisible'), ('no_molestar', 'No molestar')], default='activo', max_length=20),
        ),
    ]
