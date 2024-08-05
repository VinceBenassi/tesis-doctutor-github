from django.db import models
from django.contrib.auth.models import User
from django import forms
import os

class MateriaCuestionario(models.Model):
    nombre = models.CharField(max_length=100, unique=True)

    def __str__(self):
        return self.nombre

class TextoCuestionario(models.Model):
    materia = models.ForeignKey(MateriaCuestionario, on_delete=models.CASCADE)
    texto = models.TextField()
    fuente = models.URLField()
    
    def __str__(self):
        return f"{self.materia.nombre}: {self.texto[:50]}..."
    
class FormularioMateriaCuestionario(forms.ModelForm):
    class Meta:
        model = MateriaCuestionario
        fields = ['nombre']
        widgets = {
            'nombre': forms.TextInput(attrs={
                'class': 'form-control', 
                'placeholder': 'Ingresa el nombre de la materia',
            }),
        }

class FormularioTextoCuestionario(forms.ModelForm):
    class Meta:
        model = TextoCuestionario
        fields = ['materia', 'texto', 'fuente']
        widgets = {
            'materia': forms.Select(attrs={
                'class': 'form-control',
            }),
            'texto': forms.Textarea(attrs={
                'class': 'form-control', 
                'rows': 5,
                'placeholder': 'Ingresa el texto de la materia para generar un cuestionario',
            }),
            'fuente': forms.URLInput(attrs={
                'class': 'form-control', 
                'placeholder': 'Ingresa la URL de la fuente',
            }),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['materia'].queryset = MateriaCuestionario.objects.all()
        if not self.fields['materia'].queryset.exists():
            self.fields['materia'].widget = forms.HiddenInput()
            self.fields['materia'].required = False


# Modelo para usuarios y su inicio de sesión en Doctutor
class PerfilUsuario(models.Model):
    usuario = models.OneToOneField(User, on_delete=models.CASCADE)
    nombre = models.CharField(max_length=50, blank=True)
    apellido = models.CharField(max_length=50, blank=True)
    fotos_perfil = models.ManyToManyField('FotoPerfil', related_name='usuarios')
    foto_perfil_actual = models.ForeignKey('FotoPerfil', on_delete=models.SET_NULL, null=True, related_name='usuario_actual')
    cuestionarios_completados = models.IntegerField(default=0)
    promedio_general = models.FloatField(default=0.0)
    nivel_inicial = models.FloatField(default=1.0)

    def __str__(self):
        return f"Perfil de {self.usuario.username}"
    
    def nombre_completo(self):
        return f"{self.nombre} {self.apellido}".strip() or self.usuario.username
    
    def obtener_foto_perfil_url(self):
        if self.foto_perfil_actual:
            return self.foto_perfil_actual.imagen.url
        return '/static/img/user.png'
    


class FotoPerfil(models.Model):
    imagen = models.ImageField(upload_to='fotos_perfil/')
    fecha_subida = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return os.path.basename(self.imagen.name)

    def delete(self, *args, **kwargs):
        # Eliminar el archivo de imagen
        if self.imagen:
            if os.path.isfile(self.imagen.path):
                os.remove(self.imagen.path)
        
        # Llamar al método delete() del padre para eliminar el objeto de la base de datos
        super(FotoPerfil, self).delete(*args, **kwargs)

    

# Cambio foto perfil
class CambiarPerfil(forms.ModelForm):
    foto_perfil = forms.ModelChoiceField(
        queryset=FotoPerfil.objects.none(),
        widget=forms.RadioSelect,
        required=False
    )
    nueva_foto = forms.ImageField(required=False)

    class Meta:
        model = PerfilUsuario
        fields = ['nombre', 'apellido', 'foto_perfil']
        widgets = {
            'nombre': forms.TextInput(attrs={'class': 'form-control'}),
            'apellido': forms.TextInput(attrs={'class': 'form-control'}),
        }
    
    def __init__(self, *args, **kwargs):
        usuario = kwargs.pop('usuario', None)
        super(CambiarPerfil, self).__init__(*args, **kwargs)
        if usuario:
            perfil = PerfilUsuario.objects.get(usuario=usuario)
            self.fields['foto_perfil'].queryset = perfil.fotos_perfil.all()


class ResultadoCuestionario(models.Model):
    usuario = models.ForeignKey(User, on_delete=models.CASCADE)
    categoria = models.CharField(max_length=100)
    puntaje = models.FloatField()
    fecha = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Resultado de {self.usuario.username} en {self.categoria}"