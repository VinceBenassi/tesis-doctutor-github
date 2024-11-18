from .modelos.chatbot import predecir_clase_frase, obtener_respuesta, intentos
from .modelos.traductor import escuchar_microfono
from .modelos.clasificadorPDF import obtener_datos_materias
from .modelos.generador_cuestionarios import GeneradorCuestionarios
from .models import MateriaCuestionario, TextoCuestionario, FormularioMateriaCuestionario, FormularioTextoCuestionario, PerfilUsuario, ResultadoCuestionario, CambiarPerfil, FotoPerfil
from django.db.models import Avg
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, Http404
from django.contrib.auth import logout
from django.core.files.storage import default_storage, FileSystemStorage
from django.core.files.base import ContentFile
from django.conf import settings
from django.views import View
import subprocess
import uuid
import json
import os

# Variables globales para mantener el historial
historial_chatbot = []

# Carga los cuestionarios una sola vez al iniciar la aplicación
generador = GeneradorCuestionarios()
cuestionarios = generador.procesar_json_entrada('tutorApp/static/json/quiz.json')
generador.guardar_cuestionarios(cuestionarios, 'tutorApp/static/json/cuestionarios.json')
# Fin carga de los cuestionarios

# Obtén la ruta de la carpeta de PDFs
PDF_DIR = os.path.join(settings.BASE_DIR, 'tutorApp', 'static', 'pdf')

# Vista previa urls
class ScreenshotView(View):
    def get(self, request):
        url = request.GET.get('url')
        if not url:
            return JsonResponse({'error': 'URL no proporcionada'}, status=400)

        screenshot_filename = f'{uuid.uuid4()}.png'
        screenshot_path = os.path.join(settings.MEDIA_ROOT, 'screenshots', screenshot_filename)
        
        # Ruta a screenshot.js en utils
        screenshot_script_path = os.path.join(settings.BASE_DIR, 'utils', 'screenshot.js')
        
        # Ejecutar el script con Node.js
        try:
            subprocess.run(
                ['node', screenshot_script_path, url, screenshot_path],
                check=True
            )
        except subprocess.CalledProcessError:
            return JsonResponse({'error': 'No se pudo generar la captura de pantalla'}, status=500)
        
        screenshot_url = f'{settings.MEDIA_URL}screenshots/{screenshot_filename}'
        return JsonResponse({'screenshot_url': screenshot_url})
# Fin vista previa urls


# Vistas al frontend del proyecto
def landing_page(request):
    return render(request, 'landing.html')

# Cerrar Sesión
def salir(request):
    logout(request)
    return redirect('landing')



# Perfil
@login_required
def perfil(request):
    materias = cargar_materias(request)
    usuario = request.user
    perfil, creado = PerfilUsuario.objects.get_or_create(usuario=usuario)

    if request.method == 'POST':
        form = CambiarPerfil(request.POST, request.FILES, instance=perfil, usuario=request.user)
        
        if form.is_valid():
            if form.cleaned_data['nueva_foto']:
                nueva_foto = FotoPerfil.objects.create(imagen=form.cleaned_data['nueva_foto'])
                perfil.fotos_perfil.add(nueva_foto)
                perfil.foto_perfil_actual = nueva_foto
            elif form.cleaned_data['foto_perfil']:
                perfil.foto_perfil_actual = form.cleaned_data['foto_perfil']
            perfil.save()
            form.save()
            messages.success(request, 'Perfil actualizado con éxito.')
            return redirect('perfil')
    else:
        form = CambiarPerfil(instance=perfil, usuario=request.user)

    resultados = ResultadoCuestionario.objects.filter(usuario=usuario)
    promedio_general = resultados.aggregate(Avg('puntaje'))['puntaje__avg']
    
    if promedio_general:
        # Convertir el promedio a escala 1 a 7
        promedio_general_7 = (promedio_general / 100) * 6 + 1
        perfil.promedio_general = round(promedio_general_7, 1)
        perfil.save()

    categorias = resultados.values_list('categoria', flat=True).distinct()
    promedios_por_categoria = {}
    
    for categoria in categorias:
        promedio = resultados.filter(categoria=categoria).aggregate(Avg('puntaje'))['puntaje__avg']
        promedio_7 = (promedio / 100) * 6 + 1  # Convertir el promedio a escala 1-7
        promedios_por_categoria[categoria] = round(promedio_7, 1)

    
    resultados_recientes = resultados.order_by('-fecha')[:5]
    resultados_procesados = []
    
    for resultado in resultados_recientes:
        resultados_procesados.append({
            'categoria': resultado.categoria,
            'puntaje': convertir_puntaje(resultado.puntaje),
            'fecha': resultado.fecha
        })

    # Cálculo del progreso
    progreso = round((perfil.promedio_general - perfil.nivel_inicial) / 6 * 100, 2)  # Asumiendo que 7 es la nota máxima

    barra_materias = {
        'nombre_usuario': perfil.nombre_completo(),
        'materias': materias,
        'perfil': perfil,
        'promedios_por_categoria': promedios_por_categoria,
        'resultados_recientes': resultados_procesados,
        'form': form,
        'progreso': progreso,
        'cuestionarios_completados': perfil.cuestionarios_completados,
        'promedio_general': perfil.promedio_general,
        'foto_perfil_url': perfil.obtener_foto_perfil_url(),
    }
    return render(request, 'perfil/perfil.html', barra_materias)


def convertir_puntaje(puntaje):
    # Convertir el puntaje a escala de evaluación de 1-7
    puntaje_7 = (puntaje / 100) * 6 + 1
    return round(puntaje_7, 1)


@login_required
def eliminar_foto(request, foto_id):
    if request.method == 'POST':
        foto = get_object_or_404(FotoPerfil, id=foto_id)
        perfil = request.user.perfilusuario
        
        if foto in perfil.fotos_perfil.all():
            # Eliminar la imagen del sistema de archivos
            if foto.imagen:
                if os.path.isfile(foto.imagen.path):
                    os.remove(foto.imagen.path)
            
            # Actualizar el perfil si esta era la foto actual
            if perfil.foto_perfil_actual == foto:
                perfil.foto_perfil_actual = None
                perfil.save()
            
            # Eliminar la relación y el objeto de la base de datos
            perfil.fotos_perfil.remove(foto)
            foto.delete()
            
            messages.success(request, 'Foto eliminada con éxito.')
        else:
            messages.error(request, 'No tienes permiso para eliminar esta foto.')
    
    return redirect('perfil')
# Fin perfil







# Barra Lateral
def cargar_materias(request):
    categorias = obtener_datos_materias(PDF_DIR)
    print(f"Materias data cargada: {categorias}")  # Depuración
    return categorias


@login_required
def inicio(request):
    materias = cargar_materias(request)
    perfil, creado = PerfilUsuario.objects.get_or_create(usuario=request.user)

    barra_materias = {
        'nombre_usuario': perfil.nombre_completo(),
        'materias': materias,
        'foto_perfil_url': perfil.obtener_foto_perfil_url(),
    }

    return render(request, 'pages/inicio.html', barra_materias)



@login_required
@csrf_exempt
def traductor(request):
    materias = cargar_materias(request)
    perfil, creado = PerfilUsuario.objects.get_or_create(usuario=request.user)
    
    barra_materias = {
        'nombre_usuario': perfil.nombre_completo(),
        'materias': materias,
        'foto_perfil_url': perfil.obtener_foto_perfil_url(),
    }

    if request.method == 'POST':
        try:
            audio_file = request.FILES['audio']
            print(f"Audio file recibido: {audio_file.name}")
            
            # Guardar el archivo en MEDIA_ROOT
            file_path = default_storage.save(os.path.join('audio', audio_file.name), ContentFile(audio_file.read()))
            full_path = os.path.join(settings.MEDIA_ROOT, file_path)
            print(f"Archivo guardado en: {full_path}")
            
            # Procesar el audio
            idioma_detectado, mensaje, texto = escuchar_microfono(full_path)
            print(f"Resultados: idioma={idioma_detectado}, mensaje={mensaje}, traducción={texto}")
            
            # Eliminar el archivo después de procesarlo
            default_storage.delete(file_path)
            
            return JsonResponse({
                'idioma_detectado': idioma_detectado,
                'mensaje': mensaje,
                'texto': texto,
                'foto_perfil_url': perfil.obtener_foto_perfil_url(),
            })
        except Exception as e:
            print(f"Error en la vista traductor: {e}")
            return JsonResponse({'error': str(e)}, status=400)
        
    return render(request, 'pages/traductor.html', barra_materias)



@login_required
def chatbot(request):
    global historial_chatbot
    materias = cargar_materias(request)
    perfil, creado = PerfilUsuario.objects.get_or_create(usuario=request.user)

    if request.method == 'POST':
        mensaje = request.POST.get('consulta', '')

        if mensaje:
            # Procesar el mensaje del usuario
            clase = predecir_clase_frase(mensaje)
            respuesta = obtener_respuesta(clase, intentos)

            # Actualizar el historial
            historial_chatbot.append(('Usuario', mensaje))
            historial_chatbot.append(('Tutor', respuesta))
    else:
        # Resetear el historial en una solicitud GET
        historial_chatbot = []

    return render(request, 'pages/chatbot.html', {
        'historial_chatbot': historial_chatbot, 
        'materias': materias,
        'nombre_usuario': perfil.nombre_completo(),
        'foto_perfil_url': perfil.obtener_foto_perfil_url(),
    })



# Documentación (PDF's)
@login_required
def admin_dashboard(request):
    materias = cargar_materias(request)
    perfil, creado = PerfilUsuario.objects.get_or_create(usuario=request.user)

    barra_materias = {
        'nombre_usuario': perfil.nombre_completo(),
        'foto_perfil_url': perfil.obtener_foto_perfil_url(),
        'materias': materias,
    }

    return render(request, 'plantillas/admin_dashboard.html', barra_materias)



@login_required
def documentacion(request, materia):
    materias = cargar_materias(request)
    pdfs = materias.get(materia, [])
    perfil, creado = PerfilUsuario.objects.get_or_create(usuario=request.user)

    context = {
        'nombre_usuario': perfil.nombre_completo(),
        'materia': materia,
        'pdfs': pdfs,
        'materias': materias,
        'foto_perfil_url': perfil.obtener_foto_perfil_url(),
    }
    return render(request, 'pages/documentacion.html', context)
# Fin documentación (PDF's)



@login_required
def evaluaciones(request):
    categorias = obtener_materias_disponibles()
    materias = cargar_materias(request)
    perfil, creado = PerfilUsuario.objects.get_or_create(usuario=request.user)

    return render(request, 'pages/evaluaciones.html', {
        'nombre_usuario': perfil.nombre_completo(),
        'categorias': categorias,
        'materias': materias,
        'foto_perfil_url': perfil.obtener_foto_perfil_url(),
    })
# Fin barra lateral







# Evaluaciones Tutor
@login_required
def admin_evaluaciones(request):
    materias = cargar_materias(request)
    formulario_materia = FormularioMateriaCuestionario()
    formulario_texto = FormularioTextoCuestionario()
    perfil, creado = PerfilUsuario.objects.get_or_create(usuario=request.user)

    if request.method == 'POST':
        if 'agregar_materia' in request.POST:
            formulario_materia = FormularioMateriaCuestionario(request.POST)
            
            if formulario_materia.is_valid():
                formulario_materia.save()
                messages.success(request, "Nueva categoría añadida exitosamente.")
                
                return redirect('admin_evaluaciones')
        
        elif 'eliminar_materia' in request.POST:
            materia_id = request.POST.get('materia_id')
            
            try:
                categoria = MateriaCuestionario.objects.get(id=materia_id)
                categoria.delete()
                actualizar_quiz_json()
                messages.success(request, "Materia eliminada exitosamente.")
            
            except MateriaCuestionario.DoesNotExist:
                messages.error(request, "La categoría no existe.")
            
            return redirect('admin_evaluaciones')
        
        elif 'eliminar_texto' in request.POST:
            texto_id = request.POST.get('texto_id')
            
            try:
                texto = TextoCuestionario.objects.get(id=texto_id)
                texto.delete()
                actualizar_quiz_json()
                messages.success(request, "Texto eliminado exitosamente.")
            
            except TextoCuestionario.DoesNotExist:
                messages.error(request, "El texto no existe.")
            
            return redirect('admin_evaluaciones')
        
        elif 'editar_texto' in request.POST:
            texto_id = request.POST.get('editar_texto_id')
            materia_id = request.POST.get('materia')
            nuevo_texto = request.POST.get('texto')
            nueva_fuente = request.POST.get('fuente')
            
            try:
                texto = TextoCuestionario.objects.get(id=texto_id)
                texto.materia_id = materia_id
                texto.texto = nuevo_texto
                texto.fuente = nueva_fuente
                texto.save()
                actualizar_quiz_json()
                messages.success(request, "Texto actualizado exitosamente.")
            
            except TextoCuestionario.DoesNotExist:
                messages.error(request, "El texto no existe.")
            
            return redirect('admin_evaluaciones')
        
        elif 'agregar_texto' in request.POST:
            formulario_texto = FormularioTextoCuestionario(request.POST)
            
            if formulario_texto.is_valid():
                formulario_texto.save()
                actualizar_quiz_json()
                messages.success(request, "Texto añadido exitosamente.")
                
                return redirect('admin_evaluaciones')
        
        elif 'subir_pdf' in request.POST:
            if request.FILES['pdf_file']:
                pdf_file = request.FILES['pdf_file']
                fs = FileSystemStorage(location=PDF_DIR)
                filename = fs.save(pdf_file.name, pdf_file)
                messages.success(request, f"PDF '{filename}' subido exitosamente.")
                
                return redirect('admin_evaluaciones')

        elif 'eliminar_pdf' in request.POST:
            pdf_name = request.POST.get('pdf_name')
            pdf_path = os.path.join(PDF_DIR, pdf_name)
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
                messages.success(request, f"PDF '{pdf_name}' eliminado exitosamente.")
            else:
                messages.error(request, f"El PDF '{pdf_name}' no existe.")
            
            return redirect('admin_evaluaciones')

    categories = MateriaCuestionario.objects.all()
    textos = TextoCuestionario.objects.all().select_related('materia')
    pdfs = [f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')]

    return render(request, 'evaluaciones/admin_evaluaciones.html', {
        'nombre_usuario': perfil.nombre_completo(),
        'formulario_materia': formulario_materia,
        'formulario_texto': formulario_texto,
        'categories': categories,
        'textos': textos,
        'materias': materias,
        'pdfs': pdfs,
        'foto_perfil_url': perfil.obtener_foto_perfil_url(),
    })



def actualizar_quiz_json():
    data = {"quiz": []}
    textos = TextoCuestionario.objects.all().select_related('materia')
    
    for texto in textos:
        data['quiz'].append({
            "texto": texto.texto,
            "fuente": texto.fuente,
            "materia": texto.materia.nombre
        })

    with open('tutorApp/static/json/quiz.json', 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)



# Sección de vista y resolución de los cuestionarios
def organizar_cuestionarios():
    """Organiza los cuestionarios por materia"""
    cuestionarios_por_materia = {}
    for cuestionario in cuestionarios:
        materia = cuestionario['materia']
        
        if materia not in cuestionarios_por_materia:
            cuestionarios_por_materia[materia] = []
        
        cuestionarios_por_materia[materia].append(cuestionario)
    return cuestionarios_por_materia


def obtener_materias_disponibles():
    """Obtiene las materias que tienen cuestionarios disponibles"""
    cuestionarios = organizar_cuestionarios()
    
    return {materia: quizzes for materia, quizzes in cuestionarios.items() if quizzes}


@login_required
def categoria(request, category):
    categorias = obtener_materias_disponibles()
    materias = cargar_materias(request)
    perfil, creado = PerfilUsuario.objects.get_or_create(usuario=request.user)

    if category not in categorias:
        raise Http404("Categoría no encontrada")
    
    return render(request, 'evaluaciones/materia_quizzes.html', {
        'nombre_usuario': perfil.nombre_completo(),
        'category': category,
        'quizzes': categorias[category],
        'materias': materias,
        'foto_perfil_url': perfil.obtener_foto_perfil_url(),
    })

@login_required
def quiz(request, category, quiz_index):
    cuestionarios = organizar_cuestionarios()
    
    if category not in cuestionarios or quiz_index >= len(cuestionarios[category]):
        raise Http404("Cuestionario no encontrado")
    
    quiz = cuestionarios[category][quiz_index]
    materias = cargar_materias(request)
    perfil, creado = PerfilUsuario.objects.get_or_create(usuario=request.user)

    if request.method == 'POST':
        puntaje = float(request.POST.get('puntaje', 0))
        
        ResultadoCuestionario.objects.create(
            usuario=request.user,
            categoria=category,
            puntaje=puntaje
        )
        
        perfil.cuestionarios_completados += 1
        perfil.save()
        
        messages.success(request, "¡Cuestionario completado! Tu progreso ha sido guardado.")
        return redirect('perfil')

    # Asegurarse de que todas las preguntas tengan explicación
    for pregunta in quiz['preguntas']:
        if 'explicacion' not in pregunta:
            pregunta['explicacion'] = "No se proporcionó explicación para esta pregunta."

    quiz_json = json.dumps(quiz, ensure_ascii=False)
    
    return render(request, 'evaluaciones/quiz.html', {
        'nombre_usuario': perfil.nombre_completo(),
        'quiz': quiz_json,
        'materias': materias,
        'foto_perfil_url': perfil.obtener_foto_perfil_url(),
    })
# Fin evaluaciones Tutor