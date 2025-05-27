from django.urls import path
from django.conf.urls.static import static
from django.conf import settings
from . import views

urlpatterns = [
    path('', views.landing_page, name='landing'),
    path('salir/', views.salir, name='salir'),
    path('inicio', views.inicio, name='inicio'),

    # Perfil
    path('perfil', views.perfil, name='perfil'),
    path('eliminar_foto/<int:foto_id>/', views.eliminar_foto, name='eliminar_foto'),
    # Fin perfil

    # Barra lateral
    path('traductor/', views.traductor, name='traductor'),
    path('chatbot', views.chatbot, name='chatbot'),
    path('admin_dashboard', views.admin_dashboard, name='admin_dashboard'),
    path('documentacion/<str:materia>/', views.documentacion, name='documentacion'),
    path('evaluaciones', views.evaluaciones, name='evaluaciones'),
    # Fin barra lateral
    
    # Evaluaciones Tutor
    path('admin_evaluaciones/', views.admin_evaluaciones, name='admin_evaluaciones'),
    path('categoria/<str:category>/', views.categoria, name='categoria'),
    path('quiz/<str:category>/<int:quiz_index>/', views.quiz, name='quiz'),
    # Fin evaluaciones Tutor

    # Vista previa
    path('api/capture-screenshot/', views.ScreenshotView.as_view(), name='capture_screenshot'),
    # Fin vista previa
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
