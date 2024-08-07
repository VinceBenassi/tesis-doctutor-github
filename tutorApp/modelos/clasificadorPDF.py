# Clasificador de Materias PDF
# por Franco Benassi
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from collections import Counter
import re

def preprocesar_texto(texto):
    texto = re.sub(r'\.pdf$', '', texto, flags=re.IGNORECASE)
    texto = re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚñÑ\s]', ' ', texto)
    
    palabras_vacias = set(stopwords.words('spanish') + stopwords.words('english'))
    palabras_mantener = {'en', 'de', 'la', 'el', 'y', 'e', 'i', 'ii', 'iii', 'iv', 'v'}
    palabras_vacias = palabras_vacias - palabras_mantener
    
    tokens = word_tokenize(texto.lower())
    return ' '.join([palabra for palabra in tokens if palabra not in palabras_vacias or palabra in palabras_mantener])



def extraer_palabras_clave(texto, min_palabras=1, max_palabras=5):
    palabras = word_tokenize(texto.lower())
    frecuencia_palabras = Counter(palabras)
    palabras_clave = []

    for palabra, _ in frecuencia_palabras.most_common():
        if len(palabras_clave) >= max_palabras:
            break
        if len(palabra) > 1 or palabra in {'i', 'ii', 'iii', 'iv', 'v', 'en', 'i', 'y', 'e'}:
            palabras_clave.append(palabra)
    
    while len(palabras_clave) < min_palabras and palabras:
        palabra = palabras.pop(0)
        if palabra not in palabras_clave:
            palabras_clave.append(palabra)
    
    return palabras_clave

def simplificar_nombre_materia(nombre):
    palabras = nombre.split()

    if len(palabras) <= 2:
        return nombre
    
    elif 'en' in palabras[1:]:
        indice_en = palabras.index('en')
        return ' '.join(palabras[:min(indice_en + 3, len(palabras))])
    
    else:
        return palabras[0]



def es_palabra_comun(palabra):
    palabras_comunes = {'especialidades', 'estandares', 'medicas', 'medio', 'chile'}
    return palabra.lower() in palabras_comunes



def clasificar_pdfs(directorio_pdf):
    archivos_pdf = [f for f in os.listdir(directorio_pdf) if f.endswith('.pdf')]
    
    if not archivos_pdf:
        print("No se encontraron archivos PDF.")
        return {}

    nombres_preprocesados = [preprocesar_texto(pdf) for pdf in archivos_pdf]
    vectorizador = TfidfVectorizer(ngram_range=(1, 2))
    X = vectorizador.fit_transform(nombres_preprocesados)

    agrupamiento = DBSCAN(eps=0.5, min_samples=1)
    etiquetas_cluster = agrupamiento.fit_predict(X.toarray())

    materias = {}
    
    for pdf, etiqueta in zip(archivos_pdf, etiquetas_cluster):
        pdf_preprocesado = preprocesar_texto(pdf)
        palabras_clave = extraer_palabras_clave(pdf_preprocesado)
        
        if etiqueta == -1:
            nombre_materia = ' '.join(palabras_clave).capitalize()
        
        else:
            pdfs_cluster = [archivos_pdf[i] for i, l in enumerate(etiquetas_cluster) if l == etiqueta]
            texto_cluster = ' '.join([preprocesar_texto(pdf_cluster) for pdf_cluster in pdfs_cluster])
            palabras_clave_cluster = extraer_palabras_clave(texto_cluster)
            nombre_materia = ' '.join(palabras_clave_cluster).capitalize()
        
        nombre_materia = simplificar_nombre_materia(nombre_materia)
        nombre_materia = ' '.join([palabra for palabra in nombre_materia.split() if not es_palabra_comun(palabra)])
        nombre_materia = re.sub(r'\s+', ' ', nombre_materia).strip()
        
        # Buscar si ya existe una materia similar
        materia_existente = None
        
        for existente in materias:
            if existente.split()[0].lower() == nombre_materia.split()[0].lower():
                materia_existente = existente
                break
        
        if materia_existente:
            materias[materia_existente].append(pdf)
        
        else:
            materias[nombre_materia] = [pdf]

    return materias



def obtener_datos_materias(directorio_pdf):
    return clasificar_pdfs(directorio_pdf)