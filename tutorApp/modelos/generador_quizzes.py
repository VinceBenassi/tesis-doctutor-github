# Generador de Quizzes
# por Franco Benassi
import json
import random
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')

# Inicializar el modelo BERT en español
tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
model = AutoModelForQuestionAnswering.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")

# Inicializar el pipeline de generación de preguntas
nlp = pipeline("question-answering", model=model, tokenizer=tokenizer)

def cargar_datos_json(ruta_archivo):
    with open(ruta_archivo, 'r', encoding='utf-8') as archivo:
        return json.load(archivo)

def generar_pregunta_alternativas(texto):
    oraciones = sent_tokenize(texto)
    oracion = random.choice(oraciones)
    
    respuesta = nlp(question="¿Qué información importante contiene esta oración?", context=oracion)
    pregunta = respuesta['answer']
    
    palabras = word_tokenize(oracion)
    palabras_relevantes = [palabra for palabra in palabras if palabra.lower() not in stopwords.words('spanish')]
    
    opciones = [pregunta] + random.sample(palabras_relevantes, 3)
    random.shuffle(opciones)
    
    return {
        'tipo': 'alternativas',
        'pregunta': f"¿Cuál de las siguientes opciones es correcta según el texto?",
        'opciones': opciones,
        'respuesta_correcta': pregunta,
        'explicacion': f"La respuesta correcta es '{pregunta}' porque {oracion}"
    }

def generar_pregunta_verdadero_falso(texto):
    oraciones = sent_tokenize(texto)
    oracion = random.choice(oraciones)
    
    es_verdadero = random.choice([True, False])
    
    if es_verdadero:
        pregunta = f"¿Es verdadera la siguiente afirmación?: {oracion}"
        respuesta = True
        explicacion = f"La afirmación es verdadera. {oracion}"
    else:
        palabras = word_tokenize(oracion)
        indice_aleatorio = random.randint(0, len(palabras) - 1)
        palabra_original = palabras[indice_aleatorio]
        palabras[indice_aleatorio] = "NO " + palabra_original
        oracion_modificada = " ".join(palabras)
        pregunta = f"¿Es verdadera la siguiente afirmación?: {oracion_modificada}"
        respuesta = False
        explicacion = f"La afirmación es falsa. La versión correcta es: {oracion}"

    return {
        'tipo': 'verdadero_falso',
        'pregunta': pregunta,
        'respuesta_correcta': respuesta,
        'explicacion': explicacion
    }

def generar_pregunta_abierta(texto):
    oraciones = sent_tokenize(texto)
    oracion = random.choice(oraciones)
    
    respuesta = nlp(question="¿Qué información importante contiene esta oración?", context=oracion)
    pregunta = f"¿{respuesta['answer']}?"
    
    return {
        'tipo': 'abierta',
        'pregunta': pregunta,
        'respuesta_correcta': oracion,
        'explicacion': f"La respuesta está relacionada con: {oracion}"
    }

def generar_cuestionario(texto, num_preguntas=10):
    preguntas = []
    
    for _ in range(num_preguntas):
        tipo_pregunta = random.choice(['alternativas', 'verdadero_falso', 'abierta'])
        
        if tipo_pregunta == 'alternativas':
            pregunta = generar_pregunta_alternativas(texto)
        elif tipo_pregunta == 'verdadero_falso':
            pregunta = generar_pregunta_verdadero_falso(texto)
        else:
            pregunta = generar_pregunta_abierta(texto)
        
        preguntas.append(pregunta)
    
    return preguntas

def generar_cuestionarios_desde_json(datos_json):
    cuestionarios = {}
    
    for item in datos_json['quiz']:
        categoria = item['materia']
        texto = item['texto']
        fuente = item['fuente']
        
        if categoria not in cuestionarios:
            cuestionarios[categoria] = []
        
        cuestionario = generar_cuestionario(texto)
        cuestionarios[categoria].append({
            'fuente': fuente,
            'preguntas': cuestionario
        })
    
    return cuestionarios