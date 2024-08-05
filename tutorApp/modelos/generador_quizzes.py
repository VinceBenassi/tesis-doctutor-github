# Generador de Quizzes
# por Franco Benassi
import json
import random
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Inicializar el modelo BERT en español
modelo_nombre = "dccuchile/bert-base-spanish-wwm-uncased"
tokenizer = AutoTokenizer.from_pretrained(modelo_nombre)
model = AutoModelForQuestionAnswering.from_pretrained(modelo_nombre)

# Inicializar el pipeline de generación de preguntas
nlp = pipeline("question-answering", model=model, tokenizer=tokenizer)

def cargar_datos_json(ruta_archivo):
    with open(ruta_archivo, 'r', encoding='utf-8') as archivo:
        return json.load(archivo)

def extraer_informacion_clave(texto):
    oraciones = sent_tokenize(texto)
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('spanish'))
    tfidf_matrix = vectorizer.fit_transform(oraciones)
    
    feature_names = vectorizer.get_feature_names_out()
    dense = tfidf_matrix.todense()
    denselist = dense.tolist()
    
    scores = [(oracion, score) for oracion, score in zip(oraciones, denselist)]
    sorted_scores = sorted(scores, key=lambda x: sum(x[1]), reverse=True)
    
    return [oracion for oracion, _ in sorted_scores[:5]]

def generar_pregunta_alternativas(texto):
    informacion_clave = extraer_informacion_clave(texto)
    oracion = random.choice(informacion_clave)
    
    palabras = word_tokenize(oracion)
    palabras_pos = nltk.pos_tag(palabras)
    
    sustantivos = [palabra for palabra, pos in palabras_pos if pos.startswith('N')]
    
    if sustantivos:
        palabra_clave = random.choice(sustantivos)
        respuesta = nlp(question=f"¿Qué es {palabra_clave}?", context=texto)
        pregunta = f"¿Qué es {palabra_clave}?"
        
        opciones = [respuesta['answer']]
        otras_opciones = [p for p in sustantivos if p != palabra_clave]
        opciones += random.sample(otras_opciones, min(3, len(otras_opciones)))
        
        while len(opciones) < 4:
            opciones.append(random.choice(palabras))
        
        random.shuffle(opciones)
        
        return {
            'tipo': 'alternativas',
            'pregunta': pregunta,
            'opciones': opciones,
            'respuesta_correcta': respuesta['answer'],
            'explicacion': f"La respuesta correcta es '{respuesta['answer']}' porque {oracion}"
        }
    else:
        return generar_pregunta_abierta(texto)

def generar_pregunta_verdadero_falso(texto):
    informacion_clave = extraer_informacion_clave(texto)
    oracion = random.choice(informacion_clave)
    
    es_verdadero = random.choice([True, False])
    
    if es_verdadero:
        pregunta = f"¿Es verdadera la siguiente afirmación?: {oracion}"
        respuesta = True
        explicacion = f"La afirmación es verdadera. {oracion}"
    else:
        palabras = word_tokenize(oracion)
        palabras_pos = nltk.pos_tag(palabras)
        
        sustantivos = [i for i, (palabra, pos) in enumerate(palabras_pos) if pos.startswith('N')]
        
        if sustantivos:
            indice_aleatorio = random.choice(sustantivos)
            palabra_original = palabras[indice_aleatorio]
            palabras[indice_aleatorio] = random.choice([p for p, pos in palabras_pos if pos.startswith('N') and p != palabra_original])
            oracion_modificada = ' '.join(palabras)
            pregunta = f"¿Es verdadera la siguiente afirmación?: {oracion_modificada}"
            respuesta = False
            explicacion = f"La afirmación es falsa. La versión correcta es: {oracion}"
        else:
            return generar_pregunta_verdadero_falso(texto)

    return {
        'tipo': 'verdadero_falso',
        'pregunta': pregunta,
        'respuesta_correcta': respuesta,
        'explicacion': explicacion
    }

def generar_pregunta_abierta(texto):
    informacion_clave = extraer_informacion_clave(texto)
    oracion = random.choice(informacion_clave)
    
    palabras = word_tokenize(oracion)
    palabras_pos = nltk.pos_tag(palabras)
    
    sustantivos = [palabra for palabra, pos in palabras_pos if pos.startswith('N')]
    
    if sustantivos:
        palabra_clave = random.choice(sustantivos)
        respuesta = nlp(question=f"¿Qué es {palabra_clave}?", context=texto)
        pregunta = f"¿Qué es {palabra_clave}?"
        
        return {
            'tipo': 'abierta',
            'pregunta': pregunta,
            'respuesta_correcta': respuesta['answer'],
            'explicacion': f"La respuesta está relacionada con: {oracion}"
        }
    else:
        return generar_pregunta_abierta(texto)

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