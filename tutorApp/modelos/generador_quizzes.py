# Generador de Quizzes
# por Franco Benassi
import json
import random
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import spacy
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')
nlp = spacy.load("es_core_news_sm")

tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
model = AutoModelForQuestionAnswering.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
qa_model = pipeline("question-answering", model=model, tokenizer=tokenizer)

def cargar_datos_json(ruta_archivo):
    with open(ruta_archivo, 'r', encoding='utf-8') as archivo:
        return json.load(archivo)

def extraer_oraciones_clave(texto, n=10):
    oraciones = sent_tokenize(texto)
    return random.sample(oraciones, min(n, len(oraciones)))

def generar_pregunta(oracion):
    doc = nlp(oracion)
    sujeto = next((token for token in doc if token.dep_ == "nsubj"), None)
    verbo = next((token for token in doc if token.pos_ == "VERB"), None)
    
    if sujeto and verbo:
        pregunta = f"¿Qué {verbo.lemma_} {sujeto.text}?"
    elif verbo:
        pregunta = f"¿Qué ocurre con {verbo.lemma_}?"
    else:
        pregunta = f"¿Qué se puede decir sobre {doc[0].text}?"
    
    return pregunta.capitalize()

def generar_distractores(texto, respuesta, n=3):
    oraciones = sent_tokenize(texto)
    candidatos = [sent for sent in oraciones if sent != respuesta]
    return random.sample(candidatos, min(n, len(candidatos)))

def generar_pregunta_alternativas(texto, oracion):
    pregunta = generar_pregunta(oracion)
    respuesta_correcta = oracion
    distractores = generar_distractores(texto, respuesta_correcta, 3)
    opciones = [respuesta_correcta] + distractores
    random.shuffle(opciones)
    
    return {
        'tipo': 'alternativas',
        'pregunta': pregunta,
        'opciones': opciones,
        'respuesta_correcta': respuesta_correcta,
        'explicacion': f"La respuesta correcta es: {respuesta_correcta}"
    }

def generar_pregunta_verdadero_falso(texto, oracion):
    es_verdadero = random.choice([True, False])
    
    if es_verdadero:
        pregunta = oracion
        respuesta = True
        explicacion = "Esta afirmación es correcta según el texto."
    else:
        doc = nlp(oracion)
        palabras = list(doc)
        for i, token in enumerate(doc):
            if token.pos_ in ["NOUN", "VERB", "ADJ", "PROPN"]:
                palabras_similares = [t for t in nlp(texto) if t.pos_ == token.pos_ and t.text.lower() != token.text.lower()]
                if palabras_similares:
                    palabras[i] = random.choice(palabras_similares)
                    break
        
        pregunta = " ".join([token.text for token in palabras])
        respuesta = False
        explicacion = f"Esta afirmación es incorrecta. La versión correcta es: {oracion}"

    return {
        'tipo': 'verdadero_falso',
        'pregunta': pregunta,
        'respuesta_correcta': respuesta,
        'explicacion': explicacion
    }

def generar_pregunta_abierta(texto, oracion):
    pregunta = generar_pregunta(oracion)
    respuesta = qa_model(question=pregunta, context=texto)['answer']
    
    return {
        'tipo': 'abierta',
        'pregunta': pregunta,
        'respuesta_correcta': respuesta,
        'explicacion': f"Una posible respuesta basada en el texto es: {respuesta}"
    }

def generar_cuestionario(texto, num_preguntas=10):
    oraciones_clave = extraer_oraciones_clave(texto)
    preguntas = []
    
    while len(preguntas) < num_preguntas:
        tipo_pregunta = random.choice(['alternativas', 'verdadero_falso', 'abierta'])
        oracion = random.choice(oraciones_clave)
        
        if tipo_pregunta == 'alternativas':
            pregunta = generar_pregunta_alternativas(texto, oracion)
        elif tipo_pregunta == 'verdadero_falso':
            pregunta = generar_pregunta_verdadero_falso(texto, oracion)
        else:
            pregunta = generar_pregunta_abierta(texto, oracion)
        
        if pregunta and pregunta not in preguntas:
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