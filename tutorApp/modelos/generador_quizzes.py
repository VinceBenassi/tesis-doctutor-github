# Generador de Quizzes
# por Franco Benassi
import json
import random
import torch
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from transformers import AutoTokenizer, BertForQuestionAnswering, BertForMaskedLM
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

modelo_nombre = "mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"
tokenizador = AutoTokenizer.from_pretrained(modelo_nombre)
modelo_base = BertForQuestionAnswering.from_pretrained(modelo_nombre)
modelo_mlm = BertForMaskedLM.from_pretrained(modelo_nombre)

class GeneradorCuestionarios(nn.Module):
    def __init__(self, modelo_base):
        super(GeneradorCuestionarios, self).__init__()
        self.qa_model = modelo_base
        self.capa_generacion = nn.Linear(self.qa_model.config.hidden_size, 1000)
        self.capa_explicacion = nn.Linear(self.qa_model.config.hidden_size, 1000)

    def forward(self, input_ids, attention_mask):
        salidas = self.qa_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        start_logits, end_logits = salidas.start_logits, salidas.end_logits
        secuencia_oculta = salidas.hidden_states[-1][:, 0, :]
        generacion = self.capa_generacion(secuencia_oculta)
        explicacion = self.capa_explicacion(secuencia_oculta)
        return start_logits, end_logits, generacion, explicacion



def cargar_datos_json(ruta_archivo):
    try:
        with open(ruta_archivo, 'r', encoding='utf-8') as archivo:
            datos = json.load(archivo)
        return datos
    
    except Exception as e:
        print(f"Error al cargar el archivo JSON: {e}")
        return None



def preparar_datos(datos):
    return [item['texto'] for item in datos['quiz']]



def crear_y_entrenar_modelo(textos):
    modelo = GeneradorCuestionarios(modelo_base)
    
    entradas_codificadas = tokenizador(textos, padding=True, truncation=True, return_tensors="pt", max_length=256)
    ids_entrada = entradas_codificadas['input_ids']
    mascara_atencion = entradas_codificadas['attention_mask']
    
    conjunto_datos = TensorDataset(ids_entrada, mascara_atencion)
    cargador_datos = DataLoader(conjunto_datos, batch_size=2, shuffle=True)
    
    optimizador = optim.AdamW(modelo.parameters(), lr=1e-5)
    funcion_perdida = nn.CrossEntropyLoss()
    
    modelo.train()
    num_epocas = 1

    for epoca in range(num_epocas):
        perdida_total = 0

        for i, batch in enumerate(cargador_datos):
            ids_batch, mascara_batch = batch
            optimizador.zero_grad()
            start_logits, end_logits, generacion, explicacion = modelo(ids_batch, mascara_batch)  # Modificado aquí
            perdida = funcion_perdida(generacion, ids_batch[:, 0])  # Solo comparamos con el primer token
            perdida.backward()
            optimizador.step()
            perdida_total += perdida.item()

    return modelo



def extraer_palabras_clave(texto):
    # Implementar una función para extraer palabras clave del texto
    # Esto podría usar TF-IDF, conteo de frecuencias, o técnicas más avanzadas de NLP
    # Por ahora, usaremos una implementación simple basada en la frecuencia de palabras
    palabras = texto.lower().split()
    stop_words = set(stopwords.words('spanish'))
    palabras_filtradas = [palabra for palabra in palabras if palabra not in stop_words and palabra.isalnum()]
    frecuencia = Counter(palabras_filtradas)
    return [palabra for palabra, _ in frecuencia.most_common(5)]



def generar_importancia(frase, modelo_mlm, tokenizador):
    plantillas = [
        "Esta información es crucial porque [MASK].",
        "El concepto es relevante ya que [MASK].",
        "Entender esto es fundamental para [MASK]."
    ]
    plantilla = random.choice(plantillas)
    entradas = tokenizador(plantilla, return_tensors="pt")
    with torch.no_grad():
        salidas = modelo_mlm(**entradas)
    
    logits = salidas.logits
    mascara_token_index = torch.where(entradas["input_ids"][0] == tokenizador.mask_token_id)[0]
    predicciones = logits[0, mascara_token_index].topk(1)
    
    return tokenizador.decode(predicciones.indices[0]).strip()



def generar_contexto_adicional(frase, modelo_mlm, tokenizador):
    plantillas = [
        "Se relaciona con [MASK] en un contexto más amplio.",
        "Tiene implicaciones en [MASK] que son importantes de considerar.",
        "Este concepto es fundamental en el campo de [MASK]."
    ]
    plantilla = random.choice(plantillas)
    entradas = tokenizador(plantilla, return_tensors="pt")
    with torch.no_grad():
        salidas = modelo_mlm(**entradas)
    
    logits = salidas.logits
    mascara_token_index = torch.where(entradas["input_ids"][0] == tokenizador.mask_token_id)[0]
    predicciones = logits[0, mascara_token_index].topk(1)
    
    return tokenizador.decode(predicciones.indices[0]).strip()



def generar_explicacion(texto, modelo, tokenizador):
    # Generar una explicación basada en el texto y la pregunta
    plantillas_explicacion = [
        "Esta pregunta evalúa la comprensión del tema principal del texto, que se centra en {}.",
        "La respuesta correcta se basa en la idea central del pasaje, que trata sobre {}.",
        "Para responder correctamente, es importante entender que el texto aborda principalmente {}.",
        "La clave para esta pregunta está en reconocer que el autor enfatiza {} en el texto."
    ]
    
    palabras_clave = extraer_palabras_clave(texto)
    tema_principal = " y ".join(palabras_clave[:2])
    
    explicacion = random.choice(plantillas_explicacion).format(tema_principal)
    return explicacion



def extraer_frases_clave(texto):
    oraciones = sent_tokenize(texto)
    stop_words = set(stopwords.words('spanish'))
    vectorizador = TfidfVectorizer(stop_words=stop_words)
    tfidf_matriz = vectorizador.fit_transform(oraciones)
    
    frases_clave = []
    for i, oracion in enumerate(oraciones):
        if len(oracion.split()) > 10:
            puntuacion = tfidf_matriz[i].sum()
            frases_clave.append((oracion, puntuacion))
    
    frases_clave.sort(key=lambda x: x[1], reverse=True)
    return [frase for frase, _ in frases_clave[:10]]



def generar_pregunta_alternativas(texto, modelo, tokenizador):
    # Extraer conceptos clave del texto
    palabras_importantes = extraer_palabras_clave(texto)
    
    if len(palabras_importantes) < 3:
        return None
    
    # Generar una pregunta de análisis
    plantillas_preguntas = [
        "¿Cuál es el concepto principal abordado en el texto?",
        "¿Qué idea central se desarrolla en este pasaje?",
        "¿Cuál es la implicación más importante de la información presentada?",
        "¿Qué conclusión se puede extraer de este texto?",
        "¿Cuál es el propósito principal del autor en este pasaje?"
    ]
    pregunta = random.choice(plantillas_preguntas)
    
    # Generar opciones coherentes
    entradas = tokenizador(texto, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        salidas = modelo(**entradas, output_hidden_states=True)
    
    ultima_capa = salidas.hidden_states[-1]
    logits = modelo.capa_generacion(ultima_capa[:, 0, :])
    
    # Generar opciones usando el modelo y conceptos del texto
    opciones = set()
    while len(opciones) < 4:
        if random.random() < 0.5:  # 50% de probabilidad de usar palabras importantes del texto
            opcion = " ".join(random.sample(palabras_importantes, min(3, len(palabras_importantes))))
        else:  # 50% de probabilidad de generar una nueva opción con el modelo
            indices_prediccion = torch.topk(logits, k=3).indices[0]
            opcion = tokenizador.decode(indices_prediccion).strip()
        
        if opcion and len(opcion.split()) <= 3:
            opciones.add(opcion)
    
    opciones = list(opciones)
    
    # Seleccionar la respuesta correcta
    respuesta_correcta = random.choice(opciones)
    
    explicacion = generar_explicacion(texto, palabras_importantes)
    
    return {
        "pregunta": pregunta,
        "opciones": opciones,
        "respuesta_correcta": respuesta_correcta,
        "tipo": "alternativas",
        "explicacion": explicacion
    }

def generar_explicacion(texto, palabras_clave):
    # Generar una explicación basada en el texto y la pregunta
    plantillas_explicacion = [
        "Esta pregunta evalúa la comprensión del tema principal del texto, que se centra en {}.",
        "La respuesta correcta se basa en la idea central del pasaje, que trata sobre {}.",
        "Para responder correctamente, es importante entender que el texto aborda principalmente {}.",
        "La clave para esta pregunta está en reconocer que el autor enfatiza {} en el texto."
    ]
    
    tema_principal = " y ".join(palabras_clave[:2])
    
    explicacion = random.choice(plantillas_explicacion).format(tema_principal)
    return explicacion

def extraer_palabras_clave(texto):
    palabras = texto.lower().split()
    stop_words = set(stopwords.words('spanish'))
    palabras_filtradas = [palabra for palabra in palabras if palabra not in stop_words and palabra.isalnum()]
    frecuencia = Counter(palabras_filtradas)
    return [palabra for palabra, _ in frecuencia.most_common(5)]



def generar_pregunta_verdadero_falso(frase, modelo, tokenizador):
    palabras = frase.split()
    if len(palabras) < 10:
        return None
    
    palabras_importantes = [p for p in palabras if p.lower() not in set(stopwords.words('spanish'))]
    if len(palabras_importantes) < 3:
        return None
    
    es_verdadero = random.choice([True, False])
    
    if es_verdadero:
        pregunta = frase
    else:
        palabra_original = random.choice(palabras_importantes)
        indice_palabra = palabras.index(palabra_original)
        
        entradas = tokenizador(frase, return_tensers="pt", max_length=512, truncation=True)
        with torch.no_grad():
            start_logits, end_logits, generacion = modelo(**entradas)
        
        logits = generacion[0]
        predicciones = logits[indice_palabra].topk(5)
        
        palabra_generada = tokenizador.decode([predicciones.indices[random.randint(1, 4)].item()]).strip()
        palabras[indice_palabra] = palabra_generada
        pregunta = ' '.join(palabras)

    explicacion = generar_explicacion(frase, modelo, tokenizador)
    
    return {
        "pregunta": pregunta,
        "respuesta_correcta": "Verdadero" if es_verdadero else "Falso",
        "tipo": "verdadero_falso",
        "explicacion": explicacion
    }



def verificar_calidad_pregunta(pregunta, texto_original):
    vectorizador = TfidfVectorizer()
    vectores = vectorizador.fit_transform([pregunta['pregunta'], texto_original])
    similitud = cosine_similarity(vectores[0], vectores[1])[0][0]
    return similitud > 0.3



def generar_pregunta(texto, modelo, tokenizador):
    oraciones = sent_tokenize(texto)
    if not oraciones:
        return None
    
    oracion = random.choice(oraciones)
    palabras = oracion.split()
    if len(palabras) < 5:
        return None
    
    tipo_pregunta = random.choice(["verdadero_falso", "alternativas"])
    
    if tipo_pregunta == "verdadero_falso":
        es_verdadero = random.choice([True, False])
        if es_verdadero:
            pregunta = oracion
        else:
            indice_palabra = random.randint(0, len(palabras) - 1)
            palabras[indice_palabra] = random.choice(palabras)  # Reemplazamos una palabra al azar
            pregunta = ' '.join(palabras)
        
        return {
            "pregunta": pregunta,
            "respuesta_correcta": "Verdadero" if es_verdadero else "Falso",
            "tipo": "verdadero_falso"
        }
    else:
        palabra_clave = random.choice(palabras)
        indice_palabra = palabras.index(palabra_clave)
        palabras[indice_palabra] = "____"
        pregunta = ' '.join(palabras)
        
        opciones = [palabra_clave]
        while len(opciones) < 4:
            opcion = random.choice(palabras)
            if opcion not in opciones:
                opciones.append(opcion)
        random.shuffle(opciones)
        
        return {
            "pregunta": f"¿Qué palabra completa correctamente la siguiente oración? {pregunta}",
            "opciones": opciones,
            "respuesta_correcta": palabra_clave,
            "tipo": "alternativas"
        }

def generar_cuestionarios(datos, modelo, tokenizador):
    cuestionarios = {}
    for i, item in enumerate(datos['quiz']):
        materia = item['materia']
        
        if materia not in cuestionarios:
            cuestionarios[materia] = []
        
        preguntas = []
        intentos = 0

        while len(preguntas) < 10 and intentos < 30:
            try:
                pregunta = generar_pregunta(item['texto'], modelo, tokenizador)
                
                if pregunta:
                    preguntas.append(pregunta)
            except Exception as e:
                print(f"Error al generar pregunta: {e}")
            intentos += 1
        
        cuestionarios[materia].append({
            "texto": item['texto'],
            "fuente": item['fuente'],
            "preguntas": preguntas
        })

    return cuestionarios