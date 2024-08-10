# Generador de Quizzes
# por Franco Benassi
import json
import random
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
import spacy
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')
nlp = spacy.load("es_core_news_sm")

# Cargar modelos
bert_tokenizer = BertTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
bert_model = BertModel.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
sentence_model = SentenceTransformer('hiiamsid/sentence_similarity_spanish_es')

# Red neuronal simple para clasificación
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        return self.layer2(x)

classifier = SimpleClassifier(768, 256, 2)  # 768 es la dimensión de salida de BERT

def cargar_datos_json(ruta_archivo):
    with open(ruta_archivo, 'r', encoding='utf-8') as archivo:
        return json.load(archivo)

def extraer_oraciones_clave(texto, n=10):
    oraciones = sent_tokenize(texto)
    return random.sample(oraciones, min(n, len(oraciones)))

def generar_alternativa(oracion, texto_completo):
    doc = nlp(oracion)
    tokens = [token.text for token in doc if token.pos_ in ["NOUN", "VERB", "ADJ", "PROPN"]]
    
    if not tokens:
        return None
    
    palabra_a_reemplazar = random.choice(tokens)
    
    try:
        input_ids = bert_tokenizer.encode(oracion, return_tensors="pt")
        mask_index = (input_ids == bert_tokenizer.convert_tokens_to_ids(palabra_a_reemplazar))[0].nonzero(as_tuple=True)[0]
        
        if len(mask_index) == 0:
            return None
        
        input_ids[0][mask_index[0]] = bert_tokenizer.mask_token_id
        
        with torch.no_grad():
            outputs = bert_model(input_ids)
        
        logits = outputs.last_hidden_state
        masked_index = (input_ids[0] == bert_tokenizer.mask_token_id).nonzero().item()
        probs = logits[0, masked_index].softmax(dim=0)
        top_5_ids = probs.topk(5)[1]
        
        for token_id in top_5_ids:
            palabra_nueva = bert_tokenizer.decode([token_id])
            if palabra_nueva.lower() != palabra_a_reemplazar.lower():
                alternativa = oracion.replace(palabra_a_reemplazar, palabra_nueva)
                if alternativa != oracion and not any(alternativa in sent for sent in sent_tokenize(texto_completo)):
                    return alternativa
    
    except Exception as e:
        print(f"Error al generar alternativa: {e}")
        return None
    
    return None

def generar_pregunta_alternativas(texto, oracion):
    pregunta = oracion
    respuesta_correcta = oracion
    distractores = []
    
    while len(distractores) < 3:
        distractor = generar_alternativa(oracion, texto)
        if distractor and distractor not in distractores:
            distractores.append(distractor)
    
    opciones = [respuesta_correcta] + distractores
    random.shuffle(opciones)
    
    return {
        'tipo': 'alternativas',
        'pregunta': pregunta,
        'opciones': opciones,
        'respuesta_correcta': respuesta_correcta,
        'explicacion': generar_explicacion(texto, pregunta, respuesta_correcta)
    }

def generar_pregunta_verdadero_falso(texto, oracion):
    es_verdadero = random.choice([True, False])
    
    if es_verdadero:
        pregunta = oracion
        respuesta = True
    else:
        pregunta = generar_alternativa(oracion, texto) or oracion
        respuesta = False
    
    return {
        'tipo': 'verdadero_falso',
        'pregunta': pregunta,
        'respuesta_correcta': respuesta,
        'explicacion': generar_explicacion(texto, pregunta, oracion if es_verdadero else None)
    }

def generar_explicacion(texto, pregunta, respuesta_correcta):
    oraciones = sent_tokenize(texto)
    pregunta_embedding = sentence_model.encode(pregunta)
    
    similitudes = []
    for oracion in oraciones:
        if oracion != pregunta and oracion != respuesta_correcta:
            oracion_embedding = sentence_model.encode(oracion)
            similitud = torch.cosine_similarity(torch.tensor(pregunta_embedding), torch.tensor(oracion_embedding), dim=0)
            similitudes.append((oracion, similitud.item()))
    
    similitudes.sort(key=lambda x: x[1], reverse=True)
    
    explicacion = similitudes[0][0] if similitudes else "No se pudo generar una explicación."
    return explicacion

def generar_cuestionario(texto, num_preguntas=10):
    oraciones_clave = extraer_oraciones_clave(texto)
    preguntas = []
    
    while len(preguntas) < num_preguntas:
        tipo_pregunta = random.choice(['alternativas', 'verdadero_falso'])
        oracion = random.choice(oraciones_clave)
        
        try:
            if tipo_pregunta == 'alternativas':
                pregunta = generar_pregunta_alternativas(texto, oracion)
            else:
                pregunta = generar_pregunta_verdadero_falso(texto, oracion)
            
            if pregunta and pregunta not in preguntas:
                preguntas.append(pregunta)
        except Exception as e:
            print(f"Error al generar pregunta: {e}")
            continue
    
    return preguntas

def generar_cuestionarios_desde_json(datos_json):
    cuestionarios = {}
    
    for item in datos_json['quiz']:
        categoria = item['materia']
        texto = item['texto']
        fuente = item['fuente']
        
        if categoria not in cuestionarios:
            cuestionarios[categoria] = []
        
        try:
            cuestionario = generar_cuestionario(texto)
            cuestionarios[categoria].append({
                'fuente': fuente,
                'preguntas': cuestionario
            })
        except Exception as e:
            print(f"Error al generar cuestionario para {categoria}: {e}")
    
    return cuestionarios