# Generador de Cuestionarios
# Por Franco Benassi
import json
import random
import spacy
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
    T5ForConditionalGeneration,
    T5Tokenizer
)
from sentence_transformers import SentenceTransformer
import numpy as np
from nltk.tokenize import sent_tokenize
import re

class GeneradorCuestionarios:
    def __init__(self):
        # Cargar modelos
        self.nlp = spacy.load("es_core_news_lg")
        self.sentence_model = SentenceTransformer('hiiamsid/sentence_similarity_spanish_es')
        
        # Modelo T5 en español para generación de preguntas
        self.t5_tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
        self.t5_model = T5ForConditionalGeneration.from_pretrained("google/mt5-small")
        
        # Pipeline de clasificación de texto
        self.classifier = pipeline(
            "text-classification",
            model="dccuchile/bert-base-spanish-wwm-cased",
            tokenizer="dccuchile/bert-base-spanish-wwm-cased"
        )

    def procesar_texto(self, texto):
        """Procesa el texto utilizando SpaCy para análisis lingüístico"""
        doc = self.nlp(texto)
        return doc

    def extraer_conceptos_clave(self, doc):
        """Extrae conceptos clave del texto procesado"""
        conceptos = []
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PERSON', 'LOC', 'MISC']:
                conceptos.append(ent.text)
        
        # Extraer sustantivos y frases nominales importantes
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) > 1:  # Frases de más de una palabra
                conceptos.append(chunk.text)
                
        return list(set(conceptos))

    def generar_pregunta_alternativas(self, texto, concepto):
        """Genera una pregunta de alternativas usando T5"""
        input_text = f"genera una pregunta sobre: {concepto} contexto: {texto}"
        
        # Generar pregunta
        inputs = self.t5_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.t5_model.generate(inputs, max_length=150, min_length=50, num_return_sequences=1)
        pregunta = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Generar alternativas coherentes
        alternativas = self.generar_alternativas(texto, concepto, pregunta)
        respuesta_correcta = alternativas[0]  # La primera alternativa es la correcta
        random.shuffle(alternativas)  # Mezclar alternativas
        
        return {
            "tipo": "alternativas",
            "pregunta": pregunta,
            "opciones": alternativas,
            "respuesta_correcta": respuesta_correcta,
            "explicacion": self.generar_explicacion(texto, pregunta, respuesta_correcta)
        }

    def generar_alternativas(self, texto, concepto, pregunta):
        """Genera alternativas coherentes para la pregunta"""
        doc = self.nlp(texto)
        alternativas = []
        
        # Encontrar la respuesta correcta
        for sent in doc.sents:
            if concepto in sent.text:
                alternativas.append(self.extraer_respuesta_correcta(sent.text, concepto))
                break
        
        # Generar distractores coherentes
        distractores = self.generar_distractores(texto, concepto, alternativas[0])
        alternativas.extend(distractores[:3])  # Tomar 3 distractores
        
        return alternativas

    def extraer_respuesta_correcta(self, oracion, concepto):
        """Extrae la respuesta correcta de la oración"""
        doc = self.nlp(oracion)
        for chunk in doc.noun_chunks:
            if concepto not in chunk.text and len(chunk.text.split()) > 1:
                return chunk.text.strip()
        return ""

    def generar_distractores(self, texto, concepto, respuesta_correcta):
        """Genera distractores coherentes"""
        doc = self.nlp(texto)
        distractores = []
        
        for sent in doc.sents:
            for chunk in self.nlp(sent.text).noun_chunks:
                if (concepto not in chunk.text and 
                    chunk.text != respuesta_correcta and 
                    len(chunk.text.split()) > 1):
                    distractores.append(chunk.text.strip())
        
        # Filtrar y seleccionar los distractores más relevantes
        distractores = list(set(distractores))
        distractores = self.filtrar_distractores_similares(distractores, respuesta_correcta)
        return distractores

    def filtrar_distractores_similares(self, distractores, respuesta_correcta):
        """Filtra distractores basándose en similitud semántica"""
        if not distractores:
            return []
            
        # Calcular embeddings
        embeddings = self.sentence_model.encode(distractores + [respuesta_correcta])
        respuesta_embedding = embeddings[-1]
        
        # Calcular similitudes
        similitudes = np.dot(embeddings[:-1], respuesta_embedding)
        
        # Seleccionar distractores con similitud moderada
        distractores_filtrados = []
        for i, similitud in enumerate(similitudes):
            if 0.3 < similitud < 0.7:  # Rango de similitud adecuado
                distractores_filtrados.append(distractores[i])
        
        return distractores_filtrados[:3]  # Retornar máximo 3 distractores

    def generar_pregunta_verdadero_falso(self, texto):
        """Genera una pregunta de verdadero/falso"""
        oraciones = sent_tokenize(texto)
        oracion_original = random.choice(oraciones)
        
        # Generar versión modificada para preguntas falsas
        if random.random() < 0.5:
            enunciado = self.modificar_oracion(oracion_original)
            es_verdadero = False
        else:
            enunciado = oracion_original
            es_verdadero = True
            
        return {
            "tipo": "verdadero_falso",
            "pregunta": enunciado,
            "respuesta_correcta": "Verdadero" if es_verdadero else "Falso",
            "explicacion": self.generar_explicacion_vf(oracion_original, enunciado, es_verdadero)
        }

    def modificar_oracion(self, oracion):
        """Modifica una oración para crear una versión falsa"""
        doc = self.nlp(oracion)
        palabras = [token.text for token in doc]
        
        # Identificar palabras clave que se pueden modificar
        modificables = []
        for i, token in enumerate(doc):
            if token.pos_ in ['NOUN', 'VERB', 'ADJ']:
                modificables.append(i)
        
        if modificables:
            # Modificar una palabra clave
            indice = random.choice(modificables)
            palabras[indice] = self.obtener_antonimo_o_alternativa(doc[indice])
        
        return ' '.join(palabras)

    def obtener_antonimo_o_alternativa(self, token):
        """Obtiene un antónimo o alternativa para una palabra"""
        # Implementar lógica para obtener antónimos o alternativas
        # Este es un ejemplo simplificado
        antonimos = {
            'grande': 'pequeño',
            'bueno': 'malo',
            'alto': 'bajo',
            'rápido': 'lento',
            # Agregar más pares según sea necesario
        }
        return antonimos.get(token.text.lower(), token.text)

    def generar_explicacion(self, texto, pregunta, respuesta):
        """Genera una explicación coherente para la respuesta"""
        input_text = f"Explica por qué la respuesta '{respuesta}' es correcta para la pregunta: {pregunta}"
        
        inputs = self.t5_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.t5_model.generate(
            inputs,
            max_length=200,
            min_length=50,
            num_return_sequences=1,
            temperature=0.7
        )
        
        explicacion = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return explicacion

    def generar_explicacion_vf(self, original, modificada, es_verdadero):
        """Genera una explicación para preguntas de verdadero/falso"""
        if es_verdadero:
            return f"El enunciado es correcto porque coincide con la información: {original}"
        else:
            return f"El enunciado es falso. La información correcta es: {original}"

    def generar_cuestionario(self, texto, materia, fuente, num_preguntas=10):
        """Genera un cuestionario completo"""
        doc = self.procesar_texto(texto)
        conceptos = self.extraer_conceptos_clave(doc)
        
        preguntas = []
        num_alternativas = num_preguntas // 2
        num_vf = num_preguntas - num_alternativas
        
        # Generar preguntas de alternativas
        for _ in range(num_alternativas):
            if conceptos:
                concepto = random.choice(conceptos)
                pregunta = self.generar_pregunta_alternativas(texto, concepto)
                preguntas.append(pregunta)
        
        # Generar preguntas de verdadero/falso
        for _ in range(num_vf):
            pregunta = self.generar_pregunta_verdadero_falso(texto)
            preguntas.append(pregunta)
        
        return {
            "materia": materia,
            "fuente": fuente,
            "preguntas": preguntas
        }

    def generar_cuestionario_desde_json(self, ruta_entrada):
        """Genera cuestionarios a partir del archivo quiz.json"""
        # Cargar datos del JSON
        with open(ruta_entrada, 'r', encoding='utf-8') as f:
            datos = json.load(f)

        generador = GeneradorCuestionarios()
        cuestionarios = []

        # Generar cuestionarios para cada texto
        for item in datos['quiz']:
            cuestionario = generador.generar_cuestionario(
                item['texto'],
                item['materia'],
                item['fuente']
            )
            cuestionarios.append(cuestionario)

        # Guardar cuestionarios generados
        with open('tutorApp/static/json/cuestionarios.json', 'w', encoding='utf-8') as f:
            json.dump(cuestionarios, f, ensure_ascii=False, indent=4)

        return cuestionarios