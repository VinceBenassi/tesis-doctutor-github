# Generador de Quizzes
# por Franco Benassi
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    T5ForConditionalGeneration, 
    pipeline
)
import spacy
import json
import random
from typing import List, Dict, Tuple
import numpy as np

class GeneradorPreguntas:
    def __init__(self):
        # Cargar modelo T5 multilingüe para generación de preguntas
        self.tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
        self.question_generator = T5ForConditionalGeneration.from_pretrained("google/mt5-small")
        
        # Pipeline de question-answering en español
        self.qa_pipeline = pipeline(
            "question-answering",
            model="mrm8488/bert-spanish-cased-finetuned-squadv1-es",
            tokenizer="mrm8488/bert-spanish-cased-finetuned-squadv1-es"
        )
        
        # Modelo para clasificación de relevancia de respuestas
        self.classifier = AutoModelForSequenceClassification.from_pretrained(
            "bertin-project/bertin-roberta-base-spanish"
        )
        
        # Spacy para procesamiento de texto en español
        self.nlp = spacy.load("es_core_news_lg")
        
    def extraer_temas_principales(self, texto: str) -> List[str]:
        """Extrae los temas principales del texto usando análisis NLP"""
        doc = self.nlp(texto)
        temas = []
        
        # Extraer frases nominales importantes
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 4:
                temas.append(chunk.text)
        
        # Extraer entidades relevantes
        for ent in doc.ents:
            if ent.label_ in ['PER', 'ORG', 'LOC', 'CONCEPT']:
                temas.append(ent.text)
        
        return list(set(temas))
    
    def generar_pregunta_analisis(self, tema: str, contexto: str) -> Tuple[str, str]:
        """Genera una pregunta de análisis sobre un tema específico"""
        # Preparar prompt para T5
        prompt = f"genera una pregunta de análisis sobre: {tema}\ncontexto: {contexto}"
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        # Generar pregunta
        outputs = self.question_generator.generate(
            inputs["input_ids"],
            max_length=64,
            num_beams=4,
            no_repeat_ngram_size=2
        )
        pregunta = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Obtener respuesta usando QA pipeline
        respuesta = self.qa_pipeline(question=pregunta, context=contexto)
        return pregunta, respuesta['answer']
    
    def generar_alternativas_coherentes(self, respuesta_correcta: str, contexto: str) -> List[str]:
        """Genera alternativas coherentes pero incorrectas"""
        doc = self.nlp(contexto)
        alternativas = [respuesta_correcta]
        
        # Extraer candidatos semánticamente similares
        candidatos = []
        for sent in doc.sents:
            # Calcular similitud semántica
            similitud = sent.similarity(self.nlp(respuesta_correcta))
            if 0.5 < similitud < 0.9:  # Lo suficientemente similar pero no idéntico
                candidatos.append(sent.text)
        
        # Seleccionar las mejores alternativas
        for candidato in sorted(candidatos, key=len)[:3]:  # Tomar las 3 más cortas
            if self.es_alternativa_valida(candidato, alternativas):
                alternativas.append(self.refinar_alternativa(candidato))
        
        # Si no hay suficientes alternativas, generar algunas
        while len(alternativas) < 4:
            nueva_alt = self.generar_alternativa_similar(respuesta_correcta)
            if nueva_alt and self.es_alternativa_valida(nueva_alt, alternativas):
                alternativas.append(nueva_alt)
        
        random.shuffle(alternativas)
        return alternativas, alternativas.index(respuesta_correcta)
    
    def es_alternativa_valida(self, candidato: str, existentes: List[str]) -> bool:
        """Verifica si una alternativa es válida y coherente"""
        if not candidato or len(candidato.split()) > 15:
            return False
            
        # Verificar que no sea muy similar a las existentes
        for existente in existentes:
            if self.nlp(candidato).similarity(self.nlp(existente)) > 0.8:
                return False
        
        # Verificar que tenga estructura gramatical válida
        doc = self.nlp(candidato)
        return any(token.pos_ in ['NOUN', 'VERB', 'ADJ'] for token in doc)
    
    def refinar_alternativa(self, texto: str) -> str:
        """Refina una alternativa para hacerla más concisa y coherente"""
        doc = self.nlp(texto)
        
        # Extraer la parte más relevante si es muy larga
        if len(doc) > 10:
            chunks = list(doc.noun_chunks)
            if chunks:
                return chunks[0].text
        return texto
    
    def generar_alternativa_similar(self, texto: str) -> str:
        """Genera una alternativa similar pero incorrecta"""
        doc = self.nlp(texto)
        palabras = [token.text for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ']]
        
        if not palabras:
            return None
            
        # Reemplazar una palabra clave por un sinónimo o término relacionado
        palabra = random.choice(palabras)
        similares = []
        
        # Buscar términos similares usando los vectores de palabra
        vector = self.nlp(palabra).vector
        for token in self.nlp.vocab:
            if token.has_vector:
                similitud = np.dot(token.vector, vector) / (np.linalg.norm(token.vector) * np.linalg.norm(vector))
                if 0.5 < similitud < 0.9:
                    similares.append(token.text)
        
        if similares:
            return texto.replace(palabra, random.choice(similares))
        return None

    def generar_pregunta_verdadero_falso(self, texto: str) -> Tuple[str, str]:
        """Genera una pregunta de verdadero/falso"""
        doc = self.nlp(texto)
        oraciones = list(doc.sents)
        
        if not oraciones:
            return None, None
            
        # Seleccionar una oración informativa
        oracion = max(oraciones, key=lambda x: len([t for t in x if t.pos_ in ['NOUN', 'VERB']]))
        es_verdadero = random.choice([True, False])
        
        if es_verdadero:
            return f"¿Es correcto que {oracion.text.strip()}?", "Verdadero"
        else:
            # Modificar la oración para hacerla falsa pero plausible
            oracion_modificada = self.modificar_oracion_para_falso(oracion.text)
            return f"¿Es correcto que {oracion_modificada}?", "Falso"
    
    def modificar_oracion_para_falso(self, oracion: str) -> str:
        """Modifica una oración para hacerla falsa pero plausible"""
        doc = self.nlp(oracion)
        
        # Estrategias de modificación
        estrategias = [
            self._negar_afirmacion,
            self._cambiar_entidad,
            self._modificar_cantidad
        ]
        
        return random.choice(estrategias)(doc)
    
    def _negar_afirmacion(self, doc) -> str:
        """Niega la afirmación principal"""
        texto = doc.text
        for token in doc:
            if token.pos_ == "VERB" and token.dep_ == "ROOT":
                return texto.replace(token.text, f"no {token.text}")
        return texto
    
    def _cambiar_entidad(self, doc) -> str:
        """Cambia una entidad por otra del mismo tipo"""
        texto = doc.text
        for ent in doc.ents:
            if ent.label_ in ['PER', 'ORG', 'LOC']:
                # Usar una entidad diferente del mismo tipo
                otras_entidades = [e.text for e in self.nlp.vocab.vectors if e.has_vector]
                if otras_entidades:
                    return texto.replace(ent.text, random.choice(otras_entidades))
        return texto
    
    def _modificar_cantidad(self, doc) -> str:
        """Modifica números o cantidades"""
        texto = doc.text
        for token in doc:
            if token.like_num:
                try:
                    num = int(token.text)
                    nuevo_num = num * 2 if num < 100 else num // 2
                    return texto.replace(token.text, str(nuevo_num))
                except ValueError:
                    pass
        return texto

def generar_cuestionario(ruta_archivo: str) -> Dict:
    """Genera un cuestionario completo a partir del archivo JSON"""
    try:
        with open(ruta_archivo, 'r', encoding='utf-8') as archivo:
            datos = json.load(archivo)
    except Exception as e:
        print(f"Error al cargar el archivo JSON: {e}")
        return None

    generador = GeneradorPreguntas()
    cuestionarios = {}

    for item in datos['quiz']:
        materia = item['materia']
        if materia not in cuestionarios:
            cuestionarios[materia] = []

        texto = item['texto']
        preguntas = []
        
        # Extraer temas principales
        temas = generador.extraer_temas_principales(texto)
        
        # Generar preguntas de análisis
        for _ in range(7):
            if temas:
                tema = random.choice(temas)
                pregunta, respuesta = generador.generar_pregunta_analisis(tema, texto)
                if pregunta and respuesta:
                    alternativas, indice_correcto = generador.generar_alternativas_coherentes(
                        respuesta, texto
                    )
                    preguntas.append({
                        "pregunta": pregunta,
                        "opciones": alternativas,
                        "respuesta_correcta": alternativas[indice_correcto],
                        "tipo": "alternativas"
                    })

        # Generar preguntas de verdadero/falso
        for _ in range(3):
            pregunta, respuesta = generador.generar_pregunta_verdadero_falso(texto)
            if pregunta and respuesta:
                preguntas.append({
                    "pregunta": pregunta,
                    "opciones": ["Verdadero", "Falso"],
                    "respuesta_correcta": respuesta,
                    "tipo": "verdadero_falso"
                })

        cuestionarios[materia].append({
            "texto": texto,
            "fuente": item['fuente'],
            "preguntas": preguntas
        })

    return cuestionarios