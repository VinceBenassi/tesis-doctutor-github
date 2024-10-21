# Generador de Quizzes
# por Franco Benassi
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
    pipeline
)
import spacy
import json
import random
from typing import List, Dict, Tuple
import numpy as np

class GeneradorCuestionarios:
    def __init__(self):
        print("Inicializando modelos...")
        
        # Modelo principal para generación y comprensión
        self.model_name = "google/flan-t5-xl"  # Modelo más grande para mejor comprensión
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        
        # Pipeline específico para español
        self.qa_pipeline = pipeline(
            "question-answering",
            model="PlanTL-GOB-ES/roberta-large-bne-sqac",
            tokenizer="PlanTL-GOB-ES/roberta-large-bne-sqac"
        )
        
        # Modelo SpaCy para análisis de texto en español
        self.nlp = spacy.load("es_core_news_lg")
        
        # Tipos de preguntas analíticas
        self.tipos_preguntas = [
            "análisis_causa_efecto",
            "comparación_contraste",
            "evaluación_crítica",
            "aplicación_práctica",
            "síntesis_conceptual"
        ]

    def _extraer_temas_principales(self, texto: str) -> List[Dict]:
        """Extrae los temas principales y conceptos clave del texto"""
        doc = self.nlp(texto)
        
        # Identificar frases nominales importantes
        temas = []
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) > 1:  # Evitar conceptos de una sola palabra
                temas.append({
                    'concepto': chunk.text,
                    'raíz': chunk.root.text,
                    'contexto': chunk.sent.text
                })
        
        # Identificar relaciones entre conceptos
        relaciones = []
        for sent in doc.sents:
            for token in sent:
                if token.dep_ in ['nsubj', 'dobj'] and token.head.pos_ == 'VERB':
                    relaciones.append({
                        'sujeto': token.text,
                        'verbo': token.head.text,
                        'contexto': sent.text
                    })
        
        return {'temas': temas, 'relaciones': relaciones}

    def _generar_pregunta_analítica(self, tema: Dict, tipo: str) -> str:
        """Genera una pregunta analítica basada en el tema y tipo especificado"""
        prompt = f"""
        Contexto: {tema['contexto']}
        Concepto principal: {tema['concepto']}
        
        Genera una pregunta de {tipo} que:
        1. Sea analítica y requiera comprensión profunda
        2. No se pueda responder con solo memorización
        3. Esté relacionada con el concepto principal
        4. Sea clara y específica
        
        Pregunta:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=128,
            temperature=0.8,
            num_beams=4,
            no_repeat_ngram_size=2
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _generar_alternativas_coherentes(self, pregunta: str, respuesta_correcta: str, contexto: str) -> List[str]:
        """Genera alternativas coherentes y plausibles"""
        prompt = f"""
        Pregunta: {pregunta}
        Respuesta correcta: {respuesta_correcta}
        Contexto: {contexto}
        
        Genera tres alternativas incorrectas que:
        1. Sean plausibles y relacionadas con el tema
        2. Tengan el mismo nivel de detalle que la respuesta correcta
        3. Sean distintas entre sí
        4. Puedan hacer dudar al estudiante
        
        Alternativas:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=256,
            num_return_sequences=3,
            temperature=0.8,
            num_beams=4,
            no_repeat_ngram_size=2
        )
        
        alternativas = [respuesta_correcta]
        for output in outputs:
            alternativa = self.tokenizer.decode(output, skip_special_tokens=True)
            if alternativa not in alternativas and len(alternativa.split()) > 2:
                alternativas.append(alternativa)
        
        # Asegurar que tengamos exactamente 4 alternativas
        while len(alternativas) < 4:
            alternativas.append(self._generar_alternativa_individual(pregunta, respuesta_correcta))
        
        random.shuffle(alternativas)
        return alternativas[:4]

    def _generar_alternativa_individual(self, pregunta: str, respuesta_correcta: str) -> str:
        """Genera una alternativa individual coherente"""
        prompt = f"""
        Pregunta: {pregunta}
        Respuesta correcta: {respuesta_correcta}
        
        Genera una alternativa incorrecta pero plausible que:
        1. Tenga sentido en el contexto de la pregunta
        2. Sea diferente a: {respuesta_correcta}
        3. Pueda confundir al estudiante por ser cercana a la correcta
        
        Alternativa:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=128,
            temperature=0.7,
            num_beams=4
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _generar_pregunta_vf(self, texto: str, tema: Dict) -> Tuple[str, str]:
        """Genera una pregunta de verdadero/falso analítica"""
        prompt = f"""
        Contexto: {tema['contexto']}
        Concepto: {tema['concepto']}
        
        Genera una afirmación para verdadero/falso que:
        1. Sea específica y verificable
        2. Requiera comprensión del tema
        3. No sea obvia
        4. Use lenguaje preciso
        
        Afirmación:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=128,
            temperature=0.8,
            num_beams=4
        )
        
        afirmación = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Usar QA para verificar la respuesta
        resultado = self.qa_pipeline(
            question=f"¿Es verdadera la siguiente afirmación? {afirmación}",
            context=texto
        )
        
        es_verdadero = resultado['score'] > 0.8
        return f"¿Es correcto afirmar que {afirmación}?", "Verdadero" if es_verdadero else "Falso"

    def generar_cuestionario(self, ruta_archivo: str) -> Dict:
        """Genera el cuestionario completo"""
        try:
            with open(ruta_archivo, 'r', encoding='utf-8') as archivo:
                datos = json.load(archivo)
        except Exception as e:
            print(f"Error al cargar el archivo JSON: {e}")
            return None

        cuestionarios = {}
        
        for item in datos['quiz']:
            materia = item['materia']
            texto = item['texto']
            
            if materia not in cuestionarios:
                cuestionarios[materia] = []
            
            # Extraer temas principales
            info_texto = self._extraer_temas_principales(texto)
            preguntas = []
            
            # Generar preguntas de alternativas (7)
            for _ in range(7):
                try:
                    # Seleccionar tema y tipo de pregunta aleatorios
                    tema = random.choice(info_texto['temas'])
                    tipo_pregunta = random.choice(self.tipos_preguntas)
                    
                    # Generar pregunta
                    pregunta = self._generar_pregunta_analítica(tema, tipo_pregunta)
                    
                    # Obtener respuesta usando QA
                    qa_result = self.qa_pipeline(
                        question=pregunta,
                        context=texto
                    )
                    respuesta_correcta = qa_result['answer']
                    
                    # Generar alternativas
                    alternativas = self._generar_alternativas_coherentes(
                        pregunta,
                        respuesta_correcta,
                        tema['contexto']
                    )
                    
                    preguntas.append({
                        "pregunta": pregunta,
                        "opciones": alternativas,
                        "respuesta_correcta": respuesta_correcta,
                        "tipo": "alternativas"
                    })
                    
                except Exception as e:
                    print(f"Error al generar pregunta de alternativas: {str(e)}")
                    continue
            
            # Generar preguntas de verdadero/falso (3)
            for _ in range(3):
                try:
                    tema = random.choice(info_texto['temas'])
                    pregunta, respuesta = self._generar_pregunta_vf(texto, tema)
                    
                    preguntas.append({
                        "pregunta": pregunta,
                        "opciones": ["Verdadero", "Falso"],
                        "respuesta_correcta": respuesta,
                        "tipo": "verdadero_falso"
                    })
                    
                except Exception as e:
                    print(f"Error al generar pregunta V/F: {str(e)}")
                    continue
            
            if preguntas:
                cuestionarios[materia].extend(preguntas)
        
        return cuestionarios