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
from typing import List, Dict, Any
import numpy as np

class GeneradorCuestionarios:
    def __init__(self):
        print("Inicializando modelos...")
        
        # Modelo T5 multilingüe para generación de preguntas
        self.qa_model_name = "google/mt5-large"
        self.qa_tokenizer = AutoTokenizer.from_pretrained(self.qa_model_name)
        self.qa_model = T5ForConditionalGeneration.from_pretrained(self.qa_model_name)
        
        # Pipeline de BERT multilingüe para verificación de respuestas
        self.verifier = pipeline(
            "question-answering",
            model="mrm8488/bert-multilingual-uncased-question-answering",
            tokenizer="mrm8488/bert-multilingual-uncased-question-answering"
        )
        
        # Modelo T5 para generar alternativas 
        self.alt_model_name = "google/flan-t5-large"
        self.alt_tokenizer = AutoTokenizer.from_pretrained(self.alt_model_name)
        self.alt_model = T5ForConditionalGeneration.from_pretrained(self.alt_model_name)
        
        # SpaCy para análisis semántico
        self.nlp = spacy.load("es_core_news_lg")

    def _analizar_texto(self, texto: str) -> List[Dict]:
        """Extrae conceptos y relaciones clave del texto"""
        doc = self.nlp(texto)
        conceptos = []
        
        for sent in doc.sents:
            if len(sent.text.split()) > 8:
                # Identificar entidades y conceptos principales
                entidades = []
                for ent in sent.ents:
                    if ent.label_ in ['ORG', 'PERSON', 'LOC', 'MISC']:
                        entidades.append({
                            'texto': ent.text,
                            'tipo': ent.label_
                        })
                
                # Identificar verbos y sus argumentos
                verbos = []
                for token in sent:
                    if token.pos_ == 'VERB':
                        argumentos = []
                        for child in token.children:
                            if child.dep_ in ['nsubj', 'dobj', 'iobj']:
                                argumentos.append({
                                    'texto': child.text,
                                    'rol': child.dep_
                                })
                        if argumentos:
                            verbos.append({
                                'verbo': token.text,
                                'argumentos': argumentos
                            })
                
                if entidades or verbos:
                    conceptos.append({
                        'oracion': sent.text,
                        'entidades': entidades,
                        'verbos': verbos
                    })
        
        return conceptos

    def _generar_pregunta(self, concepto: Dict, texto_completo: str) -> Dict:
        """Genera una pregunta analítica basada en el concepto"""
        try:
            # Construir prompt específico según el tipo de concepto
            if concepto['entidades']:
                entidad = random.choice(concepto['entidades'])
                prompt = f"""
                Contexto: {concepto['oracion']}
                
                Genera una pregunta analítica sobre {entidad['texto']} que:
                1. Requiera comprensión profunda del tema
                2. Tenga múltiples respuestas posibles pero una clara correcta
                3. Se enfoque en las relaciones, causas o efectos
                4. No se pueda responder con una sola palabra
                """
            elif concepto['verbos']:
                verbo = random.choice(concepto['verbos'])
                prompt = f"""
                Contexto: {concepto['oracion']}
                
                Genera una pregunta analítica sobre la acción '{verbo['verbo']}' que:
                1. Explore las consecuencias o implicaciones
                2. Requiera análisis y razonamiento
                3. Relacione múltiples conceptos del texto
                4. Tenga una respuesta específica pero no obvia
                """
            
            # Generar pregunta
            inputs = self.qa_tokenizer(prompt, return_tensors="pt", max_length=512, 
                                   truncation=True)
            outputs = self.qa_model.generate(
                inputs.input_ids,
                max_length=150,
                num_beams=5,
                temperature=0.8,
                do_sample=True,
                top_p=0.9
            )
            pregunta = self.qa_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Obtener respuesta correcta usando el verificador
            respuesta = self.verifier(
                question=pregunta,
                context=texto_completo
            )
            
            # Generar alternativas plausibles
            alternativas = self._generar_alternativas(
                pregunta,
                respuesta['answer'],
                concepto,
                texto_completo
            )
            
            if alternativas and len(alternativas) >= 4:
                return {
                    'pregunta': pregunta,
                    'opciones': alternativas[:4],
                    'respuesta_correcta': respuesta['answer'],
                    'tipo': 'alternativas'
                }
            
            return None
            
        except Exception as e:
            print(f"Error generando pregunta: {str(e)}")
            return None

    def _generar_alternativas(
        self,
        pregunta: str,
        respuesta_correcta: str,
        concepto: Dict,
        texto: str
    ) -> List[str]:
        """Genera alternativas coherentes y plausibles"""
        alternativas = [respuesta_correcta]
        
        # Construir prompt para generar alternativas engañosas
        prompt = f"""
        Pregunta: {pregunta}
        Respuesta correcta: {respuesta_correcta}
        Contexto: {concepto['oracion']}
        
        Genera una respuesta alternativa que:
        1. Sea incorrecta pero parezca plausible
        2. Use conceptos relacionados al tema
        3. Sea del mismo tipo y estilo que la respuesta correcta
        4. Pueda confundir a alguien que no conoce bien el tema
        5. Sea gramaticalmente correcta y tenga sentido
        """
        
        try:
            # Generar múltiples alternativas
            for _ in range(5):
                inputs = self.alt_tokenizer(prompt, return_tensors="pt", 
                                        max_length=512, truncation=True)
                outputs = self.alt_model.generate(
                    inputs.input_ids,
                    max_length=100,
                    temperature=0.9,
                    do_sample=True,
                    top_p=0.9,
                    num_return_sequences=1,
                    no_repeat_ngram_size=2
                )
                
                alternativa = self.alt_tokenizer.decode(outputs[0], 
                                                    skip_special_tokens=True)
                
                # Validar calidad de la alternativa
                if self._validar_alternativa(alternativa, alternativas, pregunta):
                    alternativas.append(alternativa)
                
                if len(alternativas) >= 4:
                    break
            
            random.shuffle(alternativas)
            return alternativas
            
        except Exception as e:
            print(f"Error generando alternativas: {str(e)}")
            return []

    def _validar_alternativa(
        self,
        alternativa: str,
        existentes: List[str],
        pregunta: str
    ) -> bool:
        """Valida la coherencia y uniqueness de una alternativa"""
        if not alternativa or len(alternativa.split()) < 3:
            return False
            
        # Verificar gramática básica
        doc = self.nlp(alternativa)
        if not any(token.pos_ == 'VERB' for token in doc):
            return False
            
        # Verificar que no sea muy similar a otras alternativas
        for existente in existentes:
            similitud = self.nlp(existente).similarity(doc)
            if similitud > 0.7:  # Muy similar
                return False
                
        # Verificar relación con la pregunta
        similitud_pregunta = self.nlp(pregunta).similarity(doc)
        if similitud_pregunta < 0.3:  # Poco relacionada
            return False
            
        # Verificar coherencia semántica
        tiene_sentido = True
        for token in doc:
            if token.dep_ == 'ROOT':  # Verbo principal
                tiene_argumentos = False
                for child in token.children:
                    if child.dep_ in ['nsubj', 'dobj']:
                        tiene_argumentos = True
                        break
                if not tiene_argumentos:
                    tiene_sentido = False
                    
        return tiene_sentido

    def _generar_pregunta_vf(self, concepto: Dict, texto: str) -> Dict:
        """Genera una pregunta de verdadero/falso analítica"""
        try:
            # Construir prompt para generar afirmación
            prompt = f"""
            Contexto: {concepto['oracion']}
            
            Genera una afirmación sobre este texto que:
            1. Sea específica y verificable
            2. No sea trivialmente verdadera o falsa
            3. Requiera analizar la información del texto
            4. Se relacione con los conceptos principales
            5. Sea clara y bien redactada
            """
            
            inputs = self.qa_tokenizer(prompt, return_tensors="pt", max_length=512, 
                                   truncation=True)
            outputs = self.qa_model.generate(
                inputs.input_ids,
                max_length=100,
                temperature=0.8,
                num_beams=4,
                do_sample=True
            )
            
            afirmacion = self.qa_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Verificar veracidad
            resultado = self.verifier(
                question=afirmacion,
                context=texto
            )
            
            es_verdadero = resultado['score'] > 0.7
            
            # Validar calidad de la afirmación
            if self._validar_afirmacion_vf(afirmacion, texto):
                return {
                    "pregunta": afirmacion,
                    "opciones": ["Verdadero", "Falso"],
                    "respuesta_correcta": "Verdadero" if es_verdadero else "Falso",
                    "tipo": "verdadero_falso"
                }
            
            return None
            
        except Exception as e:
            print(f"Error generando pregunta V/F: {str(e)}")
            return None

    def _validar_afirmacion_vf(self, afirmacion: str, texto: str) -> bool:
        """Valida que una afirmación V/F sea apropiada"""
        if not afirmacion or len(afirmacion.split()) < 5:
            return False
            
        doc = self.nlp(afirmacion)
        
        # Verificar estructura gramatical
        tiene_sujeto = False
        tiene_verbo = False
        for token in doc:
            if token.dep_ == 'nsubj':
                tiene_sujeto = True
            elif token.pos_ == 'VERB':
                tiene_verbo = True
                
        if not (tiene_sujeto and tiene_verbo):
            return False
            
        # Verificar relación con el texto
        similitud = doc.similarity(self.nlp(texto))
        if similitud < 0.3:  # Poco relacionada
            return False
            
        return True

    def generar_cuestionario(self, ruta_archivo: str) -> Dict:
        """Genera el cuestionario completo"""
        try:
            with open(ruta_archivo, 'r', encoding='utf-8') as archivo:
                datos = json.load(archivo)
        except Exception as e:
            print(f"Error al cargar archivo: {str(e)}")
            return None

        cuestionarios = {}
        
        for item in datos['quiz']:
            materia = item['materia']
            texto = item['texto']
            
            if materia not in cuestionarios:
                cuestionarios[materia] = []
            
            # Analizar texto y extraer conceptos
            conceptos = self._analizar_texto(texto)
            if not conceptos:
                print(f"No se pudieron extraer conceptos de {materia}")
                continue
            
            preguntas = []
            intentos = 0
            max_intentos = 20
            
            # Generar preguntas de alternativas
            while len([p for p in preguntas if p['tipo'] == 'alternativas']) < 7 and intentos < max_intentos:
                concepto = random.choice(conceptos)
                pregunta = self._generar_pregunta(concepto, texto)
                if pregunta and self._validar_pregunta(pregunta, preguntas):
                    preguntas.append(pregunta)
                intentos += 1
            
            # Generar preguntas V/F
            intentos = 0
            while len([p for p in preguntas if p['tipo'] == 'verdadero_falso']) < 3 and intentos < max_intentos:
                concepto = random.choice(conceptos)
                pregunta = self._generar_pregunta_vf(concepto, texto)
                if pregunta and self._validar_pregunta(pregunta, preguntas):
                    preguntas.append(pregunta)
                intentos += 1
            
            # Verificar calidad final
            preguntas_validas = [p for p in preguntas if self._validar_ortografia(p)]
            
            if len(preguntas_validas) >= 8:
                random.shuffle(preguntas_validas)
                cuestionarios[materia].extend(preguntas_validas[:10])
            else:
                print(f"No se generaron suficientes preguntas válidas para {materia}")
        
        return cuestionarios

    def _validar_pregunta(self, pregunta: Dict, existentes: List[Dict]) -> bool:
        """Valida una pregunta completa"""
        if not pregunta:
            return False
            
        # Verificar duplicados aproximados
        doc_pregunta = self.nlp(pregunta['pregunta'])
        for p in existentes:
            if self.nlp(p['pregunta']).similarity(doc_pregunta) > 0.7:
                return False
        
        # Validar estructura
        doc = self.nlp(pregunta['pregunta'])
        
        # Debe tener estructura gramatical básica
        tiene_verbo = False
        tiene_sujeto = False
        for token in doc:
            if token.pos_ == 'VERB':
                tiene_verbo = True
            elif token.dep_ == 'nsubj':
                tiene_sujeto = True
                
        if not (tiene_verbo and tiene_sujeto):
            return False
            
        # Verificar longitud adecuada
        palabras = len(pregunta['pregunta'].split())
        if palabras < 5 or palabras > 50:
            return False
            
        # Para preguntas de alternativas
        if pregunta['tipo'] == 'alternativas':
            # Verificar cantidad de opciones
            if len(pregunta['opciones']) != 4:
                return False
                
            # Verificar que la respuesta correcta esté en las opciones
            if pregunta['respuesta_correcta'] not in pregunta['opciones']:
                return False
                
            # Verificar coherencia de alternativas
            for opcion in pregunta['opciones']:
                if len(opcion.split()) < 2:  # Muy corta
                    return False
                    
                # Verificar relación semántica con la pregunta
                similitud = self.nlp(opcion).similarity(doc_pregunta)
                if similitud < 0.2:  # Muy poco relacionada
                    return False
        
        # Para preguntas V/F
        elif pregunta['tipo'] == 'verdadero_falso':
            # Verificar opciones correctas
            if set(pregunta['opciones']) != {"Verdadero", "Falso"}:
                return False
                
            # Verificar respuesta válida
            if pregunta['respuesta_correcta'] not in ["Verdadero", "Falso"]:
                return False
        
        return True

    def _validar_ortografia(self, pregunta: Dict) -> bool:
        """Valida la ortografía y formato de una pregunta"""
        textos_a_validar = [pregunta['pregunta']]
        
        if pregunta['tipo'] == 'alternativas':
            textos_a_validar.extend(pregunta['opciones'])
        
        for texto in textos_a_validar:
            # Verificar signos de puntuación básicos
            if texto.count('(') != texto.count(')'):
                return False
                
            # Verificar espacios después de signos de puntuación
            for signo in ['.', ',', ';', ':']:
                if f'{signo} ' not in texto and signo in texto:
                    return False
                    
            # Verificar mayúsculas al inicio
            if not texto[0].isupper():
                return False
                
            # Verificar espacios múltiples
            if '  ' in texto:
                return False
                
            # Verificar signos de interrogación (solo para la pregunta)
            if texto == pregunta['pregunta']:
                if pregunta['tipo'] == 'alternativas':
                    if not (texto.startswith('¿') and texto.endswith('?')):
                        return False
                else:  # V/F
                    if texto.startswith('¿') or texto.endswith('?'):
                        return False
            
            # Verificar palabras básicas
            doc = self.nlp(texto)
            for token in doc:
                # Verificar palabras muy cortas que no sean artículos o preposiciones
                if (len(token.text) == 1 and 
                    token.pos_ not in ['DET', 'ADP'] and 
                    token.text.lower() not in ['y', 'o', 'a']):
                    return False
                    
                # Verificar palabras sin sentido (tokens no reconocidos)
                if (token.pos_ == 'X' and 
                    not token.like_num and 
                    not token.like_url and 
                    not token.like_email):
                    return False
        
        return True