# Generador de Quizzes
# por Franco Benassi
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
    pipeline,
    BartTokenizer,
    BartForConditionalGeneration
)
import spacy
import json
import random
from typing import List, Dict, Any
import numpy as np

class GeneradorCuestionarios:
    def __init__(self):
        print("Inicializando modelos...")
        
        # Modelo BART multilingüe para mejor comprensión del contexto
        self.qa_model_name = "facebook/mbart-large-cc25"
        self.qa_tokenizer = AutoTokenizer.from_pretrained(self.qa_model_name)
        self.qa_model = AutoModelForSeq2SeqLM.from_pretrained(self.qa_model_name)
        
        # Pipeline QA específico para español
        self.qa_pipeline = pipeline(
            "question-answering",
            model="mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es",
            tokenizer="mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"
        )
        
        # Modelo para generar alternativas plausibles
        self.alt_model_name = "facebook/bart-large-mnli"
        self.alt_tokenizer = BartTokenizer.from_pretrained(self.alt_model_name)
        self.alt_model = BartForConditionalGeneration.from_pretrained(self.alt_model_name)
        
        # SpaCy para análisis semántico en español
        self.nlp = spacy.load("es_core_news_lg")
        
        # Templates para generación de preguntas
        self.templates_preguntas = [
            "¿Cuál es la principal razón por la que {tema}?",
            "¿Qué impacto tiene {tema} en {contexto}?",
            "¿Cómo se relaciona {tema} con {concepto}?",
            "¿Por qué es importante {tema} en el contexto de {contexto}?",
            "¿Qué diferencia existe entre {tema} y {concepto}?",
            "¿Qué factores influyen en {tema}?",
            "¿Cuáles son las consecuencias de {tema}?",
            "¿Qué papel desempeña {tema} en {contexto}?"
        ]

    def _extraer_conceptos_clave(self, texto: str) -> List[Dict]:
        doc = self.nlp(texto)
        conceptos = []
        
        for sent in doc.sents:
            # Identificar oraciones relevantes
            if len(sent.text.split()) > 8:
                # Extraer entidades nombradas
                entidades = [ent.text for ent in sent.ents 
                           if ent.label_ in ['ORG', 'PERSON', 'LOC', 'EVENT', 'MISC']]
                
                # Extraer sintagmas nominales importantes
                noun_chunks = [chunk.text for chunk in sent.noun_chunks 
                             if len(chunk.text.split()) > 1]
                
                # Identificar verbos principales y sus argumentos
                verbos = []
                for token in sent:
                    if token.pos_ == 'VERB':
                        argumentos = [child.text for child in token.children 
                                    if child.dep_ in ['nsubj', 'dobj', 'iobj']]
                        if argumentos:
                            verbos.append({
                                'verbo': token.text,
                                'argumentos': argumentos
                            })
                
                if entidades or noun_chunks or verbos:
                    conceptos.append({
                        'oracion': sent.text,
                        'entidades': entidades,
                        'conceptos': noun_chunks,
                        'verbos': verbos
                    })
        
        return conceptos

    def _generar_pregunta(self, concepto: Dict, contexto: str) -> Dict:
        try:
            # Seleccionar tema principal
            tema = random.choice(concepto['conceptos']) if concepto['conceptos'] else \
                  random.choice(concepto['entidades']) if concepto['entidades'] else \
                  concepto['oracion']
            
            # Generar pregunta usando el pipeline QA
            pregunta_context = f"""
            Contexto: {concepto['oracion']}
            Genera una pregunta de análisis sobre: {tema}
            """
            
            # Usar el modelo para generar la pregunta
            inputs = self.qa_tokenizer(pregunta_context, 
                                     return_tensors="pt", 
                                     max_length=512, 
                                     truncation=True)
            
            outputs = self.qa_model.generate(
                inputs["input_ids"],
                max_length=150,
                num_beams=5,
                length_penalty=2.0,
                temperature=0.7,
                do_sample=True
            )
            
            pregunta = self.qa_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Obtener respuesta correcta
            respuesta = self.qa_pipeline(
                question=pregunta,
                context=contexto
            )
            
            # Generar alternativas plausibles
            alternativas = self._generar_alternativas(
                pregunta,
                respuesta['answer'],
                concepto,
                contexto
            )
            
            if alternativas and len(alternativas) >= 4:
                return {
                    'pregunta': self._formatear_pregunta(pregunta),
                    'opciones': alternativas[:4],
                    'respuesta_correcta': respuesta['answer'],
                    'tipo': 'alternativas'
                }
            
        except Exception as e:
            print(f"Error generando pregunta: {str(e)}")
        
        return None

    def _generar_alternativas(self, pregunta: str, respuesta: str, 
                            concepto: Dict, contexto: str) -> List[str]:
        alternativas = [respuesta]
        
        try:
            # Analizar la pregunta y respuesta
            doc_pregunta = self.nlp(pregunta)
            doc_respuesta = self.nlp(respuesta)
            
            # Identificar el tipo de respuesta esperada
            tipo_respuesta = self._identificar_tipo_respuesta(doc_respuesta)
            
            # Generar alternativas basadas en el tipo de respuesta
            for _ in range(5):
                prompt = f"""
                Pregunta: {pregunta}
                Respuesta correcta: {respuesta}
                Genera una respuesta alternativa que sea:
                1. Del mismo tipo que la respuesta correcta
                2. Relacionada con el tema
                3. Plausible pero incorrecta
                4. Similar en longitud y estructura
                """
                
                inputs = self.alt_tokenizer(prompt, 
                                          return_tensors="pt", 
                                          max_length=512, 
                                          truncation=True)
                
                outputs = self.alt_model.generate(
                    inputs["input_ids"],
                    max_length=100,
                    temperature=0.8,
                    do_sample=True
                )
                
                alternativa = self.alt_tokenizer.decode(outputs[0], 
                                                      skip_special_tokens=True)
                
                if self._validar_alternativa(alternativa, alternativas, 
                                           tipo_respuesta, pregunta):
                    alternativas.append(alternativa)
                
                if len(alternativas) >= 4:
                    break
            
            random.shuffle(alternativas)
            return alternativas
            
        except Exception as e:
            print(f"Error generando alternativas: {str(e)}")
            return []

    def _identificar_tipo_respuesta(self, doc) -> str:
        """Identifica el tipo semántico de la respuesta"""
        # Analizar entidades nombradas
        if doc.ents:
            return doc.ents[0].label_
            
        # Analizar estructura sintáctica
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN']:
                return token.pos_
            elif token.pos_ == 'VERB':
                return 'ACCION'
                
        return 'OTRO'

    def _validar_alternativa(self, alternativa: str, existentes: List[str],
                            tipo_respuesta: str, pregunta: str) -> bool:
        if not alternativa or len(alternativa.split()) < 2:
            return False
            
        doc_alt = self.nlp(alternativa)
        
        # Verificar tipo semántico
        tipo_alt = self._identificar_tipo_respuesta(doc_alt)
        if tipo_alt != tipo_respuesta:
            return False
            
        # Verificar similitud con existentes
        for existente in existentes:
            doc_existente = self.nlp(existente)
            if doc_alt.similarity(doc_existente) > 0.7:
                return False
                
        # Verificar coherencia con la pregunta
        doc_pregunta = self.nlp(pregunta)
        similitud_pregunta = doc_alt.similarity(doc_pregunta)
        if similitud_pregunta < 0.3:
            return False
            
        return True

    def _generar_vf(self, concepto: Dict, contexto: str) -> Dict:
        try:
            # Generar afirmación
            prompt = f"""
            Contexto: {concepto['oracion']}
            Genera una afirmación que:
            1. Se base en el contexto
            2. Sea específica y verificable
            3. No sea obviamente verdadera o falsa
            """
            
            inputs = self.qa_tokenizer(prompt, 
                                     return_tensors="pt", 
                                     max_length=512, 
                                     truncation=True)
            
            outputs = self.qa_model.generate(
                inputs["input_ids"],
                max_length=100,
                temperature=0.7,
                num_beams=4
            )
            
            afirmacion = self.qa_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Verificar veracidad
            resultado = self.qa_pipeline(
                question=afirmacion,
                context=contexto
            )
            
            es_verdadero = resultado['score'] > 0.7
            
            if self._validar_afirmacion_vf(afirmacion, contexto):
                return {
                    "pregunta": self._formatear_afirmacion(afirmacion),
                    "opciones": ["Verdadero", "Falso"],
                    "respuesta_correcta": "Verdadero" if es_verdadero else "Falso",
                    "tipo": "verdadero_falso"
                }
                
        except Exception as e:
            print(f"Error generando pregunta V/F: {str(e)}")
            
        return None

    def _validar_afirmacion_vf(self, afirmacion: str, contexto: str) -> bool:
        if not afirmacion or len(afirmacion.split()) < 5:
            return False
            
        doc = self.nlp(afirmacion)
        
        # Verificar estructura completa
        tiene_sujeto = False
        tiene_verbo = False
        tiene_objeto = False
        
        for token in doc:
            if token.dep_ == 'nsubj':
                tiene_sujeto = True
            elif token.pos_ == 'VERB':
                tiene_verbo = True
            elif token.dep_ in ['dobj', 'attr', 'acomp']:
                tiene_objeto = True
                
        if not (tiene_sujeto and tiene_verbo and tiene_objeto):
            return False
            
        # Verificar relación con contexto
        doc_contexto = self.nlp(contexto)
        similitud = doc.similarity(doc_contexto)
        return similitud >= 0.4

    def _formatear_pregunta(self, pregunta: str) -> str:
        """Formatea la pregunta para asegurar estructura correcta"""
        pregunta = pregunta.strip()
        if not pregunta.startswith('¿'):
            pregunta = '¿' + pregunta
        if not pregunta.endswith('?'):
            pregunta = pregunta + '?'
        return pregunta

    def _formatear_afirmacion(self, afirmacion: str) -> str:
        """Formatea la afirmación para asegurar estructura correcta"""
        afirmacion = afirmacion.strip()
        if not afirmacion[0].isupper():
            afirmacion = afirmacion[0].upper() + afirmacion[1:]
        if not afirmacion.endswith('.'):
            afirmacion = afirmacion + '.'
        return afirmacion

    def generar_cuestionario(self, ruta_archivo: str) -> Dict:
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
            
            # Extraer conceptos clave
            conceptos = self._extraer_conceptos_clave(texto)
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
                if pregunta and self._validar_pregunta_final(pregunta, preguntas):
                    preguntas.append(pregunta)
                intentos += 1
            
            # Generar preguntas V/F
            intentos = 0
            while len([p for p in preguntas if p['tipo'] == 'verdadero_falso']) < 3 and intentos < max_intentos:
                concepto = random.choice(conceptos)
                pregunta = self._generar_vf(concepto, texto)
                if pregunta and self._validar_pregunta_final(pregunta, preguntas):
                    preguntas.append(pregunta)
                intentos += 1
            
            if len(preguntas) >= 8:
                random.shuffle(preguntas)
                cuestionarios[materia].extend(preguntas[:10])
            else:
                print(f"No se generaron suficientes preguntas válidas para {materia}")
        
        return cuestionarios

    def _validar_pregunta_final(self, pregunta: Dict, existentes: List[Dict]) -> bool:
        """Validación final de calidad y coherencia de la pregunta"""
        if not pregunta or 'pregunta' not in pregunta:
            return False
            
        # Validar formato y estructura
        texto_pregunta = pregunta['pregunta']
        doc = self.nlp(texto_pregunta)
        
        # Verificar longitud apropiada
        palabras = len(texto_pregunta.split())
        if palabras < 5 or palabras > 50:
            return False
            
        # Verificar estructura sintáctica
        tiene_estructura_valida = False
        for token in doc:
            if token.pos_ == 'VERB' and token.dep_ == 'ROOT':
                for child in token.children:
                    if child.dep_ in ['nsubj', 'dobj', 'attr']:
                        tiene_estructura_valida = True
                        break
                        
        if not tiene_estructura_valida:
            return False
            
        # Validar según tipo de pregunta
        if pregunta['tipo'] == 'alternativas':
            # Verificar número y calidad de opciones
            if len(pregunta['opciones']) != 4:
                return False
                
            # Verificar respuesta correcta
            if pregunta['respuesta_correcta'] not in pregunta['opciones']:
                return False
                
            # Validar coherencia de alternativas
            alternativas_validas = 0
            for opcion in pregunta['opciones']:
                doc_opcion = self.nlp(opcion)
                
                # Verificar longitud mínima
                if len(opcion.split()) < 2:
                    continue
                    
                # Verificar relación semántica
                similitud = doc_opcion.similarity(doc)
                if similitud >= 0.3:  # Umbral de similitud mínima
                    alternativas_validas += 1
                    
                # Verificar estructura gramatical
                tiene_sentido = False
                for token in doc_opcion:
                    if token.pos_ in ['NOUN', 'VERB', 'ADJ']:
                        tiene_sentido = True
                        break
                        
                if not tiene_sentido:
                    return False
            
            if alternativas_validas < 4:
                return False
                
        elif pregunta['tipo'] == 'verdadero_falso':
            # Verificar formato correcto
            if not texto_pregunta[0].isupper() or not texto_pregunta.endswith('.'):
                return False
                
            # Verificar opciones válidas
            if set(pregunta['opciones']) != {"Verdadero", "Falso"}:
                return False
                
            # Verificar respuesta válida
            if pregunta['respuesta_correcta'] not in ["Verdadero", "Falso"]:
                return False
        
        # Verificar duplicados o similares
        for existente in existentes:
            doc_existente = self.nlp(existente['pregunta'])
            if doc.similarity(doc_existente) > 0.7:
                return False
        
        # Verificar coherencia general
        tiene_coherencia = True
        for token in doc:
            # Verificar dependencias válidas
            if token.dep_ == 'ROOT':
                tiene_argumentos = False
                for child in token.children:
                    if child.dep_ in ['nsubj', 'dobj', 'ccomp', 'xcomp']:
                        tiene_argumentos = True
                        break
                tiene_coherencia = tiene_coherencia and tiene_argumentos
        
        return tiene_coherencia

    def _limpiar_texto(self, texto: str) -> str:
        """Limpia y normaliza el texto"""
        # Eliminar espacios múltiples
        texto = ' '.join(texto.split())
        
        # Corregir espacios en signos de puntuación
        signos = {'.': '. ', ',': ', ', ';': '; ', ':': ': ', 
                 '?': '? ', '¿': '¿', '!': '! ', '¡': '¡'}
        
        for signo, reemplazo in signos.items():
            texto = texto.replace(f' {signo}', signo)
            texto = texto.replace(signo, reemplazo)
        
        # Asegurar mayúscula inicial
        texto = texto[0].upper() + texto[1:]
        
        # Corregir espacios múltiples después de normalización
        texto = ' '.join(texto.split())
        
        return texto.strip()

    def _validar_ortografia(self, texto: str) -> bool:
        """Valida la ortografía básica del texto"""
        doc = self.nlp(texto)
        
        # Verificar mayúsculas después de punto
        oraciones = [sent.text.strip() for sent in doc.sents]
        for oracion in oraciones:
            if oracion and not oracion[0].isupper():
                return False
        
        # Verificar signos de interrogación/exclamación en español
        if '?' in texto and '¿' not in texto:
            return False
        if '!' in texto and '¡' not in texto:
            return False
        
        return True