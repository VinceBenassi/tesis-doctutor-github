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
from typing import List, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import gc
import time
from tqdm import tqdm

class GeneradorCuestionarios:
    def __init__(self):
        print("Inicializando modelos...")

        # Configuración de device y memoria
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Inicialización directa de modelos
        print("Cargando modelo QA...")
        self._qa_model = pipeline(
            "question-answering",
            model="mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es",
            device=self.device,
            batch_size=1
        )
        
        print("Cargando modelo T5...")
        self._t5_model = T5ForConditionalGeneration.from_pretrained(
            "google/flan-t5-large",
            device_map=self.device
        )
        self._t5_tokenizer = AutoTokenizer.from_pretrained(
            "google/flan-t5-large",
            model_max_length=512
        )
        
        print("Cargando tokenizer BERT...")
        self._bert_tokenizer = AutoTokenizer.from_pretrained(
            "dccuchile/bert-base-spanish-wwm-cased"
        )
        
        print("Cargando modelo BERT...")
        self._bert_model = AutoModelForSequenceClassification.from_pretrained(
            "dccuchile/bert-base-spanish-wwm-cased",
            num_labels=2
        ).to(self.device)
        
        print("Cargando modelo spaCy...")
        self._nlp = spacy.load("es_core_news_lg")
        
        # Configuración 
        self.max_preguntas = 10
        self.min_similitud = 0.3
        self.timeout_global = 1800  # 30 minutos máximo por texto
        self.timeout_pregunta = 300  # 5 minutos por pregunta
        
        # Parámetros de generación
        self.generation_config = {
            'do_sample': True,
            'temperature': 0.9,
            'top_p': 0.95,
            'top_k': 50,
            'max_new_tokens': 128,
            'num_beams': 5,
            'num_return_sequences': 1,
            'no_repeat_ngram_size': 2,
            'early_stopping': True
        }

    @property
    def qa_model(self):
        return self._qa_model

    @property
    def t5_tokenizer(self):
        return self._t5_tokenizer
    
    @property
    def t5_model(self):
        return self._t5_model

    @property
    def bert_tokenizer(self):
        return self._bert_tokenizer

    @property
    def bert_model(self):
        return self._bert_model
    
    @property
    def nlp(self):
        return self._nlp

    def _limpiar_memoria(self):
        """Limpia la memoria GPU/CPU"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
    def _extraer_informacion_relevante(self, texto: str) -> List[Dict]:
        """Extrae información relevante del texto usando análisis lingüístico"""
        doc = self.nlp(texto)
        informacion = []
        
        # Extraer oraciones significativas
        for sent in doc.sents:
            if len(sent.text.split()) >= 10:
                # Analizar la oración
                analisis = {
                    'texto': sent.text,
                    'entidades': [(ent.text, ent.label_) for ent in sent.ents],
                    'conceptos': [chunk.text for chunk in sent.noun_chunks],
                    'importancia': sum(token.is_stop is False for token in sent)
                }
                informacion.append(analisis)
        
        return sorted(informacion, key=lambda x: x['importancia'], reverse=True)

    def _generar_pregunta_analisis_mejorada(self, info: Dict, timeout: float) -> Optional[Dict]:
        """Genera una pregunta de análisis con mejor control"""
        try:
            if time.time() > timeout:
                return None
                
            # Extraer conceptos clave
            conceptos = self._extraer_conceptos_clave(self.nlp(info['texto']))
            if not conceptos:
                return None
                
            # Generar pregunta en español
            concepto = random.choice(conceptos)
            prompts = [
                f"Genera una pregunta analítica en español sobre: {concepto}. Contexto: {info['texto']}",
                f"Crea una pregunta que analice el siguiente concepto en español: {concepto}. Texto: {info['texto']}",
                f"Formula una pregunta que evalúe la comprensión de {concepto} en español. Información: {info['texto']}"
            ]
            
            pregunta_generada = self._generar_texto_t5(
                random.choice(prompts),
                temperatura=0.8,
                num_beams=5
            )
            
            if not pregunta_generada:
                return None
                
            # Obtener respuesta correcta
            qa_result = self.qa_model(
                question=pregunta_generada,
                context=info['texto']
            )
            
            if qa_result['score'] < 0.7:
                return None
                
            respuesta_correcta = qa_result['answer']
            
            # Generar alternativas coherentes
            alternativas = self._generar_alternativas_semanticas(
                respuesta_correcta,
                concepto,
                info['texto']
            )
            
            if len(alternativas) < 3:
                return None
                
            # Formar pregunta completa
            opciones = alternativas + [respuesta_correcta]
            random.shuffle(opciones)
            
            return {
                'pregunta': pregunta_generada,
                'opciones': opciones,
                'respuesta_correcta': respuesta_correcta,
                'tipo': 'alternativas'
            }
            
        except Exception as e:
            print(f"Error en pregunta analítica: {str(e)}")
            return None

    def _generar_texto_t5(self, prompt: str, temperatura: float = 0.8, num_beams: int = 4) -> Optional[str]:
        """Genera texto usando T5 con mejor control"""
        try:
            inputs = self.t5_tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)
            
            outputs = self.t5_model.generate(
                **inputs,
                max_length=128,
                num_beams=num_beams,
                do_sample=True,
                temperature=temperatura,
                top_p=0.95,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
            
            return self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        except Exception as e:
            print(f"Error generando texto: {str(e)}")
            return None


    def _extraer_conceptos_clave(self, doc) -> List[str]:
        """Extrae conceptos clave usando análisis semántico"""
        conceptos = []
        
        # Identificar sintagmas nominales importantes
        for chunk in doc.noun_chunks:
            if (chunk.root.pos_ in ['NOUN', 'PROPN'] and 
                not chunk.root.is_stop and 
                len(chunk.text.split()) <= 3):
                
                # Calcular importancia usando TF-IDF
                importancia = sum(not token.is_stop for token in chunk)
                if importancia > 1:
                    conceptos.append(chunk.text)
        
        # Encontrar entidades nombradas relevantes
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT', 'EVENT', 'CONCEPT']:
                conceptos.append(ent.text)
        
        return list(set(conceptos))

    def _generar_alternativas_semanticas(self, respuesta: str, concepto: str, contexto: str) -> List[str]:
        """Genera alternativas semánticamente relacionadas y coherentes"""
        try:
            alternativas = set()
            doc_respuesta = self.nlp(respuesta)
            doc_concepto = self.nlp(concepto)
            doc_contexto = self.nlp(contexto)
            
            # 1. Extraer información relacionada del contexto
            oraciones_relevantes = []
            for sent in doc_contexto.sents:
                if (concepto.lower() in sent.text.lower() or
                    doc_concepto.similarity(self.nlp(sent.text)) > 0.5):
                    oraciones_relevantes.append(sent.text)
            
            # 2. Generar alternativas usando el contexto y T5
            prompts_alternativos = [
                f"generate incorrect but plausible answer about {concepto} in Spanish: {respuesta}",
                f"what could be mistaken for {concepto} in Spanish: {respuesta}",
                f"alternative explanation for {concepto} in Spanish: {respuesta}",
                f"similar but different answer about {concepto} in Spanish: {respuesta}"
            ]
            
            # Intentar generar alternativas usando diferentes prompts
            for prompt in prompts_alternativos:
                if len(alternativas) >= 3:
                    break
                    
                try:
                    inputs = self.t5_tokenizer(prompt, 
                        return_tensors="pt", 
                        max_length=512, 
                        truncation=True
                    ).to(self.device)
                    
                    outputs = self.t5_model.generate(
                        **inputs,
                        max_length=64,
                        num_beams=5,
                        no_repeat_ngram_size=2,
                        do_sample=True,
                        temperature=0.9,
                        top_p=0.95
                    )
                    
                    alternativa = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # Validar y limpiar la alternativa
                    if alternativa and self._validar_alternativa_compleja(
                        alternativa, respuesta, concepto, contexto):
                        alternativas.add(alternativa)
                    
                except Exception as e:
                    print(f"Error generando alternativa: {str(e)}")
                    continue
            
            # 3. Si no hay suficientes alternativas, usar información del contexto
            if len(alternativas) < 3:
                for sent in oraciones_relevantes:
                    doc_sent = self.nlp(sent)
                    for chunk in doc_sent.noun_chunks:
                        if len(alternativas) >= 3:
                            break
                        
                        if (chunk.text != respuesta and 
                            chunk.text not in alternativas and
                            len(chunk.text.split()) >= 2):
                            
                            # Intentar generar una respuesta completa basada en el chunk
                            prompt = f"generate complete answer using '{chunk.text}' about {concepto} in Spanish"
                            inputs = self.t5_tokenizer(prompt, return_tensors="pt").to(self.device)
                            outputs = self.t5_model.generate(
                                **inputs,
                                max_length=64,
                                do_sample=True,
                                temperature=0.8
                            )
                            
                            alternativa = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
                            if self._validar_alternativa_compleja(
                                alternativa, respuesta, concepto, contexto):
                                alternativas.add(alternativa)
            
            # 4. Asegurar que tenemos suficientes alternativas
            alternativas_lista = list(alternativas)
            while len(alternativas_lista) < 3:
                alternativa = self._generar_alternativa_respaldo(respuesta, concepto, contexto)
                if alternativa and alternativa not in alternativas_lista:
                    alternativas_lista.append(alternativa)
            
            return alternativas_lista[:3]
        
        except Exception as e:
            print(f"Error en generación de alternativas semánticas: {str(e)}")
            return self._generar_alternativas_simples(respuesta, contexto)


    def _validar_alternativa_compleja(self, alternativa: str, respuesta: str, 
    concepto: str, contexto: str) -> bool:
        """Realiza una validación más completa de las alternativas"""
        try:
            if not alternativa or not respuesta or alternativa.lower() == respuesta.lower():
                return False
            
            doc_alt = self.nlp(alternativa)
            doc_resp = self.nlp(respuesta)
            doc_concepto = self.nlp(concepto)
            
            # Validaciones básicas
            if len(doc_alt) < 3 or len(alternativa.split()) > len(respuesta.split()) * 2:
                return False
            
            # Verificar coherencia gramatical
            if not any(token.pos_ == 'VERB' for token in doc_alt):
                return False
            
            # Verificar similitud semántica
            similitud_concepto = doc_alt.similarity(doc_concepto)
            similitud_respuesta = doc_alt.similarity(doc_resp)
            
            if similitud_concepto < 0.3 or similitud_respuesta > 0.8 or similitud_respuesta < 0.2:
                return False
            
            # Verificar que la alternativa forme una frase completa
            if not any(token.dep_ == 'ROOT' for token in doc_alt):
                return False
            
            return True
            
        except Exception as e:
            print(f"Error en validación compleja: {str(e)}")
            return False

    def _generar_alternativa_respaldo(self, respuesta: str, concepto: str, contexto: str) -> Optional[str]:
        """Genera una alternativa de respaldo cuando otros métodos fallan"""
        try:
            # Intentar generar usando variaciones del concepto
            doc_concepto = self.nlp(concepto)
            variaciones = []
            
            # Buscar términos relacionados en el contexto
            doc_contexto = self.nlp(contexto)
            for token in doc_contexto:
                if (token.pos_ in ['NOUN', 'PROPN', 'VERB'] and 
                    0.3 < token.similarity(doc_concepto) < 0.7):
                    variaciones.append(token.text)
            
            if variaciones:
                variacion = random.choice(variaciones)
                prompt = f"generate complete answer about {variacion} in Spanish"
                inputs = self.t5_tokenizer(prompt, return_tensors="pt").to(self.device)
                outputs = self.t5_model.generate(
                    **inputs,
                    max_length=64,
                    do_sample=True,
                    temperature=0.9
                )
                
                return self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
                
            return None
            
        except Exception as e:
            print(f"Error en generación de respaldo: {str(e)}")
            return None

    def _generar_alternativas_coherentes(self, respuesta: str, contexto: str, entidades: List) -> List[str]:
        """Genera alternativas coherentes basadas en el contexto y entidades"""
        try:
            alternativas = set()
            doc_respuesta = self.nlp(respuesta)
            
            # Si la respuesta no tiene vector válido, usar enfoque alternativo
            if not doc_respuesta.vector.any():
                return self._generar_alternativas_simples(respuesta, contexto)
            
            vector_respuesta = doc_respuesta.vector.reshape(1, -1)  # Asegurar 2D
            
            # 1. Primero intentar con entidades nombradas
            if doc_respuesta.ents:
                tipo_entidad = doc_respuesta.ents[0].label_
                for ent, label in entidades:
                    if label == tipo_entidad and ent != respuesta:
                        alternativas.add(ent)
            
            # 2. Intentar con sintagmas nominales
            if len(alternativas) < 3:
                doc_contexto = self.nlp(contexto)
                for chunk in doc_contexto.noun_chunks:
                    if (chunk.text != respuesta and 
                        len(chunk.text) > 3 and 
                        chunk.root.pos_ in ['NOUN', 'PROPN']):
                        # Calcular similitud de manera segura
                        vector_chunk = chunk.vector.reshape(1, -1)
                        similitud = cosine_similarity(vector_chunk, vector_respuesta)[0][0]
                        
                        if 0.3 <= similitud <= 0.7:
                            alternativas.add(chunk.text)
                            if len(alternativas) >= 3:
                                break
            
            # 3. Usar T5 para generar alternativas faltantes
            intentos = 0
            while len(alternativas) < 3 and intentos < 5:
                alt_generada = self._generar_alternativa_t5(respuesta, contexto)
                if alt_generada and alt_generada not in alternativas:
                    if self._validar_alternativa(alt_generada, respuesta, contexto):
                        alternativas.add(alt_generada)
                intentos += 1
            
            # Si aún necesitamos más alternativas, usar el método simple
            if len(alternativas) < 3:
                alternativas_simples = self._generar_alternativas_simples(respuesta, contexto)
                alternativas.update(alternativas_simples)
            
            alternativas_lista = list(alternativas)[:3]
            # Asegurar que tenemos exactamente 3 alternativas
            while len(alternativas_lista) < 3:
                alternativas_lista.append(f"Opción adicional {len(alternativas_lista) + 1}")
            
            return alternativas_lista
            
        except Exception as e:
            print(f"Error generando alternativas: {str(e)}")
            return [f"Opción {i}" for i in range(1, 4)]

   
    def _generar_alternativas_simples(self, respuesta: str, contexto: str) -> List[str]:
        """Método de respaldo para generar alternativas cuando los métodos principales fallan"""
        try:
            alternativas = set()
            doc_contexto = self.nlp(contexto)
            
            # Extraer sustantivos y nombres propios como alternativas
            for token in doc_contexto:
                if len(alternativas) >= 3:
                    break
                if (token.pos_ in ['NOUN', 'PROPN'] and 
                    token.text != respuesta and 
                    len(token.text) > 3):
                    alternativas.add(token.text)
            
            # Si aún necesitamos más alternativas, usar chunks
            if len(alternativas) < 3:
                for chunk in doc_contexto.noun_chunks:
                    if len(alternativas) >= 3:
                        break
                    if (chunk.text != respuesta and 
                        len(chunk.text) > 3 and 
                        chunk.root.pos_ in ['NOUN', 'PROPN']):
                        alternativas.add(chunk.text)
            
            # Asegurar que tenemos suficientes alternativas
            while len(alternativas) < 3:
                alt_t5 = self._generar_alternativa_t5(respuesta, contexto)
                if alt_t5 and alt_t5 not in alternativas:
                    alternativas.add(alt_t5)
            
            return list(alternativas)[:3]
            
        except Exception as e:
            print(f"Error en alternativas simples: {str(e)}")
            return [f"Opción {i}" for i in range(1, 4)]  # Último recurso
        
    def _generar_alternativa_t5(self, respuesta: str, contexto: str) -> Optional[str]:
        """Genera una alternativa usando T5"""
        try:
            # Crear prompt para generar alternativa similar pero diferente
            prompt = f"generate alternative similar to but different from: {respuesta}"
            inputs = self.t5_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            
            outputs = self.t5_model.generate(
                **inputs,
                max_length=64,
                num_beams=4,
                no_repeat_ngram_size=2,
                do_sample=True,
                temperature=0.7
            )
            
            alternativa = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Validar que la alternativa sea diferente y coherente
            if (alternativa and 
                alternativa != respuesta and 
                len(alternativa.split()) <= len(respuesta.split()) + 2):
                return alternativa
                
            return None
            
        except Exception as e:
            print(f"Error generando alternativa T5: {str(e)}")
            return None

    def _validar_alternativa(self, alternativa: str, respuesta: str, contexto: str) -> bool:
        """Valida la coherencia de una alternativa usando múltiples métricas"""
        try:
            if not alternativa or not respuesta:
                return False
            
            # Convertir a minúsculas para comparación
            alt_lower = alternativa.lower().strip()
            resp_lower = respuesta.lower().strip()
            
            # Evitar alternativas idénticas o muy similares
            if alt_lower == resp_lower:
                return False
                
            # Validar longitud similar (permitir cierta variación)
            palabras_alt = alt_lower.split()
            palabras_resp = resp_lower.split()
            if abs(len(palabras_alt) - len(palabras_resp)) > 3:
                return False
                
            # Análisis con spaCy
            doc_alt = self.nlp(alternativa)
            doc_resp = self.nlp(respuesta)
            
            # Verificar que ambos documentos tienen contenido válido
            if not doc_alt.vector.any() or not doc_resp.vector.any():
                return False
                
            # Calcular similitud usando vectores (corregido)
            vector_alt = doc_alt.vector.reshape(1, -1)  # Asegurar 2D
            vector_resp = doc_resp.vector.reshape(1, -1)  # Asegurar 2D
            
            similitud = cosine_similarity(vector_alt, vector_resp)[0][0]
            
            # Validar similitud semántica (no muy similar ni muy diferente)
            if similitud > 0.9 or similitud < 0.2:
                return False
                
            # Verificar que sean del mismo tipo gramatical
            pos_alt = {token.pos_ for token in doc_alt}
            pos_resp = {token.pos_ for token in doc_resp}
            if not pos_alt.intersection(pos_resp):
                return False
                
            # Verificar coherencia con el contexto
            doc_contexto = self.nlp(contexto)
            vector_contexto = doc_contexto.vector.reshape(1, -1)  # Asegurar 2D
            similitud_contexto = cosine_similarity(vector_alt, vector_contexto)[0][0]
            
            if similitud_contexto < 0.1:  # Debe tener alguna relación con el contexto
                return False
            
            return True
            
        except Exception as e:
            print(f"Error en validación de alternativa: {str(e)}")
            return False

    def _generar_verdadero_falso(self, info: Dict) -> Optional[Dict]:
        """Genera una pregunta de verdadero/falso mediante modificación inteligente"""
        try:
            doc = self.nlp(info['texto'])
            
            # Decidir si mantener verdadera o modificar
            es_verdadero = random.choice([True, False])
            
            if es_verdadero:
                afirmacion = info['texto']
            else:
                # Modificar la afirmación de manera coherente
                afirmacion = self._modificar_afirmacion_coherente(doc, info['entidades'])
            
            if afirmacion:
                return {
                    'pregunta': afirmacion,
                    'respuesta_correcta': "Verdadero" if es_verdadero else "Falso",
                    'tipo': 'verdadero_falso',
                    'explicacion': self._generar_explicacion(info['texto'], es_verdadero, afirmacion)
                }
            
            return None
            
        except Exception as e:
            print(f"Error generando pregunta V/F: {str(e)}")
            return None

    def _modificar_afirmacion_coherente(self, doc, entidades: List) -> Optional[str]:
        """Modifica una afirmación de manera coherente para hacerla falsa"""
        texto_modificado = doc.text
        
        # Estrategias de modificación
        estrategias = [
            self._modificar_entidad,
            self._modificar_cantidad,
            self._modificar_concepto
        ]
        
        # Intentar estrategias hasta lograr una modificación válida
        random.shuffle(estrategias)
        for estrategia in estrategias:
            modificacion = estrategia(doc, entidades)
            if modificacion:
                texto_modificado = modificacion
                break
                
        return texto_modificado if texto_modificado != doc.text else None

    def _modificar_entidad(self, doc, entidades: List) -> Optional[str]:
        """Modifica una entidad nombrada manteniendo coherencia"""
        if not doc.ents:
            return None
            
        ent = random.choice(doc.ents)
        texto = doc.text
        
        # Encontrar una entidad similar pero diferente
        for otra_ent, label in entidades:
            if (otra_ent != ent.text and 
                label == ent.label_ and 
                self.nlp(otra_ent).similarity(self.nlp(ent.text)) > 0.5):
                return texto.replace(ent.text, otra_ent)
                
        return None

    def _modificar_cantidad(self, doc, _) -> Optional[str]:
        """Modifica valores numéricos de manera coherente"""
        texto = doc.text
        for token in doc:
            if token.like_num:
                try:
                    num = float(token.text)
                    # Modificar el número de manera realista
                    nuevo_num = str(int(num * random.uniform(1.5, 2.5)))
                    return texto.replace(token.text, nuevo_num)
                except:
                    continue
        return None

    def _modificar_concepto(self, doc, _) -> Optional[str]:
        """Modifica conceptos clave manteniendo coherencia gramatical"""
        texto = doc.text
        for token in doc:
            if token.pos_ in ['NOUN', 'VERB', 'ADJ'] and len(token.text) > 3:
                # Generar alternativa usando el modelo T5
                prompt = f"generate opposite of: {token.text}"
                inputs = self.t5_tokenizer(prompt, return_tensors="pt")
                outputs = self.t5_model.generate(**inputs, max_length=32)
                alternativa = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                if alternativa and alternativa != token.text:
                    return texto.replace(token.text, alternativa)
        return None

    def _generar_explicacion(self, original: str, es_verdadero: bool, modificada: str) -> str:
        """Genera una explicación detallada para la respuesta"""
        if es_verdadero:
            return f"Esta afirmación es correcta según el texto original: '{original}'"
        else:
            # Analizar las diferencias
            doc_orig = self.nlp(original)
            doc_mod = self.nlp(modificada)
            
            diferencias = []
            for token_orig, token_mod in zip(doc_orig, doc_mod):
                if token_orig.text != token_mod.text:
                    diferencias.append(f"'{token_orig.text}' fue cambiado por '{token_mod.text}'")
            
            explicacion = f"La afirmación es falsa. En el texto original dice: '{original}'. "
            if diferencias:
                explicacion += f"Las diferencias son: {', '.join(diferencias)}"
            
            return explicacion

    def _generar_cuestionario_completo(self, texto: str) -> List[Dict]:
        """Genera un cuestionario completo con control de calidad"""
        preguntas = []
        alternativas_count = 0
        vf_count = 0
        max_intentos = 30  # Máximo de intentos para conseguir 10 preguntas
        
        try:
            info_relevante = self._extraer_informacion_relevante(texto)
            if not info_relevante:
                return []
            
            # Control de timeouts
            inicio = time.time()
            timeout_global = inicio + 1200  # 20 minutos máximo por cuestionario
            
            with tqdm(total=self.max_preguntas, desc="Generando preguntas") as pbar:
                intentos = 0
                
                while len(preguntas) < self.max_preguntas and intentos < max_intentos:
                    if time.time() > timeout_global:
                        print("\nTimeout global alcanzado")
                        break
                        
                    # Elegir tipo de pregunta según lo que falta
                    if alternativas_count < 6:
                        tipo = 'alternativas'
                    elif vf_count < 4:
                        tipo = 'verdadero_falso'
                    else:
                        break
                        
                    try:
                        # Generar pregunta con timeout individual
                        timeout_pregunta = time.time() + 180  # 3 minutos por pregunta
                        
                        with torch.no_grad():
                            info = random.choice(info_relevante)
                            
                            if tipo == 'alternativas':
                                pregunta = self._generar_pregunta_analisis_mejorada(info, timeout_pregunta)
                            else:
                                pregunta = self._generar_verdadero_falso_mejorado(info, timeout_pregunta)
                            
                            if pregunta:
                                # Validar y post-procesar
                                pregunta_procesada = self._post_procesar_pregunta(pregunta)
                                
                                if pregunta_procesada and self._validar_pregunta_final(pregunta_procesada, texto):
                                    preguntas.append(pregunta_procesada)
                                    if pregunta_procesada['tipo'] == 'alternativas':
                                        alternativas_count += 1
                                    else:
                                        vf_count += 1
                                    pbar.update(1)
                                    
                                    # Limpiar memoria cada 2 preguntas exitosas
                                    if len(preguntas) % 2 == 0:
                                        self._limpiar_memoria()
                                    
                    except Exception as e:
                        print(f"\nError en intento {intentos}: {str(e)}")
                    
                    intentos += 1
            
            # Verificar cantidad mínima de preguntas
            if len(preguntas) >= self.max_preguntas * 0.8:  # Al menos 80% de las preguntas requeridas
                return preguntas
            return []
            
        except Exception as e:
            print(f"\nError generando cuestionario: {str(e)}")
            return []

    def _validar_pregunta(self, pregunta: Dict, texto_original: str) -> bool:
        """Valida la calidad y coherencia de una pregunta generada"""
        try:
            # Validaciones básicas
            if not pregunta.get('pregunta'):
                return False
                
            # Validar coherencia semántica
            doc = self.nlp(pregunta['pregunta'])
            if not any(token.pos_ == 'VERB' for token in doc):
                return False
                
            # Validar estructura según tipo
            if pregunta['tipo'] == 'alternativas':
                if not (
                    'opciones' in pregunta and 
                    'respuesta_correcta' in pregunta and
                    len(pregunta['opciones']) == 4 and
                    pregunta['respuesta_correcta'] in pregunta['opciones']
                ):
                    return False
                    
            elif pregunta['tipo'] == 'verdadero_falso':
                if not (
                    'respuesta_correcta' in pregunta and
                    pregunta['respuesta_correcta'] in ['Verdadero', 'Falso'] and
                    'explicacion' in pregunta
                ):
                    return False
            
            # Validar relevancia con el texto original
            vectorizer = TfidfVectorizer()
            tfidf = vectorizer.fit_transform([pregunta['pregunta'], texto_original])
            similitud = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
            
            return similitud >= self.min_similitud
            
        except Exception as e:
            print(f"Error en validación: {str(e)}")
            return False

    def procesar_archivo(self, ruta_archivo: str) -> Dict:
        """Procesa archivo JSON con mejoras en la estructura"""
        try:
            with open(ruta_archivo, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            resultados = {}
            print(f"\nProcesando archivo...")
            
            # Procesar las materias que existen en el JSON
            for materia in data.keys():
                if not data[materia]:  # Materia sin contenido
                    continue
                    
                print(f"\nProcesando materia: {materia}")
                resultados[materia] = []
                
                # Procesar cada texto de la materia
                for item in tqdm(data[materia], desc=f"Textos de {materia}"):
                    if not item.get('texto'):
                        continue
                    
                    # Generar cuestionario para este texto
                    cuestionario = self._generar_cuestionario_completo(item['texto'])
                    
                    if cuestionario:
                        resultados[materia].append({
                            'texto': item['texto'],
                            'fuente': item.get('fuente', ''),
                            'preguntas': cuestionario
                        })
                        
                        # Guardar avance parcial
                        self._guardar_resultados(resultados, f"tmp_{materia}_cuestionario.json")
            
            # Guardar resultados finales
            if resultados:
                self._guardar_resultados(resultados)
                print("\nCuestionarios generados exitosamente")
            else:
                print("\nNo se pudieron generar cuestionarios")
                
            return resultados
            
        except Exception as e:
            print(f"\nError procesando archivo: {str(e)}")
            return {}

    def _post_procesar_pregunta(self, pregunta: Dict) -> Optional[Dict]:
        """Procesa una única pregunta para garantizar calidad"""
        try:
            pregunta_limpia = self._limpiar_texto(pregunta['pregunta'])
            
            if pregunta['tipo'] == 'alternativas':
                opciones_limpias = [self._limpiar_texto(opcion) for opcion in pregunta['opciones']]
                respuesta_limpia = self._limpiar_texto(pregunta['respuesta_correcta'])
                
                if self._verificar_coherencia_final({
                    'pregunta': pregunta_limpia,
                    'opciones': opciones_limpias,
                    'respuesta_correcta': respuesta_limpia
                }):
                    return {
                        'pregunta': pregunta_limpia,
                        'opciones': opciones_limpias,
                        'respuesta_correcta': respuesta_limpia,
                        'tipo': 'alternativas'
                    }
            
            elif pregunta['tipo'] == 'verdadero_falso':
                explicacion_limpia = self._limpiar_texto(pregunta['explicacion'])
                
                if self._verificar_coherencia_final({
                    'pregunta': pregunta_limpia,
                    'explicacion': explicacion_limpia
                }):
                    return {
                        'pregunta': pregunta_limpia,
                        'respuesta_correcta': pregunta['respuesta_correcta'],
                        'tipo': 'verdadero_falso',
                        'explicacion': explicacion_limpia
                    }
            
            return None
            
        except Exception as e:
            print(f"Error en post-procesamiento: {str(e)}")
            return None

    def _limpiar_texto(self, texto: str) -> str:
        """Limpia y normaliza el texto"""
        # Eliminar espacios múltiples
        texto = ' '.join(texto.split())
        
        # Asegurar puntuación correcta
        if texto and not texto[-1] in '.?!':
            texto += '.'
            
        # Capitalizar primera letra
        texto = texto[0].upper() + texto[1:] if texto else texto
        
        return texto

    def _verificar_coherencia_final(self, pregunta: Dict) -> bool:
        """Realiza una verificación final de coherencia"""
        try:
            doc = self.nlp(pregunta['pregunta'])
            
            # Verificar estructura gramatical básica
            tiene_verbo = False
            tiene_sujeto = False
            
            for token in doc:
                if token.pos_ == 'VERB':
                    tiene_verbo = True
                if token.dep_ in ['nsubj', 'nsubjpass']:
                    tiene_sujeto = True
            
            if not (tiene_verbo and tiene_sujeto):
                return False
            
            # Para preguntas de alternativas
            if 'opciones' in pregunta:
                # Verificar que las opciones sean del mismo tipo semántico
                embeddings_opciones = [self.nlp(opcion).vector for opcion in pregunta['opciones']]
                similitudes = cosine_similarity(embeddings_opciones)
                
                # Comprobar que las opciones tengan cierta similitud entre sí
                if not (0.3 <= similitudes.mean() <= 0.8):
                    return False
                
                # Verificar que las opciones sean respuestas válidas a la pregunta
                pregunta_sin_signos = pregunta['pregunta'].replace('¿', '').replace('?', '')
                for opcion in pregunta['opciones']:
                    oracion_prueba = pregunta_sin_signos.replace('_____', opcion)
                    doc_prueba = self.nlp(oracion_prueba)
                    
                    # Verificar coherencia gramatical de la oración resultante
                    if not self._verificar_gramatica(doc_prueba):
                        return False
            
            # Para preguntas de verdadero/falso
            if 'explicacion' in pregunta:
                # Verificar que la explicación sea coherente con la pregunta
                doc_explicacion = self.nlp(pregunta['explicacion'])
                similitud = doc.similarity(doc_explicacion)
                
                if similitud < 0.4:
                    return False
            
            return True
            
        except Exception as e:
            print(f"Error en verificación final: {str(e)}")
            return False

    def _verificar_gramatica(self, doc) -> bool:
        """Verifica la coherencia gramatical de una oración"""
        # Verificar estructura básica
        tiene_verbo_principal = False
        tiene_sujeto = False
        
        for token in doc:
            if token.dep_ == 'ROOT' and token.pos_ == 'VERB':
                tiene_verbo_principal = True
            if token.dep_ in ['nsubj', 'nsubjpass']:
                tiene_sujeto = True
            
            # Verificar concordancia básica
            if token.dep_ == 'det' and token.head.pos_ == 'NOUN':
                if token.morph.get('Gender') != token.head.morph.get('Gender') or \
                   token.morph.get('Number') != token.head.morph.get('Number'):
                    return False
        
        return tiene_verbo_principal and tiene_sujeto

    def _guardar_resultados(self, resultados: Dict, nombre_archivo: str = 'cuestionarios_generados.json'):
        """Guarda resultados con manejo de errores"""
        try:
            ruta_salida = f'tutorApp/static/json/{nombre_archivo}'
            with open(ruta_salida, 'w', encoding='utf-8') as f:
                json.dump(resultados, f, ensure_ascii=False, indent=4)
                
            print(f"\nResultados guardados en '{ruta_salida}'")
            
        except Exception as e:
            print(f"Error guardando resultados: {str(e)}")
            # Intentar guardar en ubicación alternativa
            try:
                with open(nombre_archivo, 'w', encoding='utf-8') as f:
                    json.dump(resultados, f, ensure_ascii=False, indent=4)
                print(f"Resultados guardados en ubicación alternativa: '{nombre_archivo}'")
            except:
                print("No se pudieron guardar los resultados")

    def __del__(self):
        """Limpieza al destruir la instancia"""
        self._limpiar_memoria()