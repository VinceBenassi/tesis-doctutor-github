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

        # Configuración de device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Inicialización directa de modelos
        print("Cargando modelo QA...")
        self._qa_model = pipeline(
            "question-answering",
            model="mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es",
            device=self.device
        )
        
        print("Cargando tokenizer T5...")
        self._t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
        
        print("Cargando modelo T5...")
        self._t5_model = T5ForConditionalGeneration.from_pretrained(
            "google/flan-t5-large"
        ).to(self.device)
        
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
        self.timeout = 300  # 5 minutos máximo por pregunta

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

    def _generar_pregunta_analisis(self, info: Dict) -> Optional[Dict]:
        """Genera una pregunta de análisis más coherente"""
        try:
            # 1. Extraer conceptos clave del texto
            doc = self.nlp(info['texto'])
            conceptos_clave = self._extraer_conceptos_clave(doc)
            if not conceptos_clave:
                return None

            # 2. Generar pregunta usando T5 con contexto específico
            concepto = random.choice(conceptos_clave)
            prompts = [
                f"generate question about '{concepto}' definition in Spanish: {info['texto']}",
                f"generate question about '{concepto}' purpose in Spanish: {info['texto']}",
                f"generate question about '{concepto}' characteristics in Spanish: {info['texto']}",
                f"generate question about '{concepto}' importance in Spanish: {info['texto']}"
            ]
            prompt = random.choice(prompts)
            inputs = self.t5_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            
            outputs = self.t5_model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                no_repeat_ngram_size=2,
                temperature=0.8
            )
            
            pregunta = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 3. Usar QA para encontrar la respuesta correcta
            qa_result = self.qa_model(question=pregunta, context=info['texto'])
            
            if qa_result['score'] > 0.7:
                respuesta_correcta = qa_result['answer']
                
                # 4. Generar alternativas conceptualmente relacionadas
                alternativas = self._generar_alternativas_semanticas(
                    respuesta_correcta,
                    concepto,
                    info['texto']
                )
                
                if alternativas and len(alternativas) >= 3:
                    opciones = alternativas + [respuesta_correcta]
                    random.shuffle(opciones)
                    
                    return {
                        'pregunta': pregunta,
                        'opciones': opciones,
                        'respuesta_correcta': respuesta_correcta,
                        'tipo': 'alternativas'
                    }
            
            return None
            
        except Exception as e:
            print(f"Error generando pregunta: {str(e)}")
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
        """Genera alternativas semánticamente relacionadas"""
        try:
            alternativas = set()
            doc_respuesta = self.nlp(respuesta)
            doc_concepto = self.nlp(concepto)
            
            # 1. Extraer información relacionada del contexto
            doc_contexto = self.nlp(contexto)
            oraciones_relevantes = []
            
            for sent in doc_contexto.sents:
                if concepto.lower() in sent.text.lower():
                    oraciones_relevantes.append(sent.text)
            
            # 2. Generar alternativas basadas en el contexto
            for oracion in oraciones_relevantes:
                # Usar T5 para generar variaciones
                prompt = f"generate alternative answer about {concepto} in Spanish: {oracion}"
                inputs = self.t5_tokenizer(prompt, return_tensors="pt")
                outputs = self.t5_model.generate(
                    **inputs,
                    max_length=64,
                    num_beams=4,
                    no_repeat_ngram_size=2,
                    do_sample=True,
                    temperature=0.8
                )
                
                alternativa = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                if self._validar_alternativa_semantica(alternativa, respuesta, concepto):
                    alternativas.add(alternativa)
            
            # 3. Si no hay suficientes, generar usando relaciones conceptuales
            while len(alternativas) < 3:
                prompt = f"generate related but incorrect answer about {concepto} in Spanish"
                inputs = self.t5_tokenizer(prompt, return_tensors="pt")
                outputs = self.t5_model.generate(
                    **inputs,
                    max_length=64,
                    num_beams=4,
                    do_sample=True,
                    temperature=0.9
                )
                
                alternativa = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
                if self._validar_alternativa_semantica(alternativa, respuesta, concepto):
                    alternativas.add(alternativa)
            
            return list(alternativas)[:3]
            
        except Exception as e:
            print(f"Error generando alternativas semánticas: {str(e)}")
            return []

    def _validar_alternativa_semantica(self, alternativa: str, respuesta: str, concepto: str) -> bool:
        """Valida la coherencia semántica de una alternativa"""
        try:
            if not alternativa or alternativa == respuesta:
                return False
            
            doc_alt = self.nlp(alternativa)
            doc_resp = self.nlp(respuesta)
            doc_concepto = self.nlp(concepto)
            
            # Verificar longitud y estructura
            if abs(len(doc_alt) - len(doc_resp)) > 5:
                return False
            
            # Verificar que contenga el concepto o términos relacionados
            similitud_concepto = doc_alt.similarity(doc_concepto)
            if similitud_concepto < 0.3:
                return False
            
            # Verificar que sea una alternativa válida pero no idéntica
            similitud_respuesta = doc_alt.similarity(doc_resp)
            if similitud_respuesta > 0.8 or similitud_respuesta < 0.2:
                return False
            
            # Verificar coherencia gramatical
            tiene_verbo = any(token.pos_ == 'VERB' for token in doc_alt)
            tiene_sustantivo = any(token.pos_ == 'NOUN' for token in doc_alt)
            
            return tiene_verbo and tiene_sustantivo
            
        except Exception as e:
            print(f"Error en validación semántica: {str(e)}")
            return False

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

    def _generar_cuestionario_con_timeout(self, texto: str, timeout: int = 1200) -> List[Dict]:
        """Genera un cuestionario con mejor control de tiempo"""
        preguntas = []
        start_time = time.time()
        intentos_maximos = 20  # Permitir más intentos para lograr las 10 preguntas
        
        try:
            info_relevante = self._extraer_informacion_relevante(texto)
            if not info_relevante:
                return []
            
            # Intentar generar preguntas hasta tener 10 o agotar tiempo/intentos
            with tqdm(total=self.max_preguntas) as pbar:
                intentos = 0
                while len(preguntas) < self.max_preguntas and intentos < intentos_maximos:
                    if time.time() - start_time > timeout:
                        print("\nTimeout global alcanzado")
                        break
                    
                    info = info_relevante[intentos % len(info_relevante)]
                    intentos += 1
                    
                    try:
                        # Alternar entre tipos de preguntas
                        if len(preguntas) < 6:
                            nueva_pregunta = self._generar_pregunta_analisis(info)
                        else:
                            nueva_pregunta = self._generar_verdadero_falso(info)
                        
                        if nueva_pregunta and self._validar_pregunta(nueva_pregunta, texto):
                            preguntas.append(nueva_pregunta)
                            pbar.update(1)
                            
                    except Exception as e:
                        print(f"\nError en intento {intentos}: {str(e)}")
                        continue
                    
                    # Limpiar memoria cada 3 intentos
                    if intentos % 3 == 0:
                        self._limpiar_memoria()
            
            return preguntas
            
        except Exception as e:
            print(f"\nError en generación de cuestionario: {str(e)}")
            return preguntas

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
        """Procesa archivo JSON con manejo mejorado de errores y timeout"""
        resultados = {}
        
        try:
            # Cargar y validar datos
            with open(ruta_archivo, 'r', encoding='utf-8') as f:
                datos = json.load(f)
                if not isinstance(datos, dict):
                    print("Error: Formato de datos inválido")
                    return {}

            total_materias = len(datos.keys())
            print(f"Procesando {total_materias} materias...")
            
            # Procesar cada materia
            for materia in tqdm(datos.keys(), desc="Procesando materias"):
                # Inicializar lista para la materia
                resultados[materia] = []
                print(f"\nProcesando {materia}...")
                
                # Obtener contenido de la materia
                contenido = datos[materia]
                if not contenido:
                    continue
                    
                # Procesar cada entrada de contenido
                for item in tqdm(contenido, desc=f"Procesando textos de {materia}"):
                    if not isinstance(item, dict) or 'texto' not in item:
                        continue
                        
                    # Establecer tiempo límite por texto
                    start_time = time.time()
                    preguntas = []
                    
                    try:
                        # Generar cuestionario con timeout extendido
                        preguntas = self._generar_cuestionario_con_timeout(
                            item['texto'],
                            timeout=600  # 10 minutos por texto
                        )
                        
                        if preguntas:
                            registro = {
                                'texto': item['texto'],
                                'fuente': item.get('fuente', ''),
                                'preguntas': preguntas
                            }
                            resultados[materia].append(registro)
                            
                            # Guardar resultados parciales
                            self._guardar_resultados(
                                resultados,
                                f"resultados_parciales_{materia}.json"
                            )
                    
                    except Exception as e:
                        print(f"\nError procesando texto en {materia}: {str(e)}")
                        continue
                    
                    finally:
                        self._limpiar_memoria()
                        
                # Verificar si se generaron preguntas para la materia
                if not resultados[materia]:
                    print(f"No se generaron preguntas para {materia}")
            
            # Guardar resultados finales solo si hay contenido
            if any(resultados.values()):
                self._guardar_resultados(resultados)
            else:
                print("No se generaron preguntas para ninguna materia")
            
            return resultados
            
        except Exception as e:
            print(f"\nError procesando archivo: {str(e)}")
            return resultados

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