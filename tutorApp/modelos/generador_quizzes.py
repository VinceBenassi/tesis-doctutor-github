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

class GeneradorCuestionarios:
    def __init__(self):
        print("Inicializando modelos...")
        
        # Modelo principal para generación de preguntas
        self.qa_model = pipeline(
            "question-answering",
            model="mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"
        )
        
        # Modelo T5 para generación de texto
        self.t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
        self.t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
        
        # Modelo BERT para clasificación y verificación
        self.bert_tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
        self.bert_model = AutoModelForSequenceClassification.from_pretrained(
            "dccuchile/bert-base-spanish-wwm-cased",
            num_labels=2
        )
        
        # Modelo spaCy para análisis lingüístico
        self.nlp = spacy.load("es_core_news_lg")
        
        # Configuración
        self.max_preguntas = 10
        self.min_similitud = 0.3
        
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
        """Genera una pregunta de análisis usando T5"""
        try:
            # Generar pregunta
            prompt = f"generate analytical question in Spanish: {info['texto']}"
            inputs = self.t5_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            
            outputs = self.t5_model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                no_repeat_ngram_size=2
            )
            
            pregunta = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Validar respuesta usando QA
            qa_result = self.qa_model(
                question=pregunta,
                context=info['texto']
            )
            
            if qa_result['score'] > 0.7:
                respuesta_correcta = qa_result['answer']
                alternativas = self._generar_alternativas_coherentes(
                    respuesta_correcta,
                    info['texto'],
                    info['entidades']
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

    def _generar_alternativas_coherentes(self, respuesta: str, contexto: str, entidades: List) -> List[str]:
        """Genera alternativas coherentes basadas en el contexto y entidades"""
        try:
            alternativas = set()
            doc_respuesta = self.nlp(respuesta)
            
            # Identificar tipo de respuesta
            if doc_respuesta.ents:
                tipo_entidad = doc_respuesta.ents[0].label_
                # Buscar entidades similares del mismo tipo
                for ent, label in entidades:
                    if label == tipo_entidad and ent != respuesta:
                        alternativas.add(ent)
            
            # Si no hay suficientes alternativas, usar análisis semántico
            if len(alternativas) < 3:
                doc_contexto = self.nlp(contexto)
                doc_respuesta_vector = doc_respuesta.vector
                
                for token in doc_contexto:
                    if (token.pos_ in ['NOUN', 'PROPN'] and 
                        token.text != respuesta and 
                        len(token.text) > 3):
                        # Calcular similitud usando los vectores de palabras
                        similitud = cosine_similarity(
                            [token.vector],
                            [doc_respuesta_vector]
                        )[0][0]
                        
                        if 0.3 <= similitud <= 0.7:
                            alternativas.add(token.text)
            
            # Si aún no hay suficientes alternativas, intentar con sintagmas nominales
            if len(alternativas) < 3:
                for chunk in self.nlp(contexto).noun_chunks:
                    if (chunk.text != respuesta and 
                        len(chunk.text) > 3 and 
                        chunk.root.pos_ in ['NOUN', 'PROPN']):
                        
                        similitud = cosine_similarity(
                            [chunk.vector],
                            [doc_respuesta_vector]
                        )[0][0]
                        
                        if 0.3 <= similitud <= 0.7:
                            alternativas.add(chunk.text)
            
            # Validar coherencia de las alternativas encontradas
            alternativas_validadas = []
            for alt in alternativas:
                if self._validar_alternativa(alt, respuesta, contexto):
                    alternativas_validadas.append(alt)
            
            # Si no se tienen suficientes alternativas, generar algunas usando T5
            while len(alternativas_validadas) < 3:
                nueva_alt = self._generar_alternativa_t5(respuesta, contexto)
                if nueva_alt and nueva_alt not in alternativas_validadas:
                    if self._validar_alternativa(nueva_alt, respuesta, contexto):
                        alternativas_validadas.append(nueva_alt)
            
            return list(alternativas_validadas)[:3]
            
        except Exception as e:
            print(f"Error generando alternativas: {str(e)}")
            return []
        
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
        """Valida la coherencia de una alternativa"""
        if not alternativa or alternativa == respuesta:
            return False
            
        # Validar longitud similar
        if abs(len(alternativa.split()) - len(respuesta.split())) > 3:
            return False
            
        # Validar coherencia semántica
        doc_alt = self.nlp(alternativa)
        doc_resp = self.nlp(respuesta)
        similitud = doc_alt.similarity(doc_resp)
        
        if similitud > 0.9 or similitud < 0.2:
            return False
            
        return True

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

    def generar_cuestionario(self, texto: str) -> List[Dict]:
        """Genera un cuestionario completo con 10 preguntas balanceadas"""
        preguntas = []
        
        try:
            # Extraer información relevante
            info_relevante = self._extraer_informacion_relevante(texto)
            
            # Generar preguntas balanceadas (6 alternativas, 4 V/F)
            while len(preguntas) < self.max_preguntas and info_relevante:
                info = info_relevante.pop(0)
                
                if len(preguntas) < 6:
                    pregunta = self._generar_pregunta_analisis(info)
                else:
                    pregunta = self._generar_verdadero_falso(info)
                    
                if pregunta and self._validar_pregunta(pregunta, texto):
                    preguntas.append(pregunta)
            
            return preguntas
            
        except Exception as e:
            print(f"Error generando cuestionario: {str(e)}")
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
        """Procesa archivo JSON y genera cuestionarios"""
        try:
            with open(ruta_archivo, 'r', encoding='utf-8') as f:
                datos = json.load(f)
            
            resultados = {}
            
            # Procesar cada materia y sus textos
            for materia in datos.keys():
                if not datos[materia]:  # Ignorar materias sin contenido
                    continue
                
                resultados[materia] = []
                print(f"\nProcesando {materia}...")
                
                for item in datos[materia]:
                    if not item.get('texto'):  # Ignorar items sin texto
                        continue
                        
                    preguntas = self.generar_cuestionario(item['texto'])
                    
                    # Validar calidad del cuestionario completo
                    if len(preguntas) == self.max_preguntas:
                        resultados[materia].append({
                            'texto': item['texto'],
                            'fuente': item.get('fuente', ''),
                            'preguntas': self._post_procesar_preguntas(preguntas)
                        })
            
            # Guardar resultados
            self._guardar_resultados(resultados)
            
            return resultados
            
        except Exception as e:
            print(f"Error procesando archivo: {str(e)}")
            return {}

    def _post_procesar_preguntas(self, preguntas: List[Dict]) -> List[Dict]:
        """Realiza un post-procesamiento final de las preguntas para garantizar calidad"""
        preguntas_procesadas = []
        
        for pregunta in preguntas:
            # Limpiar y normalizar texto
            pregunta_limpia = self._limpiar_texto(pregunta['pregunta'])
            
            if pregunta['tipo'] == 'alternativas':
                opciones_limpias = [self._limpiar_texto(opcion) for opcion in pregunta['opciones']]
                respuesta_limpia = self._limpiar_texto(pregunta['respuesta_correcta'])
                
                # Verificar coherencia final
                if self._verificar_coherencia_final({
                    'pregunta': pregunta_limpia,
                    'opciones': opciones_limpias,
                    'respuesta_correcta': respuesta_limpia
                }):
                    preguntas_procesadas.append({
                        'pregunta': pregunta_limpia,
                        'opciones': opciones_limpias,
                        'respuesta_correcta': respuesta_limpia,
                        'tipo': 'alternativas'
                    })
                    
            elif pregunta['tipo'] == 'verdadero_falso':
                explicacion_limpia = self._limpiar_texto(pregunta['explicacion'])
                
                if self._verificar_coherencia_final({
                    'pregunta': pregunta_limpia,
                    'explicacion': explicacion_limpia
                }):
                    preguntas_procesadas.append({
                        'pregunta': pregunta_limpia,
                        'respuesta_correcta': pregunta['respuesta_correcta'],
                        'tipo': 'verdadero_falso',
                        'explicacion': explicacion_limpia
                    })
        
        return preguntas_procesadas

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

    def _guardar_resultados(self, resultados: Dict):
        """Guarda los resultados en un archivo JSON"""
        try:
            ruta_salida = 'tutorApp/static/json/cuestionarios_generados.json'
            with open(ruta_salida, 'w', encoding='utf-8') as f:
                json.dump(resultados, f, ensure_ascii=False, indent=4)
                
            print(f"\nResultados guardados en '{ruta_salida}'")
            
        except Exception as e:
            print(f"Error guardando resultados: {str(e)}")