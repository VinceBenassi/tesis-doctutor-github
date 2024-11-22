# Generador de Cuestionarios
# Por Franco Benassi
import json
import os
import re
import random
import spacy
import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional
from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer,
    BertTokenizer, 
    BertModel,
    BertForQuestionAnswering,
    BertForSequenceClassification,
    pipeline
)
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from collections import defaultdict

class GeneradorCuestionarios:
    def __init__(self):
        """Inicializa los modelos y configuraciones necesarias."""
        try:
            # Modelos de transformers
            self.t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
            self.t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
            
            # Modelo BERT específico para español
            self.bert_model_name = "dccuchile/bert-base-spanish-wwm-cased"
            self.bert_tokenizer = BertTokenizer.from_pretrained(self.bert_model_name)
            self.bert_model = BertForQuestionAnswering.from_pretrained(self.bert_model_name)
            self.bert_qa = pipeline(
                "question-answering",
                model=self.bert_model_name,
                tokenizer=self.bert_tokenizer
            )
            
            # Modelo de embeddings para español
            self.sentence_model = SentenceTransformer('hiiamsid/sentence_similarity_spanish_es')
            
            # Spacy para procesamiento en español
            try:
                self.nlp = spacy.load("es_core_news_lg")  # Intentar cargar modelo grande
            except OSError:
                print("Modelo grande de spacy no encontrado, usando modelo pequeño...")
                self.nlp = spacy.load("es_core_news_sm")

            # Configurar dispositivo (GPU/CPU)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Usando dispositivo: {self.device}")
            
            # Mover modelos al dispositivo apropiado
            self._mover_modelos_a_device()
            
            # Fine-tuning configuration for T5
            self.t5_config = {
                'max_length': 512,
                'num_beams': 4,
                'length_penalty': 2.0,
                'no_repeat_ngram_size': 3,
                'early_stopping': True
            }

            # Agregar diccionario de palabras relacionadas
            self.palabras_relacionadas = {
                # Sustantivos comunes
                'persona': ['individuo', 'ser humano', 'sujeto'],
                'problema': ['dificultad', 'obstáculo', 'inconveniente'],
                'proceso': ['procedimiento', 'método', 'sistema'],
                'desarrollo': ['evolución', 'progreso', 'avance'],
                'resultado': ['consecuencia', 'efecto', 'desenlace'],
                
                # Verbos comunes
                'hacer': ['realizar', 'ejecutar', 'efectuar'],
                'generar': ['producir', 'crear', 'originar'],
                'analizar': ['examinar', 'estudiar', 'investigar'],
                'implementar': ['aplicar', 'instaurar', 'establecer'],
                'mejorar': ['optimizar', 'perfeccionar', 'desarrollar'],
                
                # Adjetivos comunes
                'importante': ['relevante', 'significativo', 'fundamental'],
                'complejo': ['complicado', 'difícil', 'elaborado'],
                'efectivo': ['eficaz', 'eficiente', 'funcional'],
                'nuevo': ['reciente', 'moderno', 'actual'],
                'principal': ['primordial', 'esencial', 'básico'],
                
                # Términos técnicos
                'sistema': ['estructura', 'organización', 'mecanismo'],
                'tecnología': ['técnica', 'herramienta', 'instrumento'],
                'método': ['procedimiento', 'técnica', 'metodología'],
                'concepto': ['idea', 'noción', 'principio'],
                'función': ['rol', 'papel', 'tarea']
            }
            
            print("Inicialización exitosa de todos los modelos")
            
        except Exception as e:
            print(f"Error durante la inicialización: {str(e)}")
            raise

    def _mover_modelos_a_device(self):
        """Mueve los modelos al dispositivo correspondiente y configura para mixed precision."""
        try:
            self.t5_model = self.t5_model.to(self.device)
            self.bert_model = self.bert_model.to(self.device)
            
            # Habilitar mixed precision si hay GPU disponible
            if self.device.type == 'cuda':
                self.t5_model = self.t5_model.half()  # Convertir a FP16
                self.bert_model = self.bert_model.half()
                
        except Exception as e:
            print(f"Error moviendo modelos al dispositivo: {str(e)}")
            raise

    def _prepare_t5_input(self, texto: str, task_prefix: str = "generate question: ") -> torch.Tensor:
        """Prepara el input para T5 con el formato correcto."""
        input_text = f"{task_prefix}{texto}"
        inputs = self.t5_tokenizer(
            input_text,
            max_length=self.t5_config['max_length'],
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        return inputs.input_ids.to(self.device)

    def generar_cuestionario(self, texto: str, materia: str, fuente: str,
                            num_preguntas: int = 10) -> Dict[str, Any]:
        """Genera un cuestionario completo basado en el texto."""
        # Análisis semántico del texto
        doc = self.nlp(texto)
        conceptos_clave = self._extraer_conceptos_clave(doc)
        oraciones = [sent.text.strip() for sent in doc.sents]
        embedding_texto = self._obtener_embedding_texto(texto)
        
        preguntas = []
        num_alternativas = num_preguntas // 2
        num_vf = num_preguntas - num_alternativas
        
        # Generar preguntas de alternativas
        for _ in range(num_alternativas):
            pregunta = self._generar_pregunta_alternativas(
                texto, 
                conceptos_clave,
                embedding_texto,
                oraciones
            )
            if pregunta:
                preguntas.append(pregunta)
                
        # Generar preguntas de verdadero/falso
        for _ in range(num_vf):
            pregunta = self._generar_pregunta_vf(
                texto,
                conceptos_clave,
                embedding_texto,
                oraciones
            )
            if pregunta:
                preguntas.append(pregunta)
                
        return {
            "materia": materia,
            "fuente": fuente,
            "preguntas": preguntas
        }

    def _extraer_conceptos_clave(self, doc) -> List[str]:
        """Extrae conceptos clave usando análisis lingüístico y embeddings."""
        # Extraer sustantivos y frases nominales importantes
        candidatos = []
        for chunk in doc.noun_chunks:
            if not any(token.is_stop for token in chunk):
                candidatos.append(chunk.text)
                
        # Extraer entidades nombradas
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PERSON', 'LOC', 'CONCEPT']:
                candidatos.append(ent.text)
                
        # Calcular importancia usando embeddings
        if not candidatos:
            return []
            
        embeddings = self.sentence_model.encode(candidatos)
        importancia = np.mean(embeddings, axis=1)
        
        # Seleccionar los conceptos más relevantes
        indices = np.argsort(importancia)[-10:]  # Top 10 conceptos
        return [candidatos[i] for i in indices]

    def _obtener_embedding_texto(self, texto: str) -> np.ndarray:
        """Obtiene el embedding del texto completo."""
        return self.sentence_model.encode(texto)
    
    def _generar_explicacion_detallada(self, pregunta: str, respuesta: str, 
                                  concepto: str, contexto: str) -> str:
        """Genera una explicación detallada para una pregunta de alternativas."""
        # 1. Encontrar la oración que contiene la respuesta
        doc = self.nlp(contexto)
        oracion_respuesta = ""
        for sent in doc.sents:
            if respuesta.lower() in sent.text.lower():
                oracion_respuesta = sent.text
                break
        
        # 2. Generar explicación base
        prompt = (
            f"Explica por qué '{respuesta}' es la respuesta correcta "
            f"a la pregunta '{pregunta}' en el contexto de {concepto}"
        )
        
        inputs = self.t5_tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.t5_model.generate(
            inputs.input_ids,
            max_length=150,
            num_beams=4,
            temperature=0.8,
            no_repeat_ngram_size=2
        )
        
        explicacion_base = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 3. Enriquecer la explicación con evidencia del texto
        doc_respuesta = self.nlp(respuesta)
        palabras_clave = [
            token.text.lower() for token in doc_respuesta 
            if not token.is_stop and not token.is_punct
        ]
        
        evidencias = []
        for sent in doc.sents:
            if any(palabra in sent.text.lower() for palabra in palabras_clave):
                evidencias.append(sent.text)
        
        # 4. Construir explicación enriquecida
        partes_explicacion = []
        
        # Introducción
        partes_explicacion.append(f"La respuesta correcta es '{respuesta}'.")
        
        # Explicación base generada por T5
        partes_explicacion.append(explicacion_base)
        
        # Evidencia del texto
        if oracion_respuesta:
            partes_explicacion.append(f"\nEsto se evidencia en el texto cuando menciona: '{oracion_respuesta}'")
        
        # Información adicional
        if evidencias:
            partes_explicacion.append("\nInformación adicional relevante:")
            for i, evidencia in enumerate(evidencias[:2], 1):
                if evidencia != oracion_respuesta:
                    partes_explicacion.append(f"{i}. {evidencia}")
        
        # Conexión con el concepto
        partes_explicacion.append(f"\nEsta respuesta está directamente relacionada con el concepto de {concepto} porque:")
        
        # Generar explicación de la relación usando T5
        relacion_prompt = f"explica la relación entre {respuesta} y el concepto de {concepto}"
        inputs_relacion = self.t5_tokenizer(relacion_prompt, return_tensors="pt").to(self.device)
        outputs_relacion = self.t5_model.generate(
            inputs_relacion.input_ids,
            max_length=50,
            num_beams=3,
            temperature=0.7
        )
        explicacion_relacion = self.t5_tokenizer.decode(outputs_relacion[0], skip_special_tokens=True)
        partes_explicacion.append(explicacion_relacion)
        
        # 5. Validar y mejorar la coherencia
        explicacion_completa = " ".join(partes_explicacion)
        explicacion_completa = self._mejorar_coherencia_explicacion(explicacion_completa)
        
        return explicacion_completa

    def _mejorar_coherencia_explicacion(self, explicacion: str) -> str:
        """Mejora la coherencia y fluidez de la explicación."""
        # 1. Eliminar repeticiones
        doc = self.nlp(explicacion)
        oraciones = list(doc.sents)
        oraciones_filtradas = []
        contenido_previo = []  # Cambiado de set a lista
        
        for oracion in oraciones:
            # Convertir a formato comparable
            texto_comparable = ' '.join(
                token.lemma_.lower() for token in oracion 
                if not token.is_stop and not token.is_punct
            )
            
            # Verificar si es suficientemente diferente
            es_diferente = True
            for prev in contenido_previo:
                if self._calcular_similitud_textos(texto_comparable, prev) > 0.8:
                    es_diferente = False
                    break
                    
            if es_diferente:
                oraciones_filtradas.append(oracion.text)
                contenido_previo.append(texto_comparable)
        
        # 2. Mejorar transiciones
        conectores = [
            "Además,", "Por otra parte,", "Asimismo,", "En particular,",
            "Es importante destacar que", "Cabe mencionar que",
            "En este contexto,", "De esta manera,", "Por lo tanto,"
        ]
        
        texto_mejorado = []
        for i, oracion in enumerate(oraciones_filtradas):
            if i > 0 and not any(oracion.startswith(c) for c in conectores):
                texto_mejorado.append(random.choice(conectores))
            texto_mejorado.append(oracion.strip())
        
        # 3. Asegurar terminación apropiada
        if texto_mejorado:
            if not texto_mejorado[-1].rstrip().endswith(('.', '!', '?')):
                texto_mejorado[-1] = texto_mejorado[-1] + '.'
        
        # 4. Manejar caso de texto vacío
        if not texto_mejorado:
            return explicacion  # Devolver texto original si no hay mejoras
            
        return ' '.join(texto_mejorado)

    def _calcular_similitud_textos(self, texto1: str, texto2: str) -> float:
        """Calcula la similitud entre dos textos usando embeddings de manera segura."""
        try:
            # Asegurar que los textos no estén vacíos
            if not texto1.strip() or not texto2.strip():
                return 0.0
                
            # Calcular embeddings
            embedding1 = self.sentence_model.encode(texto1)
            embedding2 = self.sentence_model.encode(texto2)
            
            # Calcular similitud
            similitud = float(cosine_similarity([embedding1], [embedding2])[0][0])
            
            # Asegurar valor válido
            return max(0.0, min(1.0, similitud))
        except Exception as e:
            print(f"Error calculando similitud: {str(e)}")
            return 0.0

    def _generar_pregunta_alternativas(self, texto: str, conceptos: List[str],
                                     embedding_texto: np.ndarray,
                                     oraciones: List[str]) -> Optional[Dict[str, Any]]:
        """Genera una pregunta de alternativas usando T5 y BERT."""
        # Seleccionar concepto y contexto relevante
        concepto = random.choice(conceptos)
        contexto = self._seleccionar_contexto_relevante(
            concepto, texto, oraciones, embedding_texto)
        
        # Generar pregunta usando T5
        pregunta = self._generar_pregunta_t5(concepto, contexto)
        if not pregunta:
            return None
            
        # Obtener respuesta usando BERT QA
        respuesta = self.bert_qa(
            question=pregunta,
            context=contexto
        )["answer"]
        
        # Generar distractores
        distractores = self._generar_distractores_avanzados(
            pregunta, respuesta, concepto, texto, embedding_texto)
        if len(distractores) < 3:
            return None
            
        opciones = [respuesta] + distractores[:3]
        random.shuffle(opciones)
        
        # Generar explicación
        explicacion = self._generar_explicacion_detallada(
            pregunta, respuesta, concepto, contexto)
            
        return {
            "tipo": "alternativas",
            "pregunta": pregunta,
            "opciones": opciones,
            "respuesta_correcta": respuesta,
            "explicacion": explicacion
        }

    def _generar_pregunta_t5(self, concepto: str, contexto: str) -> str:
        """Genera una pregunta usando T5 con prompts específicos."""
        prompts = [
            f"genera una pregunta sobre {concepto} basada en: {contexto}",
            f"crea una pregunta que evalúe el concepto de {concepto} en el texto: {contexto}",
            f"formula una pregunta para evaluar la comprensión de {concepto} dado: {contexto}"
        ]
        
        mejor_pregunta = None
        mejor_score = -1
        
        for prompt in prompts:
            inputs = self.t5_tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.t5_model.generate(
                inputs.input_ids,
                max_length=64,
                num_return_sequences=3,
                num_beams=5,
                no_repeat_ngram_size=2,
                temperature=0.7
            )
            
            for output in outputs:
                pregunta = self.t5_tokenizer.decode(output, skip_special_tokens=True)
                score = self._evaluar_calidad_pregunta(pregunta, contexto)
                
                if score > mejor_score:
                    mejor_pregunta = pregunta
                    mejor_score = score
                    
        return mejor_pregunta

    def _evaluar_calidad_pregunta(self, pregunta: str, contexto: str) -> float:
        """Evalúa la calidad de una pregunta generada."""
        # Criterios de evaluación
        tiene_palabra_pregunta = any(w in pregunta.lower() 
                                   for w in ['qué', 'cuál', 'cómo', 'por qué', 'dónde'])
        longitud_adecuada = 10 <= len(pregunta.split()) <= 20
        tiene_signos_pregunta = '?' in pregunta
        
        # Calcular similitud semántica con el contexto
        similitud = cosine_similarity(
            [self.sentence_model.encode(pregunta)],
            [self.sentence_model.encode(contexto)]
        )[0][0]
        
        # Combinar criterios
        score = (0.4 * similitud +
                0.2 * tiene_palabra_pregunta +
                0.2 * longitud_adecuada +
                0.2 * tiene_signos_pregunta)
                
        return float(score)

    def _generar_distractores_avanzados(self, pregunta: str, respuesta: str,
                                  concepto: str, texto: str,
                                  embedding_texto: np.ndarray) -> List[str]:
        """Genera distractores usando múltiples técnicas avanzadas."""
        # Usamos una lista en lugar de un set para almacenar distractores
        distractores = []
        
        try:
            # 1. Extraer candidatos del texto usando análisis sintáctico
            doc = self.nlp(texto)
            candidatos_sintacticos = self._extraer_candidatos_sintacticos(
                doc, respuesta, concepto)
                
            # 2. Generar candidatos usando T5 (con manejo de errores)
            try:
                candidatos_t5 = self._generar_candidatos_t5(pregunta, respuesta, concepto)
            except Exception as e:
                print(f"Error generando candidatos T5: {str(e)}")
                candidatos_t5 = []
                
            # 3. Encontrar frases semánticamente similares
            candidatos_semanticos = self._encontrar_frases_similares(
                respuesta, texto, embedding_texto)
                
            # Combinar todos los candidatos
            todos_candidatos = candidatos_sintacticos + candidatos_t5 + candidatos_semanticos
            
            # Si tenemos muy pocos candidatos, generar algunos adicionales
            if len(todos_candidatos) < 3:
                candidatos_extra = self._generar_candidatos_alternativos(
                    respuesta, concepto, texto)
                todos_candidatos.extend(candidatos_extra)
            
            # Filtrar y seleccionar los mejores distractores
            for candidato in todos_candidatos:
                if (isinstance(candidato, str) and  # Verificar que sea string
                    candidato not in distractores and  # Evitar duplicados
                    self._es_distractor_valido(candidato, respuesta, pregunta)):
                    distractores.append(candidato)
                    if len(distractores) >= 5:
                        break
            
            # Si aún no tenemos suficientes distractores, generar algunos genéricos
            while len(distractores) < 3:
                distractor_generico = self._generar_distractor_generico(
                    respuesta, concepto)
                if distractor_generico and distractor_generico not in distractores:
                    distractores.append(distractor_generico)
            
            # Asegurar que tenemos exactamente 3 distractores
            return distractores[:3]
            
        except Exception as e:
            print(f"Error en generación de distractores: {str(e)}")
            # En caso de error, devolver distractores genéricos
            return [
                f"Alternativa incorrecta para {concepto}",
                f"Otra opción relacionada con {concepto}",
                f"Una variante de {concepto}"
            ]

    def _generar_candidatos_alternativos(self, respuesta: str, concepto: str, 
                                    texto: str) -> List[str]:
        """Genera candidatos adicionales cuando hay pocos distractores disponibles."""
        candidatos = []
        
        # 1. Usar sinónimos y antónimos comunes
        doc = self.nlp(respuesta)
        for token in doc:
            if token.pos_ in ["NOUN", "VERB", "ADJ"]:
                # Aquí podrías usar un diccionario de sinónimos/antónimos
                # Por ahora usamos una versión simplificada
                if token.text in self.palabras_relacionadas:
                    candidatos.extend(self.palabras_relacionadas[token.text])
        
        # 2. Modificar la estructura de la respuesta
        palabras = respuesta.split()
        if len(palabras) > 1:
            # Cambiar orden de palabras
            candidatos.append(" ".join(reversed(palabras)))
            # Sustituir palabras clave
            for i, palabra in enumerate(palabras):
                if palabra.lower() != concepto.lower():
                    nueva_palabra = self._obtener_palabra_similar(palabra)
                    if nueva_palabra:
                        palabras_mod = palabras.copy()
                        palabras_mod[i] = nueva_palabra
                        candidatos.append(" ".join(palabras_mod))
        
        return candidatos

    def _obtener_palabra_similar(self, palabra: str) -> Optional[str]:
        """Obtiene una palabra similar a la dada."""
        # Diccionario simple de palabras similares (debería expandirse)
        similares = {
            'grande': ['enorme', 'extenso', 'amplio'],
            'pequeño': ['diminuto', 'reducido', 'chico'],
            'importante': ['relevante', 'significativo', 'crucial'],
            'difícil': ['complejo', 'complicado', 'arduo'],
            'fácil': ['simple', 'sencillo', 'básico']
            # Agregar más palabras según necesidad
        }
        
        palabra = palabra.lower()
        if palabra in similares:
            return random.choice(similares[palabra])
        return None

    def _generar_distractor_generico(self, respuesta: str, concepto: str) -> Optional[str]:
        """Genera un distractor genérico cuando no hay suficientes candidatos."""
        plantillas = [
            f"Otro aspecto de {concepto}",
            f"Una variante de {concepto}",
            f"Un elemento similar a {respuesta}",
            f"Una alternativa a {concepto}",
            f"Un caso especial de {concepto}"
        ]
        
        return random.choice(plantillas)

    def _extraer_candidatos_sintacticos(self, doc, respuesta: str, concepto: str) -> List[str]:
        """Extrae candidatos basados en análisis sintáctico."""
        candidatos = []
        respuesta_doc = self.nlp(respuesta)
        
        # Obtener el patrón sintáctico de la respuesta
        respuesta_patron = None
        for token in respuesta_doc:
            if token.dep_ == "ROOT":
                respuesta_patron = token.pos_
                break
        
        if not respuesta_patron:
            return candidatos
            
        # Buscar frases con estructura similar
        for sent in doc.sents:
            for chunk in sent.noun_chunks:
                # Verificar que el chunk no sea igual a la respuesta o al concepto
                if (chunk.text.lower() != respuesta.lower() and 
                    concepto.lower() not in chunk.text.lower()):
                    
                    # Verificar patrón sintáctico similar
                    chunk_patron = None
                    for token in chunk:
                        if token.dep_ == "ROOT" or token.dep_ == "HEAD":
                            chunk_patron = token.pos_
                            break
                            
                    if chunk_patron == respuesta_patron:
                        candidatos.append(chunk.text)
        
        # Extraer también entidades nombradas similares
        for ent in doc.ents:
            if (ent.text.lower() != respuesta.lower() and 
                concepto.lower() not in ent.text.lower() and
                len(ent.text.split()) <= len(respuesta.split()) + 2):
                candidatos.append(ent.text)
        
        # Eliminar duplicados manteniendo el orden
        return list(dict.fromkeys(candidatos))

    def _obtener_patron_sintactico(self, doc_or_span) -> str:
        """Obtiene el patrón sintáctico de un texto de forma más robusta."""
        patron = []
        for token in doc_or_span:
            if token.dep_ == "ROOT" or token.dep_ in ["nsubj", "dobj", "attr"]:
                patron.append(f"{token.pos_}_{token.dep_}")
        return "+".join(patron) if patron else "NOUN_nsubj"  # Patrón por defecto

    def _generar_candidatos_t5(self, pregunta: str, respuesta: str,
                              concepto: str) -> List[str]:
        """Genera candidatos usando T5."""
        prompt = (f"genera alternativas incorrectas similares a '{respuesta}' "
                 f"para la pregunta: {pregunta}")
        
        inputs = self.t5_tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.t5_model.generate(
            inputs.input_ids,
            max_length=50,
            num_return_sequences=3,
            num_beams=5,
            temperature=0.8
        )
        
        return [self.t5_tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs]

    def _encontrar_frases_similares(self, respuesta: str, texto: str,
                                  embedding_texto: np.ndarray) -> List[str]:
        """Encuentra frases semánticamente similares a la respuesta."""
        doc = self.nlp(texto)
        frases = [sent.text for sent in doc.sents]
        
        # Calcular similitud con cada frase
        embedding_respuesta = self.sentence_model.encode(respuesta)
        embeddings_frases = self.sentence_model.encode(frases)
        
        similitudes = cosine_similarity([embedding_respuesta], embeddings_frases)[0]
        
        # Seleccionar frases más similares pero no idénticas
        candidatos = []
        for i in np.argsort(similitudes)[-5:]:
            if similitudes[i] > 0.5 and similitudes[i] < 0.95:
                candidatos.append(frases[i])
                
        return candidatos

    def _es_distractor_valido(self, candidato: str, respuesta: str,
                             pregunta: str) -> bool:
        """Valida si un distractor es apropiado."""
        if not candidato or candidato.lower() == respuesta.lower():
            return False
            
        # Verificar longitud similar
        if abs(len(candidato.split()) - len(respuesta.split())) > 3:
            return False
            
        # Verificar que no sea subconjunto o superconjunto
        if (respuesta.lower() in candidato.lower() or
            candidato.lower() in respuesta.lower()):
            return False
            
        # Verificar similitud semántica
        similitud = cosine_similarity(
            [self.sentence_model.encode(candidato)],
            [self.sentence_model.encode(respuesta)]
        )[0][0]
        
        return 0.3 < similitud < 0.8

    def _seleccionar_contexto_relevante(self, concepto: str, texto: str,
                                      oraciones: List[str],
                                      embedding_texto: np.ndarray) -> str:
        """Selecciona el contexto más relevante para generar una pregunta."""
        # Encontrar oraciones que contienen el concepto
        oraciones_relevantes = []
        for oracion in oraciones:
            if concepto.lower() in oracion.lower():
                oraciones_relevantes.append(oracion)
                
        # Si no hay oraciones con el concepto, usar similitud semántica
        if not oraciones_relevantes:
            embedding_concepto = self.sentence_model.encode(concepto)
            embeddings_oraciones = self.sentence_model.encode(oraciones)
            
            similitudes = cosine_similarity([embedding_concepto], embeddings_oraciones)[0]
            indices_relevantes = np.argsort(similitudes)[-3:]  # Top 3 oraciones
            oraciones_relevantes = [oraciones[i] for i in indices_relevantes]
        
        # Expandir contexto para incluir oraciones vecinas
        contexto_expandido = self._expandir_contexto(oraciones_relevantes, oraciones)
        
        return ' '.join(contexto_expandido)

    def _expandir_contexto(self, oraciones_relevantes: List[str],
                      todas_oraciones: List[str]) -> List[str]:
        """Expande el contexto incluyendo oraciones vecinas relevantes."""
        # Usar lista en lugar de set para el contexto
        contexto = []
        contexto.extend(oraciones_relevantes)
        
        # Obtener índices de oraciones relevantes
        indices_relevantes = []
        for i, oracion in enumerate(todas_oraciones):
            if oracion in oraciones_relevantes:
                indices_relevantes.append(i)
        
        # Agregar oraciones vecinas que podrían ser relevantes
        oraciones_vecinas = set()  # Usamos set solo para oraciones individuales
        for idx in indices_relevantes:
            # Agregar oración anterior si existe
            if idx > 0:
                oraciones_vecinas.add(todas_oraciones[idx - 1])
            # Agregar oración siguiente si existe    
            if idx < len(todas_oraciones) - 1:
                oraciones_vecinas.add(todas_oraciones[idx + 1])
        
        # Agregar oraciones vecinas a la lista de contexto
        for oracion in todas_oraciones:
            if oracion in oraciones_vecinas and oracion not in contexto:
                contexto.append(oracion)
        
        # Mantener el orden original de las oraciones
        contexto_ordenado = [
            oracion for oracion in todas_oraciones
            if oracion in contexto
        ]
        
        return contexto_ordenado

    def _generar_pregunta_vf(self, texto: str, conceptos: List[str],
                            embedding_texto: np.ndarray,
                            oraciones: List[str]) -> Optional[Dict[str, Any]]:
        """Genera una pregunta de verdadero/falso usando modelos neuronales."""
        # Seleccionar un concepto y su contexto
        concepto = random.choice(conceptos)
        contexto = self._seleccionar_contexto_relevante(
            concepto, texto, oraciones, embedding_texto)
        
        # Determinar si será verdadero o falso
        es_verdadero = random.choice([True, False])
        
        if es_verdadero:
            afirmacion = self._generar_afirmacion_verdadera(contexto, concepto)
        else:
            afirmacion = self._generar_afirmacion_falsa(contexto, concepto)
            
        if not afirmacion:
            return None
            
        explicacion = self._generar_explicacion_vf(
            afirmacion, contexto, es_verdadero, concepto)
            
        return {
            "tipo": "verdadero_falso",
            "pregunta": afirmacion,
            "respuesta_correcta": "Verdadero" if es_verdadero else "Falso",
            "explicacion": explicacion
        }

    def _generar_afirmacion_verdadera(self, contexto: str, concepto: str) -> Optional[str]:
        """Genera una afirmación verdadera basada en el contexto."""
        # Usar T5 para generar afirmaciones candidatas
        prompt = f"genera una afirmación verdadera sobre {concepto} basada en: {contexto}"
        
        inputs = self.t5_tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.t5_model.generate(
            inputs.input_ids,
            max_length=64,
            num_return_sequences=3,
            num_beams=5,
            temperature=0.7
        )
        
        candidatos = [
            self.t5_tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        
        # Evaluar y seleccionar la mejor afirmación
        mejor_afirmacion = None
        mejor_score = -1
        
        for afirmacion in candidatos:
            score = self._evaluar_calidad_afirmacion(afirmacion, contexto, True)
            if score > mejor_score:
                mejor_afirmacion = afirmacion
                mejor_score = score
                
        return mejor_afirmacion

    def _generar_afirmacion_falsa(self, contexto: str, concepto: str) -> Optional[str]:
        """Genera una afirmación falsa pero plausible."""
        # Estrategias para generar afirmaciones falsas
        estrategias = [
            self._modificar_hechos_clave,
            self._alterar_relaciones,
            self._cambiar_cantidades,
            self._invertir_logica
        ]
        
        # Intentar cada estrategia hasta obtener una buena afirmación
        for estrategia in random.sample(estrategias, len(estrategias)):
            afirmacion = estrategia(contexto, concepto)
            if afirmacion:
                score = self._evaluar_calidad_afirmacion(afirmacion, contexto, False)
                if score > 0.7:  # Umbral de calidad
                    return afirmacion
                    
        return None
    
    def _obtener_verbo_opuesto(self, verbo: str) -> str:
        """Obtiene el opuesto de un verbo."""
        verbos_opuestos = {
            'aumentar': 'disminuir',
            'subir': 'bajar',
            'crecer': 'decrecer',
            'mejorar': 'empeorar',
            'incluir': 'excluir',
            'permitir': 'prohibir',
            'aceptar': 'rechazar',
            'aprobar': 'desaprobar',
            'construir': 'destruir',
            'crear': 'eliminar',
            'abrir': 'cerrar',
            'comenzar': 'terminar',
            'iniciar': 'finalizar',
            'entrar': 'salir',
            'ganar': 'perder',
            'aparecer': 'desaparecer',
            'acelerar': 'frenar',
            'activar': 'desactivar',
            'agregar': 'quitar',
            'ayudar': 'obstaculizar',
            # Agregar más pares según necesidad
        }
        
        verbo = verbo.lower()
        # Buscar en verbos opuestos
        if verbo in verbos_opuestos:
            return verbos_opuestos[verbo]
            
        # Si no se encuentra un opuesto, negar el verbo
        return f"no {verbo}"

    def _modificar_hechos_clave(self, contexto: str, concepto: str) -> Optional[str]:
        """Modifica hechos clave del contexto para crear una afirmación falsa."""
        doc = self.nlp(contexto)
        
        # Extraer hechos clave (sujeto-verbo-objeto)
        hechos = []
        for sent in doc.sents:
            for token in sent:
                if token.dep_ == "ROOT":  # Verbo principal
                    sujeto = None
                    objeto = None
                    
                    # Encontrar sujeto
                    for hijo in token.children:
                        if hijo.dep_ == "nsubj":
                            sujeto = hijo
                            break
                            
                    # Encontrar objeto
                    for hijo in token.children:
                        if hijo.dep_ in ["dobj", "attr"]:
                            objeto = hijo
                            break
                            
                    if sujeto and objeto:
                        hechos.append({
                            "sujeto": sujeto,
                            "verbo": token,
                            "objeto": objeto
                        })
        
        if not hechos:
            return None
            
        # Seleccionar un hecho al azar y modificarlo
        hecho = random.choice(hechos)
        
        # Diferentes tipos de modificaciones
        modificaciones = [
            lambda h: f"{h['objeto'].text} {h['verbo'].text} {h['sujeto'].text}",  # Invertir sujeto y objeto
            lambda h: f"{h['sujeto'].text} no {h['verbo'].text} {h['objeto'].text}",  # Negar el verbo
            lambda h: f"{h['sujeto'].text} {self._obtener_verbo_opuesto(h['verbo'].text)} {h['objeto'].text}"  # Usar verbo opuesto
        ]
        
        afirmacion_modificada = random.choice(modificaciones)(hecho)
        return afirmacion_modificada

    def _alterar_relaciones(self, contexto: str, concepto: str) -> Optional[str]:
        """Altera las relaciones entre entidades en el contexto."""
        doc = self.nlp(contexto)
        entidades = list(doc.ents)
        
        if len(entidades) < 2:
            return None
            
        # Seleccionar dos entidades al azar
        ent1, ent2 = random.sample(entidades, 2)
        
        # Patrones para crear relaciones falsas
        patrones = [
            f"{ent1.text} no tiene relación con {ent2.text}",
            f"{ent1.text} es opuesto a {ent2.text}",
            f"{ent1.text} reemplaza a {ent2.text}",
            f"A diferencia de {ent1.text}, {ent2.text}",
            f"{ent1.text} excluye a {ent2.text}"
        ]
        
        return random.choice(patrones)

    def _cambiar_cantidades(self, contexto: str, concepto: str) -> Optional[str]:
        """Modifica valores numéricos manteniendo la plausibilidad."""
        doc = self.nlp(contexto)
        numeros = []
        
        for token in doc:
            if token.like_num:
                try:
                    num = float(token.text)
                    numeros.append((token, num))
                except ValueError:
                    continue
        
        if not numeros:
            return None
            
        token, num = random.choice(numeros)
        
        # Modificar el número de manera plausible
        if num.is_integer():
            nuevo_num = int(num * random.uniform(1.5, 3.0))
        else:
            nuevo_num = round(num * random.uniform(1.5, 3.0), 2)
            
        contexto_modificado = contexto.replace(token.text, str(nuevo_num))
        return contexto_modificado

    def _invertir_logica(self, contexto: str, concepto: str) -> Optional[str]:
        """Invierte la lógica de una afirmación manteniendo la coherencia."""
        doc = self.nlp(contexto)
        
        # Palabras y frases para inversión lógica
        inversiones = {
            'siempre': 'nunca',
            'todos': 'ninguno',
            'debe': 'no debe',
            'puede': 'no puede',
            'es necesario': 'no es necesario',
            'causa': 'no causa',
            'resulta en': 'no resulta en',
            'implica': 'no implica'
        }
        
        texto_modificado = contexto
        modificado = False
        
        for original, inversion in inversiones.items():
            if original in texto_modificado.lower():
                texto_modificado = texto_modificado.replace(original, inversion)
                texto_modificado = texto_modificado.replace(
                    original.capitalize(), inversion.capitalize())
                modificado = True
                break
                
        if not modificado:
            return None
            
        return texto_modificado

    def _evaluar_calidad_afirmacion(self, afirmacion: str, contexto: str,
                                  es_verdadera: bool) -> float:
        """Evalúa la calidad de una afirmación generada."""
        # Criterios de evaluación
        coherencia_sintactica = self._evaluar_coherencia_sintactica(afirmacion)
        similitud_semantica = self._evaluar_similitud_semantica(
            afirmacion, contexto)
        plausibilidad = self._evaluar_plausibilidad(afirmacion, es_verdadera)
        
        # Pesos para cada criterio
        weights = {
            'coherencia': 0.3,
            'similitud': 0.4,
            'plausibilidad': 0.3
        }
        
        score = (weights['coherencia'] * coherencia_sintactica +
                weights['similitud'] * similitud_semantica +
                weights['plausibilidad'] * plausibilidad)
                
        return score

    def _evaluar_coherencia_sintactica(self, texto: str) -> float:
        """Evalúa la coherencia sintáctica de un texto."""
        doc = self.nlp(texto)
        
        # Verificar estructura básica
        tiene_verbo = False
        tiene_sujeto = False
        
        for token in doc:
            if token.pos_ == "VERB":
                tiene_verbo = True
            if token.dep_ == "nsubj":
                tiene_sujeto = True
                
        estructura_basica = tiene_verbo and tiene_sujeto
        
        # Verificar concordancia
        concordancia = True
        for token in doc:
            if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                if token.morph.get("Number") != token.head.morph.get("Number"):
                    concordancia = False
                    
        return float(estructura_basica and concordancia)

    def _evaluar_similitud_semantica(self, texto1: str, texto2: str) -> float:
        """Evalúa la similitud semántica entre dos textos."""
        embedding1 = self.sentence_model.encode(texto1)
        embedding2 = self.sentence_model.encode(texto2)
        
        similitud = cosine_similarity([embedding1], [embedding2])[0][0]
        return float(similitud)

    def _evaluar_plausibilidad(self, afirmacion: str, es_verdadera: bool) -> float:
        """Evalúa qué tan plausible es una afirmación."""
        doc = self.nlp(afirmacion)
        
        # Verificar coherencia lógica
        tiene_contradicciones = False
        palabras_incompatibles = [
            ('siempre', 'nunca'),
            ('todo', 'nada'),
            ('completamente', 'parcialmente')
        ]
        
        texto = afirmacion.lower()
        for par in palabras_incompatibles:
            if par[0] in texto and par[1] in texto:
                tiene_contradicciones = True
                
        # Verificar estructura temporal coherente
        tiempos_verbales = [token.morph.get("Tense") for token in doc if token.pos_ == "VERB"]
        coherencia_temporal = len(set(tiempos_verbales)) <= 2
        
        # Calcular score
        score = (
            0.4 * (not tiene_contradicciones) +
            0.3 * coherencia_temporal +
            0.3 * (len(afirmacion.split()) > 3)
        )
        
        # Ajustar según si debe ser verdadera o falsa
        if not es_verdadera:
            score = 1 - (score * 0.5)  # Mantener plausibilidad pero no demasiada
            
        return float(score)

    def _generar_explicacion_vf(self, afirmacion: str, contexto: str,
                               es_verdadero: bool, concepto: str) -> str:
        """Genera una explicación detallada para una pregunta de verdadero/falso."""
        if es_verdadero:
            prompt = (f"Explica por qué la siguiente afirmación es verdadera: "
                     f"'{afirmacion}' en el contexto de {concepto}")
        else:
            prompt = (f"Explica por qué la siguiente afirmación es falsa: "
                     f"'{afirmacion}' considerando el contexto: {contexto}")
            
        # Generar explicación base usando T5
        inputs = self.t5_tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.t5_model.generate(
            inputs.input_ids,
            max_length=150,
            num_beams=4,
            temperature=0.8,
            no_repeat_ngram_size=2
        )
        
        explicacion_base = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Enriquecer la explicación con información adicional
        explicacion_enriquecida = self._enriquecer_explicacion(
            explicacion_base, afirmacion, contexto, es_verdadero, concepto)
        
        return explicacion_enriquecida

    def _enriquecer_explicacion(self, explicacion: str, afirmacion: str,
                               contexto: str, es_verdadero: bool,
                               concepto: str) -> str:
        """Enriquece una explicación con detalles adicionales y evidencia."""
        doc_contexto = self.nlp(contexto)
        doc_afirmacion = self.nlp(afirmacion)
        
        # Encontrar evidencia de apoyo en el contexto
        evidencias = self._encontrar_evidencias(doc_contexto, doc_afirmacion)
        
        # Identificar conceptos relacionados
        conceptos_relacionados = self._identificar_conceptos_relacionados(
            concepto, contexto)
        
        # Construir explicación enriquecida
        partes_explicacion = [explicacion]
        
        # Agregar evidencia específica
        if evidencias:
            partes_explicacion.append("\nEvidencia específica del texto:")
            for i, evidencia in enumerate(evidencias, 1):
                partes_explicacion.append(f"{i}. {evidencia}")
                
        # Agregar conexiones conceptuales
        if conceptos_relacionados:
            partes_explicacion.append("\nConceptos relacionados relevantes:")
            for concepto_rel, relacion in conceptos_relacionados:
                partes_explicacion.append(f"- {concepto_rel}: {relacion}")
                
        # Agregar conclusión
        partes_explicacion.append(self._generar_conclusion_explicacion(
            es_verdadero, afirmacion, concepto))
        
        return "\n".join(partes_explicacion)

    def _encontrar_evidencias(self, doc_contexto: spacy.tokens.Doc,
                            doc_afirmacion: spacy.tokens.Doc) -> List[str]:
        """Encuentra evidencias en el contexto que apoyan o contradicen la afirmación."""
        evidencias = []
        
        # Extraer palabras clave de la afirmación
        palabras_clave = set()
        for token in doc_afirmacion:
            if not token.is_stop and not token.is_punct:
                palabras_clave.add(token.lemma_)
                
        # Buscar oraciones relevantes en el contexto
        for sent in doc_contexto.sents:
            palabras_coincidentes = set()
            for token in sent:
                if token.lemma_ in palabras_clave:
                    palabras_coincidentes.add(token.lemma_)
                    
            # Si la oración contiene suficientes palabras clave
            if len(palabras_coincidentes) >= 2:
                evidencias.append(sent.text)
                
        # Limitar a las 3 evidencias más relevantes
        return evidencias[:3]

    def _identificar_conceptos_relacionados(self, concepto: str,
                                          contexto: str) -> List[Tuple[str, str]]:
        """Identifica conceptos relacionados y su relación con el concepto principal."""
        doc = self.nlp(contexto)
        conceptos_relacionados = []
        
        # Extraer sustantivos y entidades nombradas
        candidatos = set()
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"] and token.text.lower() != concepto.lower():
                candidatos.add(token.text)
                
        for ent in doc.ents:
            if ent.text.lower() != concepto.lower():
                candidatos.add(ent.text)
                
        # Calcular similitud semántica y encontrar relaciones
        embedding_concepto = self.sentence_model.encode(concepto)
        
        for candidato in candidatos:
            embedding_candidato = self.sentence_model.encode(candidato)
            similitud = cosine_similarity([embedding_concepto], [embedding_candidato])[0][0]
            
            if similitud > 0.4:  # Umbral de similitud
                relacion = self._determinar_relacion(concepto, candidato, contexto)
                conceptos_relacionados.append((candidato, relacion))
                
        # Ordenar por relevancia y limitar cantidad
        conceptos_relacionados.sort(key=lambda x: len(x[1]), reverse=True)
        return conceptos_relacionados[:3]

    def _determinar_relacion(self, concepto1: str, concepto2: str,
                           contexto: str) -> str:
        """Determina la relación entre dos conceptos basándose en el contexto."""
        # Patrones de relación comunes
        patrones_relacion = [
            (r"{}\s+es\s+(?:un|una|el|la)\s+{}", "es un tipo de"),
            (r"{}\s+tiene\s+{}", "es parte de"),
            (r"{}\s+contiene\s+{}", "contiene"),
            (r"{}\s+implica\s+{}", "implica"),
            (r"{}\s+causa\s+{}", "causa"),
            (r"{}\s+afecta\s+{}", "afecta a"),
            (r"{}\s+se\s+relaciona\s+con\s+{}", "está relacionado con")
        ]
        
        # Buscar patrones en ambas direcciones
        for patron, tipo_relacion in patrones_relacion:
            patron_forward = patron.format(
                re.escape(concepto1), re.escape(concepto2))
            patron_backward = patron.format(
                re.escape(concepto2), re.escape(concepto1))
            
            if re.search(patron_forward, contexto, re.IGNORECASE):
                return tipo_relacion
            elif re.search(patron_backward, contexto, re.IGNORECASE):
                return f"es {tipo_relacion} por"
                
        # Si no se encuentra un patrón específico, usar relación genérica
        return "está relacionado con"

    def _generar_conclusion_explicacion(self, es_verdadero: bool,
                                      afirmacion: str, concepto: str) -> str:
        """Genera una conclusión apropiada para la explicación."""
        if es_verdadero:
            plantillas = [
                f"En conclusión, la afirmación sobre {concepto} es correcta según el contexto proporcionado.",
                f"Por lo tanto, podemos confirmar que esta afirmación sobre {concepto} es verdadera.",
                f"La evidencia presentada respalda claramente esta afirmación sobre {concepto}."
            ]
        else:
            plantillas = [
                f"En conclusión, la afirmación sobre {concepto} no es precisa según el contexto dado.",
                f"Por lo tanto, esta afirmación sobre {concepto} debe considerarse falsa.",
                f"La evidencia presentada contradice esta afirmación sobre {concepto}."
            ]
            
        return random.choice(plantillas)

    def procesar_json_entrada(self, ruta_json: str) -> List[Dict[str, Any]]:
        """Procesa el archivo JSON de entrada y genera cuestionarios."""
        try:
            # Verificar y cargar el JSON
            if not os.path.exists(ruta_json):
                raise FileNotFoundError(f"No se encontró el archivo: {ruta_json}")
            
            with open(ruta_json, 'r', encoding='utf-8') as f:
                datos = json.load(f)
            
            # Validar estructura del JSON
            if 'quiz' not in datos or not isinstance(datos['quiz'], list):
                raise ValueError("Formato de JSON inválido: debe contener una lista 'quiz'")
            
            cuestionarios = []
            for item in datos['quiz']:
                try:
                    # Verificar campos requeridos
                    if not all(campo in item for campo in ['texto', 'materia', 'fuente']):
                        print(f"Advertencia: item ignorado por falta de campos requeridos: {item}")
                        continue
                    
                    # Generar cuestionario
                    cuestionario = self.generar_cuestionario(
                        texto=item['texto'],
                        materia=item['materia'],
                        fuente=item['fuente']
                    )
                    
                    # Verificar calidad del cuestionario
                    if cuestionario and self._validar_cuestionario(cuestionario):
                        # Convertir todo a tipos básicos para asegurar serialización JSON
                        cuestionario_limpio = {
                            'materia': str(cuestionario['materia']),
                            'fuente': str(cuestionario['fuente']),
                            'preguntas': []
                        }
                        
                        for pregunta in cuestionario['preguntas']:
                            pregunta_limpia = {
                                'tipo': str(pregunta['tipo']),
                                'pregunta': str(pregunta['pregunta']),
                                'respuesta_correcta': str(pregunta['respuesta_correcta']),
                                'explicacion': str(pregunta['explicacion'])
                            }
                            
                            if pregunta['tipo'] == 'alternativas':
                                pregunta_limpia['opciones'] = [str(opt) for opt in pregunta['opciones']]
                            
                            cuestionario_limpio['preguntas'].append(pregunta_limpia)
                        
                        cuestionarios.append(cuestionario_limpio)
                
                except Exception as e:
                    print(f"Error procesando item del cuestionario: {str(e)}")
                    continue
            
            return cuestionarios
            
        except json.JSONDecodeError as e:
            print(f"Error decodificando JSON: {str(e)}")
            return []
        except Exception as e:
            print(f"Error procesando JSON de entrada: {str(e)}")
            return []

    def _validar_cuestionario(self, cuestionario: Dict[str, Any]) -> bool:
        """Valida la calidad de un cuestionario generado."""
        if not cuestionario.get('preguntas'):
            return False
            
        for pregunta in cuestionario['preguntas']:
            # Verificar campos requeridos
            if not all(campo in pregunta for campo in ['tipo', 'pregunta', 'respuesta_correcta', 'explicacion']):
                return False
            
            # Verificar tipo de pregunta
            if pregunta['tipo'] not in ['alternativas', 'verdadero_falso']:
                return False
            
            # Validar preguntas de alternativas
            if pregunta['tipo'] == 'alternativas':
                if 'opciones' not in pregunta:
                    return False
                if not isinstance(pregunta['opciones'], list):
                    return False
                if len(pregunta['opciones']) != 4:
                    return False
                if pregunta['respuesta_correcta'] not in pregunta['opciones']:
                    return False
            
            # Validar preguntas de verdadero/falso
            if pregunta['tipo'] == 'verdadero_falso':
                if pregunta['respuesta_correcta'] not in ['Verdadero', 'Falso']:
                    return False
            
            # Validar longitud mínima de explicaciones
            if len(pregunta['explicacion'].split()) < 20:
                return False
        
        return True

    def guardar_cuestionarios(self, cuestionarios: List[Dict[str, Any]],
                             ruta_salida: str) -> None:
        """Guarda los cuestionarios generados en un archivo JSON."""
        try:
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
            
            if not cuestionarios:
                print("Advertencia: No hay cuestionarios para guardar")
                return
            
            # Guardar cuestionarios
            with open(ruta_salida, 'w', encoding='utf-8') as f:
                json.dump({
                    "fecha_generacion": "2024-11-15",
                    "total_cuestionarios": len(cuestionarios),
                    "cuestionarios": cuestionarios
                }, f, ensure_ascii=False, indent=4)
            
            print(f"Cuestionarios guardados exitosamente en {ruta_salida}")
            
        except Exception as e:
            print(f"Error guardando cuestionarios: {str(e)}")