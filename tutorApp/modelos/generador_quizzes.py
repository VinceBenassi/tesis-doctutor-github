# Generador de Quizzes
# por Franco Benassi
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    AutoModelForQuestionAnswering,
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
        
        # Modelo T5 especializado en generación de preguntas
        self.gen_model_name = "google/flan-t5-xl"  # Modelo más grande para mejor comprensión
        self.gen_tokenizer = AutoTokenizer.from_pretrained(self.gen_model_name)
        self.gen_model = T5ForConditionalGeneration.from_pretrained(self.gen_model_name)
        
        # Modelo de comprensión lectora multilingüe
        self.qa_model_name = "deepset/xlm-roberta-large-squad2"
        self.qa_pipeline = pipeline(
            "question-answering",
            model=self.qa_model_name,
            tokenizer=self.qa_model_name
        )
        
        # Pipeline de resumen para identificar conceptos clave
        self.summary_pipeline = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            tokenizer="facebook/bart-large-cnn"
        )
        
        # Modelo SpaCy para análisis semántico avanzado
        self.nlp = spacy.load("es_core_news_lg")
        
    def _extraer_conceptos_clave(self, texto: str) -> List[Dict[str, Any]]:
        """Extrae conceptos clave y sus relaciones del texto"""
        # Primero obtener un resumen para identificar los puntos más importantes
        resumen = self.summary_pipeline(texto, max_length=150, min_length=50)[0]['summary_text']
        
        doc = self.nlp(texto)
        conceptos = []
        
        # Identificar entidades nombradas y sus contextos
        entidades_importantes = [ent for ent in doc.ents if ent.label_ in 
                               ['ORG', 'PERSON', 'EVENT', 'CONCEPT', 'LAW']]
        
        # Identificar frases clave usando análisis sintáctico
        for sent in doc.sents:
            # Buscar estructuras que indiquen definiciones o explicaciones
            if any(token.dep_ in ['ROOT', 'nsubj'] for token in sent):
                concepto = {
                    'texto': sent.text,
                    'entidades': [ent.text for ent in sent.ents],
                    'palabras_clave': [token.text for token in sent 
                                     if token.pos_ in ['NOUN', 'VERB', 'ADJ'] 
                                     and token.is_stop == False],
                    'importancia': len(sent.ents) + len([token for token in sent 
                                                       if token.pos_ in ['NOUN', 'VERB']])
                }
                
                # Analizar relaciones semánticas
                relaciones = []
                for token in sent:
                    if token.dep_ in ['nsubj', 'dobj', 'pobj']:
                        relacion = {
                            'tipo': token.dep_,
                            'palabra': token.text,
                            'raiz': token.head.text
                        }
                        relaciones.append(relacion)
                
                concepto['relaciones'] = relaciones
                conceptos.append(concepto)
        
        return sorted(conceptos, key=lambda x: x['importancia'], reverse=True)

    def _generar_pregunta_analitica(self, concepto: Dict[str, Any], texto_completo: str) -> Dict:
        """Genera una pregunta analítica que requiere comprensión profunda"""
        tipos_pregunta = [
            "¿Qué efectos tiene {} en {}?",
            "¿Cómo se relaciona {} con {}?",
            "¿Por qué es importante {} para {}?",
            "¿Qué diferencias existen entre {} y {}?",
            "¿Qué factores influyen en {}?",
            "¿Cuáles son las principales características de {}?",
            "¿Cómo impacta {} en el desarrollo de {}?",
            "¿Qué papel juega {} en el contexto de {}?"
        ]
        
        try:
            # Seleccionar elementos para la pregunta
            elementos = concepto['palabras_clave']
            if len(elementos) < 2:
                return None
                
            # Construir pregunta base
            plantilla = random.choice(tipos_pregunta)
            if '{}' in plantilla:
                elementos_seleccionados = random.sample(elementos, 
                                                      min(2, plantilla.count('{}')))
                pregunta_base = plantilla.format(*elementos_seleccionados)
            else:
                pregunta_base = plantilla.format(random.choice(elementos))
            
            # Generar contexto para T5
            prompt = f"""
            Contexto: {concepto['texto']}
            
            Genera una respuesta detallada para: {pregunta_base}
            
            La respuesta debe:
            1. Ser específica y precisa
            2. Basarse en el contexto dado
            3. Tener entre 10 y 30 palabras
            """
            
            # Generar respuesta correcta
            inputs = self.gen_tokenizer(prompt, return_tensors="pt", max_length=512, 
                                      truncation=True)
            outputs = self.gen_model.generate(
                inputs.input_ids,
                max_length=100,
                num_beams=5,
                temperature=0.7,
                top_p=0.9,
                no_repeat_ngram_size=2
            )
            respuesta_correcta = self.gen_tokenizer.decode(outputs[0], 
                                                         skip_special_tokens=True)
            
            # Verificar calidad de la respuesta
            if len(respuesta_correcta.split()) < 3:
                return None
                
            # Generar alternativas plausibles
            alternativas = self._generar_alternativas_plausibles(
                pregunta_base,
                respuesta_correcta,
                concepto,
                texto_completo
            )
            
            if len(alternativas) < 4:
                return None
                
            return {
                "pregunta": pregunta_base,
                "opciones": alternativas,
                "respuesta_correcta": respuesta_correcta,
                "tipo": "alternativas"
            }
            
        except Exception as e:
            print(f"Error al generar pregunta analítica: {str(e)}")
            return None

    def _generar_alternativas_plausibles(
        self,
        pregunta: str,
        respuesta_correcta: str,
        concepto: Dict[str, Any],
        texto_completo: str
    ) -> List[str]:
        """Genera alternativas plausibles y engañosas"""
        alternativas = [respuesta_correcta]
        
        # Construir prompt para alternativas
        prompt_base = f"""
        Pregunta: {pregunta}
        Respuesta correcta: {respuesta_correcta}
        
        Genera una respuesta alternativa que:
        1. Sea lógicamente posible pero incorrecta
        2. Use conceptos relacionados del mismo campo
        3. Tenga un estilo y longitud similar a la respuesta correcta
        4. Sea suficientemente diferente de la respuesta correcta
        5. Pueda parecer correcta a primera vista
        
        Conceptos relacionados: {', '.join(concepto['palabras_clave'])}
        """
        
        try:
            for _ in range(5):  # Intentamos generar más alternativas de las necesarias
                inputs = self.gen_tokenizer(prompt_base, return_tensors="pt", 
                                          max_length=512, truncation=True)
                outputs = self.gen_model.generate(
                    inputs.input_ids,
                    max_length=100,
                    temperature=0.8,
                    top_p=0.9,
                    num_beams=4,
                    no_repeat_ngram_size=2
                )
                
                alternativa = self.gen_tokenizer.decode(outputs[0], 
                                                      skip_special_tokens=True)
                
                # Verificar calidad de la alternativa
                if self._es_alternativa_valida(alternativa, alternativas, 
                                             pregunta, respuesta_correcta):
                    alternativas.append(alternativa)
                
                if len(alternativas) >= 4:
                    break
                    
            # Si no tenemos suficientes alternativas, generamos más usando variaciones
            while len(alternativas) < 4:
                nueva_alt = self._generar_variacion(respuesta_correcta, 
                                                  concepto['palabras_clave'])
                if self._es_alternativa_valida(nueva_alt, alternativas, 
                                             pregunta, respuesta_correcta):
                    alternativas.append(nueva_alt)
            
            random.shuffle(alternativas)
            return alternativas[:4]
            
        except Exception as e:
            print(f"Error al generar alternativas: {str(e)}")
            return []

    def _es_alternativa_valida(
        self,
        alternativa: str,
        alternativas_existentes: List[str],
        pregunta: str,
        respuesta_correcta: str
    ) -> bool:
        """Verifica si una alternativa es válida y suficientemente diferente"""
        if not alternativa or len(alternativa.split()) < 3:
            return False
            
        alt_doc = self.nlp(alternativa)
        
        # No debe ser muy similar a la pregunta
        if self.nlp(pregunta).similarity(alt_doc) > 0.7:
            return False
            
        # No debe ser muy similar a la respuesta correcta
        if self.nlp(respuesta_correcta).similarity(alt_doc) > 0.8:
            return False
            
        # No debe ser muy similar a otras alternativas
        for existente in alternativas_existentes:
            if self.nlp(existente).similarity(alt_doc) > 0.7:
                return False
                
        # Debe mantener coherencia gramatical
        doc = self.nlp(alternativa)
        if not any(token.pos_ in ['VERB'] for token in doc):
            return False
            
        return True

    def _generar_variacion(self, texto_base: str, conceptos_relacionados: List[str]) -> str:
        """Genera una variación de un texto manteniendo estructura pero cambiando significado"""
        doc = self.nlp(texto_base)
        
        # Identificar estructura sintáctica
        estructura = [(token.pos_, token.dep_) for token in doc]
        
        # Generar nueva versión manteniendo estructura
        nuevo_texto = []
        for token in doc:
            if token.pos_ in ['NOUN', 'VERB', 'ADJ']:
                # Reemplazar con concepto relacionado o sinónimo
                if random.random() < 0.5 and conceptos_relacionados:
                    nuevo_texto.append(random.choice(conceptos_relacionados))
                else:
                    sinonimos = [t.text for t in token.vocab 
                               if t.is_lower == token.is_lower and 
                               t.prob >= -15 and t.prob <= -5]
                    if sinonimos:
                        nuevo_texto.append(random.choice(sinonimos))
                    else:
                        nuevo_texto.append(token.text)
            else:
                nuevo_texto.append(token.text)
        
        return ' '.join(nuevo_texto)

    def _generar_pregunta_vf(self, concepto: Dict[str, Any], texto_completo: str) -> Dict:
        """Genera una pregunta de verdadero/falso que requiere análisis"""
        try:
            # Construir afirmación base usando el concepto
            if not concepto['relaciones']:
                return None
                
            relacion = random.choice(concepto['relaciones'])
            contexto = concepto['texto']
            
            prompt = f"""
            Contexto: {contexto}
            
            Genera una afirmación sobre la relación entre {relacion['palabra']} y {relacion['raiz']} que:
            1. Sea específica y verificable
            2. Requiera análisis del contexto
            3. No sea obvia
            4. Se pueda determinar su veracidad
            5. Sea autocontenida
            
            La afirmación debe ser completa y tener sentido por sí misma.
            """
            
            # Generar afirmación
            inputs = self.gen_tokenizer(prompt, return_tensors="pt", max_length=512, 
                                      truncation=True)
            outputs = self.gen_model.generate(
                inputs.input_ids,
                max_length=100,
                temperature=0.7,
                num_beams=4,
                no_repeat_ngram_size=2
            )
            
            afirmacion = self.gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Verificar si la afirmación es verdadera
            verificacion_prompt = f"""
            Contexto: {texto_completo}
            
            Verifica si esta afirmación es verdadera: "{afirmacion}"
            
            Responde solo con "Verdadero" o "Falso".
            """
            
            inputs = self.gen_tokenizer(verificacion_prompt, return_tensors="pt", 
                                      max_length=512, truncation=True)
            outputs = self.gen_model.generate(
                inputs.input_ids,
                max_length=10,
                temperature=0.3
            )
            
            resultado = self.gen_tokenizer.decode(outputs[0], 
                                                skip_special_tokens=True).strip()
            
            if resultado not in ["Verdadero", "Falso"]:
                return None
                
            return {
                "pregunta": afirmacion,
                "opciones": ["Verdadero", "Falso"],
                "respuesta_correcta": resultado,
                "tipo": "verdadero_falso"
            }
            
        except Exception as e:
            print(f"Error al generar pregunta V/F: {str(e)}")
            return None

    def generar_cuestionario(self, ruta_archivo: str) -> Dict:
        """Genera el cuestionario completo a partir del archivo JSON"""
        try:
            with open(ruta_archivo, 'r', encoding='utf-8') as archivo:
                datos = json.load(archivo)
        except Exception as e:
            print(f"Error al cargar el archivo JSON: {str(e)}")
            return None

        cuestionarios = {}
        
        for item in datos['quiz']:
            materia = item['materia']
            texto = item['texto']
            
            if materia not in cuestionarios:
                cuestionarios[materia] = []
            
            # Extraer conceptos clave del texto
            conceptos = self._extraer_conceptos_clave(texto)
            if not conceptos:
                print(f"No se pudieron extraer conceptos clave para {materia}")
                continue
                
            preguntas = []
            max_intentos = 20  # Límite de intentos para evitar bucles infinitos
            
            # Generar preguntas de alternativas (7)
            intentos = 0
            while len([p for p in preguntas if p['tipo'] == 'alternativas']) < 7 and intentos < max_intentos:
                # Seleccionar concepto al azar priorizando los más importantes
                pesos = [c['importancia'] for c in conceptos]
                total = sum(pesos)
                if total == 0:
                    break
                    
                probabilidades = [p/total for p in pesos]
                concepto = np.random.choice(conceptos, p=probabilidades)
                
                pregunta = self._generar_pregunta_analitica(concepto, texto)
                if pregunta and self._es_pregunta_unica(pregunta['pregunta'], preguntas):
                    preguntas.append(pregunta)
                intentos += 1
                
            # Generar preguntas de verdadero/falso (3)
            intentos = 0
            while len([p for p in preguntas if p['tipo'] == 'verdadero_falso']) < 3 and intentos < max_intentos:
                concepto = random.choice(conceptos)
                pregunta = self._generar_pregunta_vf(concepto, texto)
                if pregunta and self._es_pregunta_unica(pregunta['pregunta'], preguntas):
                    preguntas.append(pregunta)
                intentos += 1
            
            # Verificar calidad del cuestionario generado
            if len(preguntas) >= 8:  # Al menos 8 preguntas válidas
                # Realizar verificación final de coherencia
                preguntas_validas = []
                for pregunta in preguntas:
                    # Verificar longitud mínima de pregunta y respuestas
                    if len(pregunta['pregunta'].split()) < 4:
                        continue
                        
                    if pregunta['tipo'] == 'alternativas':
                        # Verificar que todas las alternativas sean suficientemente largas
                        if any(len(opt.split()) < 2 for opt in pregunta['opciones']):
                            continue
                        
                        # Verificar que la respuesta correcta esté en las opciones
                        if pregunta['respuesta_correcta'] not in pregunta['opciones']:
                            continue
                    
                    preguntas_validas.append(pregunta)
                
                # Ordenar preguntas aleatoriamente
                random.shuffle(preguntas_validas)
                cuestionarios[materia].extend(preguntas_validas[:10])  # Tomar solo 10 preguntas
            else:
                print(f"No se generaron suficientes preguntas válidas para {materia}")
        
        return cuestionarios

    def _es_pregunta_unica(self, pregunta: str, preguntas: List[Dict]) -> bool:
        """Verifica si una pregunta es única en el conjunto"""
        if not pregunta:
            return False
            
        pregunta_doc = self.nlp(pregunta)
        
        # Comparar con preguntas existentes
        for p in preguntas:
            # Verificar similitud semántica
            if self.nlp(p['pregunta']).similarity(pregunta_doc) > 0.6:
                return False
                
            # Verificar similitud de palabras clave
            palabras_clave_nueva = set(token.text.lower() 
                                    for token in pregunta_doc 
                                    if token.pos_ in ['NOUN', 'VERB', 'ADJ'])
            palabras_clave_existente = set(token.text.lower() 
                                        for token in self.nlp(p['pregunta']) 
                                        if token.pos_ in ['NOUN', 'VERB', 'ADJ'])
            
            # Si comparten más del 50% de palabras clave, considerar similar
            palabras_comunes = palabras_clave_nueva.intersection(palabras_clave_existente)
            if len(palabras_comunes) / min(len(palabras_clave_nueva), 
                                        len(palabras_clave_existente)) > 0.5:
                return False
        
        return True