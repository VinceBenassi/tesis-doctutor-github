# Generador de Quizzes
# por Franco Benassi
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    pipeline
)
import spacy
import json
import random
from typing import List, Dict, Tuple

class GeneradorCuestionarios:
    def __init__(self):
        print("Inicializando modelos...")
        
        # Modelo T5 multilingüe para mejor generación en español
        self.model_name = "google/mt5-large"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        
        # Pipeline de QA específico para español
        self.qa_pipeline = pipeline(
            "question-answering",
            model="mrm8488/bert-spanish-cased-finetuned-squadv1-es",
            tokenizer="mrm8488/bert-spanish-cased-finetuned-squadv1-es"
        )
        
        # Modelo SpaCy para análisis semántico en español
        self.nlp = spacy.load("es_core_news_lg")
        
        # Plantillas para diferentes tipos de preguntas analíticas
        self.plantillas_preguntas = {
            "análisis": [
                "¿Cuál es la principal implicación de {concepto} en el contexto de {tema}?",
                "¿De qué manera {concepto} influye en {tema}?",
                "¿Qué papel desempeña {concepto} en relación con {tema}?",
            ],
            "evaluación": [
                "¿Qué evidencia respalda la importancia de {concepto} en {tema}?",
                "¿Cómo se relaciona {concepto} con los resultados de {tema}?",
                "¿Qué aspectos de {concepto} son más relevantes para {tema}?",
            ],
            "síntesis": [
                "¿Qué conclusiones se pueden extraer sobre {concepto} en el contexto de {tema}?",
                "¿Cómo se integra {concepto} con otros aspectos de {tema}?",
                "¿Qué relación existe entre {concepto} y el desarrollo de {tema}?",
            ]
        }

    def _extraer_conceptos_clave(self, texto: str) -> List[Dict]:
        """Extrae conceptos clave y sus relaciones del texto usando análisis semántico"""
        doc = self.nlp(texto)
        
        # Identificar conceptos importantes usando análisis de dependencias
        conceptos = []
        for sent in doc.sents:
            # Extraer frases nominales importantes
            for chunk in sent.noun_chunks:
                if len(chunk.text.split()) > 1:  # Evitar conceptos simples
                    # Analizar el contexto semántico
                    contexto = [token.text for token in sent 
                              if token.dep_ in ['ROOT', 'nsubj', 'dobj']]
                    
                    # Identificar tema general de la oración
                    tema = next((token.text for token in sent 
                               if token.dep_ == 'ROOT' and token.pos_ == 'VERB'), '')
                    
                    conceptos.append({
                        'concepto': chunk.text,
                        'tema': tema,
                        'contexto': ' '.join(contexto),
                        'oración': sent.text,
                        'importancia': len([token for token in chunk 
                                          if token.pos_ in ['NOUN', 'PROPN']])
                    })
        
        # Ordenar por importancia y eliminar duplicados
        conceptos = sorted(conceptos, key=lambda x: x['importancia'], reverse=True)
        return self._eliminar_conceptos_duplicados(conceptos)

    def _eliminar_conceptos_duplicados(self, conceptos: List[Dict]) -> List[Dict]:
        """Elimina conceptos duplicados o muy similares"""
        únicos = []
        textos_vistos = set()
        
        for concepto in conceptos:
            texto_normalizado = concepto['concepto'].lower()
            if texto_normalizado not in textos_vistos:
                únicos.append(concepto)
                textos_vistos.add(texto_normalizado)
        
        return únicos

    def _generar_pregunta(self, concepto: Dict, texto_completo: str) -> Dict:
        """Genera una pregunta analítica basada en el concepto"""
        # Seleccionar tipo de pregunta y plantilla
        tipo = random.choice(list(self.plantillas_preguntas.keys()))
        plantilla = random.choice(self.plantillas_preguntas[tipo])
        
        # Generar pregunta inicial
        pregunta_base = plantilla.format(
            concepto=concepto['concepto'],
            tema=concepto['tema']
        )
        
        # Usar QA para obtener la respuesta correcta
        try:
            resultado_qa = self.qa_pipeline(
                question=pregunta_base,
                context=texto_completo
            )
            respuesta_correcta = resultado_qa['answer']
            
            # Verificar que la respuesta tenga sentido
            if len(respuesta_correcta.split()) < 2:
                raise ValueError("Respuesta demasiado corta")
                
            # Generar alternativas coherentes
            alternativas = self._generar_alternativas(
                pregunta_base,
                respuesta_correcta,
                concepto,
                texto_completo
            )
            
            return {
                "pregunta": pregunta_base,
                "opciones": alternativas,
                "respuesta_correcta": respuesta_correcta,
                "tipo": "alternativas"
            }
            
        except Exception as e:
            print(f"Error al generar pregunta: {str(e)}")
            return None

    def _generar_alternativas(self, pregunta: str, respuesta_correcta: str, 
                            concepto: Dict, texto_completo: str) -> List[str]:
        """Genera alternativas coherentes usando el contexto semántico"""
        prompt = f"""
        Texto: {texto_completo}
        Pregunta: {pregunta}
        Respuesta correcta: {respuesta_correcta}
        
        Genera tres alternativas incorrectas pero plausibles que:
        1. Sean relacionadas al tema y concepto
        2. Tengan longitud y estilo similar a la respuesta correcta
        3. Sean lógicamente posibles pero incorrectas
        4. No sean obviamente falsas
        """
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", 
                                  max_length=512, truncation=True)
            
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=128,
                num_return_sequences=3,
                temperature=0.8,
                num_beams=4,
                no_repeat_ngram_size=2
            )
            
            alternativas = [respuesta_correcta]
            for output in outputs:
                alternativa = self.tokenizer.decode(output, skip_special_tokens=True)
                if self._es_alternativa_válida(alternativa, alternativas):
                    alternativas.append(alternativa)
            
            # Asegurar 4 alternativas únicas
            while len(alternativas) < 4:
                nueva_alt = self._generar_alternativa_individual(
                    pregunta, respuesta_correcta, concepto
                )
                if self._es_alternativa_válida(nueva_alt, alternativas):
                    alternativas.append(nueva_alt)
            
            random.shuffle(alternativas)
            return alternativas[:4]
            
        except Exception as e:
            print(f"Error al generar alternativas: {str(e)}")
            return []

    def _es_alternativa_válida(self, alternativa: str, 
                             alternativas_existentes: List[str]) -> bool:
        """Verifica si una alternativa es válida y única"""
        if not alternativa or len(alternativa.split()) < 2:
            return False
            
        # Verificar similitud con alternativas existentes
        for existente in alternativas_existentes:
            if self.nlp(alternativa).similarity(self.nlp(existente)) > 0.8:
                return False
        
        return True

    def _generar_pregunta_vf(self, concepto: Dict, texto_completo: str) -> Dict:
        """Genera una pregunta de verdadero/falso analítica"""
        prompt = f"""
        Basado en este texto: {concepto['oración']}
        
        Genera una afirmación analítica sobre {concepto['concepto']} que:
        1. Sea específica y verificable
        2. Requiera comprensión del tema
        3. No sea trivialmente verdadera o falsa
        4. Use lenguaje preciso y claro
        """
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", 
                                  max_length=512, truncation=True)
            
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=128,
                temperature=0.7,
                num_beams=4
            )
            
            afirmación = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Verificar la validez usando QA
            resultado = self.qa_pipeline(
                question=f"¿Es verdadera esta afirmación? {afirmación}",
                context=texto_completo
            )
            
            es_verdadero = resultado['score'] > 0.7
            
            return {
                "pregunta": f"¿Es correcto afirmar que {afirmación}?",
                "opciones": ["Verdadero", "Falso"],
                "respuesta_correcta": "Verdadero" if es_verdadero else "Falso",
                "tipo": "verdadero_falso"
            }
            
        except Exception as e:
            print(f"Error al generar pregunta V/F: {str(e)}")
            return None

    def generar_cuestionario(self, ruta_archivo: str) -> Dict:
        """Genera el cuestionario completo procesando el archivo JSON"""
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
            
            # Extraer conceptos clave
            conceptos = self._extraer_conceptos_clave(texto)
            preguntas = []
            
            # Generar preguntas de alternativas (7)
            intentos_alternativas = 0
            while len([p for p in preguntas if p['tipo'] == 'alternativas']) < 7 and intentos_alternativas < 14:
                concepto = random.choice(conceptos)
                pregunta = self._generar_pregunta(concepto, texto)
                if pregunta and self._es_pregunta_única(pregunta, preguntas):
                    preguntas.append(pregunta)
                intentos_alternativas += 1
            
            # Generar preguntas de verdadero/falso (3)
            intentos_vf = 0
            while len([p for p in preguntas if p['tipo'] == 'verdadero_falso']) < 3 and intentos_vf < 6:
                concepto = random.choice(conceptos)
                pregunta = self._generar_pregunta_vf(concepto, texto)
                if pregunta and self._es_pregunta_única(pregunta, preguntas):
                    preguntas.append(pregunta)
                intentos_vf += 1
            
            if preguntas:
                cuestionarios[materia].extend(preguntas)
        
        return cuestionarios

    def _es_pregunta_única(self, pregunta: Dict, preguntas: List[Dict]) -> bool:
        """Verifica si una pregunta es única en el conjunto"""
        for p in preguntas:
            if self.nlp(pregunta['pregunta']).similarity(self.nlp(p['pregunta'])) > 0.7:
                return False
        return True