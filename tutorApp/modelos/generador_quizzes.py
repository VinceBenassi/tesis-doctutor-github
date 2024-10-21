# Generador de Quizzes
# por Franco Benassi
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    AutoModelForQuestionAnswering,
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
        
        # Modelo FLAN-T5 para generación de preguntas y respuestas
        self.gen_model_name = "google/flan-t5-large"
        self.gen_tokenizer = AutoTokenizer.from_pretrained(self.gen_model_name)
        self.gen_model = AutoModelForSeq2SeqLM.from_pretrained(self.gen_model_name)
        
        # Modelo multilingüe para verificación de respuestas
        self.qa_model_name = "deepset/xlm-roberta-large-squad2"
        self.qa_pipeline = pipeline(
            "question-answering",
            model=self.qa_model_name,
            tokenizer=self.qa_model_name
        )
        
        # Modelo SpaCy para análisis semántico
        self.nlp = spacy.load("es_core_news_lg")
        
    def _extraer_temas_principales(self, texto: str) -> List[Dict[str, Any]]:
        """Extrae los temas principales y conceptos relevantes del texto"""
        doc = self.nlp(texto)
        
        # Identificar oraciones importantes usando criterios lingüísticos
        temas = []
        for sent in doc.sents:
            # Analizar la estructura de la oración
            sujetos = [chunk for chunk in sent.noun_chunks 
                      if chunk.root.dep_ in ['nsubj', 'nsubjpass']]
            verbos = [token for token in sent if token.pos_ == 'VERB']
            objetos = [chunk for chunk in sent.noun_chunks 
                      if chunk.root.dep_ in ['dobj', 'pobj']]
            
            if sujetos and verbos:
                tema = {
                    'oración': sent.text,
                    'sujetos': [s.text for s in sujetos],
                    'verbos': [v.text for v in verbos],
                    'objetos': [o.text for o in objetos],
                    'importancia': len(sujetos) + len(verbos) + len(objetos)
                }
                temas.append(tema)
        
        return sorted(temas, key=lambda x: x['importancia'], reverse=True)

    def _generar_pregunta_analítica(self, tema: Dict[str, Any], texto_completo: str) -> Dict:
        """Genera una pregunta analítica basada en el tema identificado"""
        # Construir prompt para generar pregunta
        elementos = tema['sujetos'] + tema['objetos']
        if not elementos:
            return None
            
        concepto = random.choice(elementos)
        
        prompt = f"""
        Genera una pregunta analítica en español sobre este tema: {tema['oración']}
        
        La pregunta debe:
        1. Requerir análisis y comprensión
        2. Estar relacionada con: {concepto}
        3. Tener una respuesta específica
        4. Ser clara y bien formulada
        5. No ser una pregunta de completar espacios
        6. No ser una pregunta trivial
        
        Formato: Solo la pregunta
        """
        
        try:
            # Generar pregunta
            inputs = self.gen_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            outputs = self.gen_model.generate(
                inputs.input_ids,
                max_length=128,
                temperature=0.7,
                num_beams=4,
                no_repeat_ngram_size=2
            )
            pregunta = self.gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Obtener respuesta correcta usando QA
            resultado_qa = self.qa_pipeline(
                question=pregunta,
                context=texto_completo
            )
            
            if resultado_qa['score'] < 0.5 or len(resultado_qa['answer'].split()) < 2:
                return None
                
            respuesta_correcta = resultado_qa['answer']
            
            # Generar alternativas plausibles
            alternativas = self._generar_alternativas_coherentes(
                pregunta, 
                respuesta_correcta,
                tema,
                texto_completo
            )
            
            if len(alternativas) < 4:
                return None
                
            return {
                "pregunta": pregunta,
                "opciones": alternativas,
                "respuesta_correcta": respuesta_correcta,
                "tipo": "alternativas"
            }
            
        except Exception as e:
            print(f"Error al generar pregunta analítica: {str(e)}")
            return None

    def _generar_alternativas_coherentes(
        self, 
        pregunta: str, 
        respuesta_correcta: str, 
        tema: Dict[str, Any],
        texto_completo: str
    ) -> List[str]:
        """Genera alternativas coherentes y plausibles"""
        prompt = f"""
        Pregunta: {pregunta}
        Respuesta correcta: {respuesta_correcta}
        Contexto: {tema['oración']}
        
        Genera tres alternativas incorrectas que:
        1. Sean del mismo tipo que la respuesta correcta
        2. Estén relacionadas con el tema
        3. Sean plausibles pero claramente incorrectas
        4. Tengan longitud y estilo similar
        5. Sean lógicamente posibles
        
        Formato: Una alternativa por línea
        """
        
        try:
            inputs = self.gen_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            outputs = self.gen_model.generate(
                inputs.input_ids,
                max_length=256,
                num_return_sequences=3,
                temperature=0.8,
                num_beams=4,
                no_repeat_ngram_size=2
            )
            
            alternativas = [respuesta_correcta]
            for output in outputs:
                alt = self.gen_tokenizer.decode(output, skip_special_tokens=True).strip()
                # Verificar que la alternativa sea única y coherente
                if (alt and 
                    len(alt.split()) >= 2 and
                    self._es_alternativa_válida(alt, alternativas, pregunta)):
                    alternativas.append(alt)
            
            # Asegurar que tenemos 4 alternativas únicas
            while len(alternativas) < 4:
                nueva_alt = self._generar_alternativa_individual(
                    pregunta, respuesta_correcta, tema
                )
                if nueva_alt and self._es_alternativa_válida(nueva_alt, alternativas, pregunta):
                    alternativas.append(nueva_alt)
            
            random.shuffle(alternativas)
            return alternativas[:4]
            
        except Exception as e:
            print(f"Error al generar alternativas: {str(e)}")
            return []

    def _es_alternativa_válida(
        self, 
        alternativa: str, 
        alternativas_existentes: List[str],
        pregunta: str
    ) -> bool:
        """Verifica si una alternativa es válida y coherente"""
        if not alternativa or len(alternativa.split()) < 2:
            return False
            
        # Verificar similitud con pregunta y otras alternativas
        alt_doc = self.nlp(alternativa)
        
        # No debe ser muy similar a la pregunta
        if self.nlp(pregunta).similarity(alt_doc) > 0.8:
            return False
            
        # No debe ser muy similar a otras alternativas
        for existente in alternativas_existentes:
            if self.nlp(existente).similarity(alt_doc) > 0.7:
                return False
        
        return True

    def _generar_alternativa_individual(
        self, 
        pregunta: str, 
        respuesta_correcta: str, 
        tema: Dict[str, Any]
    ) -> str:
        """Genera una alternativa individual plausible"""
        prompt = f"""
        Pregunta: {pregunta}
        Respuesta correcta: {respuesta_correcta}
        
        Genera una respuesta incorrecta pero plausible que:
        1. Sea del mismo tipo que la respuesta correcta
        2. Esté relacionada con: {' '.join(tema['sujetos'])}
        3. Sea específica y clara
        4. Tenga longitud similar a la respuesta correcta
        
        Formato: Solo la respuesta
        """
        
        try:
            inputs = self.gen_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            outputs = self.gen_model.generate(
                inputs.input_ids,
                max_length=128,
                temperature=0.8,
                num_beams=4
            )
            
            return self.gen_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            
        except Exception as e:
            print(f"Error al generar alternativa individual: {str(e)}")
            return None

    def _generar_pregunta_vf(self, tema: Dict[str, Any], texto_completo: str) -> Dict:
        """Genera una pregunta de verdadero/falso analítica"""
        prompt = f"""
        Genera una afirmación analítica en español sobre: {tema['oración']}
        
        La afirmación debe:
        1. Ser específica y verificable
        2. Requerir comprensión del tema
        3. No ser obvia
        4. Estar relacionada con: {' '.join(tema['sujetos'] + tema['objetos'])}
        5. Ser autocontenida (no referirse a elementos externos)
        
        Formato: Solo la afirmación
        """
        
        try:
            inputs = self.gen_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            outputs = self.gen_model.generate(
                inputs.input_ids,
                max_length=128,
                temperature=0.7,
                num_beams=4
            )
            
            afirmación = self.gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Verificar si la afirmación es verdadera
            resultado = self.qa_pipeline(
                question=f"¿Es verdadera esta afirmación? {afirmación}",
                context=texto_completo
            )
            
            if resultado['score'] < 0.6:
                return None
                
            es_verdadero = resultado['score'] > 0.8
            
            return {
                "pregunta": afirmación,
                "opciones": ["Verdadero", "Falso"],
                "respuesta_correcta": "Verdadero" if es_verdadero else "Falso",
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
            
            # Extraer temas principales
            temas = self._extraer_temas_principales(texto)
            preguntas = []
            
            # Generar preguntas de alternativas (7)
            intentos_alternativas = 0
            while len([p for p in preguntas if p['tipo'] == 'alternativas']) < 7 and intentos_alternativas < 14:
                tema = random.choice(temas)
                pregunta = self._generar_pregunta_analítica(tema, texto)
                if pregunta and self._es_pregunta_única(pregunta, preguntas):
                    preguntas.append(pregunta)
                intentos_alternativas += 1
            
            # Generar preguntas de verdadero/falso (3)
            intentos_vf = 0
            while len([p for p in preguntas if p['tipo'] == 'verdadero_falso']) < 3 and intentos_vf < 6:
                tema = random.choice(temas)
                pregunta = self._generar_pregunta_vf(tema, texto)
                if pregunta and self._es_pregunta_única(pregunta, preguntas):
                    preguntas.append(pregunta)
                intentos_vf += 1
            
            if preguntas:
                cuestionarios[materia].extend(preguntas)
        
        return cuestionarios

    def _es_pregunta_única(self, pregunta: Dict, preguntas: List[Dict]) -> bool:
        """Verifica si una pregunta es única en el conjunto"""
        if not pregunta:
            return False
            
        pregunta_doc = self.nlp(pregunta['pregunta'])
        for p in preguntas:
            if self.nlp(p['pregunta']).similarity(pregunta_doc) > 0.7:
                return False
        return True