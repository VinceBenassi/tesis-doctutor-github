# Generador de Quizzes
# por Franco Benassi
import torch
from transformers import (
    AutoModelForQuestionAnswering, 
    AutoTokenizer,
    T5ForConditionalGeneration,
    pipeline
)
import spacy
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
import json
import random
from tqdm import tqdm

@dataclass
class Pregunta:
    enunciado: str
    tipo: str
    respuesta_correcta: str
    opciones: Optional[List[str]] = None
    explicacion: Optional[str] = None

class GeneradorCuestionarios:
    def __init__(self):
        # Modelo principal para Question-Answering
        self.qa_model = pipeline(
            "question-answering",
            model="mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es",
            tokenizer="mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"
        )
        
        # Modelo T5 para generación de texto
        self.t5_tokenizer = AutoTokenizer.from_pretrained(
            "google/mt5-small"
        )
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            "google/mt5-small"
        )
        
        # Modelo spaCy para análisis lingüístico
        self.nlp = spacy.load("es_core_news_lg")
        
        # Pipeline para generación de preguntas
        self.qg_pipeline = pipeline(
            "text2text-generation",
            model="mrm8488/t5-base-spanish-squad2-question-generation",
            tokenizer="mrm8488/t5-base-spanish-squad2-question-generation"
        )

    def _extraer_informacion_relevante(self, texto: str) -> List[Dict]:
        """
        Extrae información relevante del texto usando análisis semántico
        """
        doc = self.nlp(texto)
        info_relevante = []
        
        # Analizar cada oración
        for sent in doc.sents:
            # Verificar longitud mínima
            if len(sent.text.split()) < 8:
                continue
                
            # Extraer entidades nombradas
            entidades = [
                (ent.text, ent.label_) 
                for ent in sent.ents 
                if ent.label_ in ['ORG', 'PERSON', 'LOC', 'MISC']
            ]
            
            # Extraer conceptos clave
            conceptos = [
                token.text 
                for token in sent 
                if token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop
            ]
            
            if entidades or conceptos:
                info_relevante.append({
                    'texto': sent.text,
                    'entidades': entidades,
                    'conceptos': conceptos,
                    'embedding': sent.vector
                })
        
        return info_relevante

    def _generar_pregunta_qa(self, contexto: str) -> Optional[Dict]:
        """
        Genera una pregunta usando el modelo QA
        """
        try:
            # Generar pregunta usando T5
            input_text = f"generate question: {contexto}"
            inputs = self.t5_tokenizer(
                input_text, 
                max_length=512, 
                truncation=True, 
                return_tensors="pt"
            )
            
            outputs = self.t5_model.generate(
                **inputs,
                max_length=64,
                num_beams=4,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            
            pregunta = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Obtener respuesta usando QA
            respuesta = self.qa_model(
                question=pregunta,
                context=contexto
            )
            
            if respuesta['score'] < 0.5:
                return None
                
            return {
                'pregunta': pregunta,
                'respuesta': respuesta['answer'],
                'contexto': contexto,
                'score': respuesta['score']
            }
            
        except Exception as e:
            print(f"Error generando pregunta QA: {str(e)}")
            return None
        
    def _generar_alternativas(self, pregunta: str, respuesta: str, contexto: str) -> List[str]:
        """
        Genera alternativas coherentes usando análisis semántico y similitud contextual
        """
        try:
            # Vectorizar respuesta correcta
            doc_respuesta = self.nlp(respuesta)
            vec_respuesta = doc_respuesta.vector
            
            # Analizar el contexto completo
            doc_contexto = self.nlp(contexto)
            
            alternativas = set()
            # Encontrar frases similares pero diferentes
            for sent in doc_contexto.sents:
                # Calcular similitud con la respuesta correcta
                similitud = doc_respuesta.similarity(sent)
                
                if 0.3 <= similitud <= 0.7:  # Similitud moderada
                    # Extraer fragmentos relevantes
                    for chunk in sent.noun_chunks:
                        if (len(chunk.text.split()) <= len(respuesta.split()) + 2 and
                            chunk.text.lower() != respuesta.lower()):
                            alternativas.add(chunk.text)
            
            # Si no hay suficientes alternativas, generar usando T5
            while len(alternativas) < 3:
                input_text = f"generate alternative: {pregunta} | correct: {respuesta}"
                inputs = self.t5_tokenizer(
                    input_text,
                    max_length=512,
                    truncation=True,
                    return_tensors="pt"
                )
                
                outputs = self.t5_model.generate(
                    **inputs,
                    max_length=64,
                    num_beams=4,
                    temperature=0.8,
                    top_p=0.9,
                    do_sample=True
                )
                
                alternativa = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
                if self._validar_alternativa(alternativa, respuesta, pregunta):
                    alternativas.add(alternativa)
            
            # Asegurar que tenemos exactamente 3 alternativas
            alternativas = list(alternativas)[:3]
            alternativas.append(respuesta)
            random.shuffle(alternativas)
            
            return alternativas
            
        except Exception as e:
            print(f"Error generando alternativas: {str(e)}")
            return []

    def _validar_alternativa(self, alternativa: str, respuesta: str, pregunta: str) -> bool:
        """
        Valida que una alternativa sea coherente y diferente de la respuesta correcta
        """
        try:
            if not alternativa or alternativa.lower() == respuesta.lower():
                return False
                
            # Verificar longitud similar
            if abs(len(alternativa.split()) - len(respuesta.split())) > 3:
                return False
                
            # Verificar estructura gramatical
            doc_alt = self.nlp(alternativa)
            if not any(token.pos_ in ['NOUN', 'PROPN', 'VERB'] for token in doc_alt):
                return False
                
            # Verificar coherencia semántica
            doc_pregunta = self.nlp(pregunta)
            similitud = doc_alt.similarity(doc_pregunta)
            
            return 0.3 <= similitud <= 0.8
            
        except Exception as e:
            print(f"Error validando alternativa: {str(e)}")
            return False

    def _generar_pregunta_vf(self, info: Dict) -> Optional[Dict]:
        """
        Genera una pregunta de verdadero/falso modificando información del texto
        """
        try:
            texto_original = info['texto']
            doc = self.nlp(texto_original)
            
            # Decidir si será verdadera o falsa
            es_verdadera = random.choice([True, False])
            
            if es_verdadera:
                return {
                    'pregunta': texto_original,
                    'respuesta': 'Verdadero',
                    'explicacion': f"Esta afirmación es correcta según el texto original."
                }
            
            # Modificar el texto para crear una afirmación falsa
            texto_modificado = self._modificar_texto_vf(doc, info)
            
            if texto_modificado:
                return {
                    'pregunta': texto_modificado,
                    'respuesta': 'Falso',
                    'explicacion': f"La afirmación correcta es: {texto_original}"
                }
            
            return None
            
        except Exception as e:
            print(f"Error generando pregunta V/F: {str(e)}")
            return None

    def _modificar_texto_vf(self, doc, info: Dict) -> Optional[str]:
        """
        Modifica el texto para crear una afirmación falsa pero plausible
        """
        try:
            # Identificar elemento clave a modificar
            elementos_modificables = []
            
            # Buscar entidades nombradas
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'PERSON', 'LOC', 'GPE', 'CARDINAL', 'DATE']:
                    elementos_modificables.append({
                        'texto': ent.text,
                        'tipo': ent.label_,
                        'inicio': ent.start_char,
                        'fin': ent.end_char
                    })
            
            # Buscar conceptos clave
            for token in doc:
                if (token.pos_ in ['NOUN', 'PROPN'] and 
                    not token.is_stop and 
                    token.text not in [e['texto'] for e in elementos_modificables]):
                    elementos_modificables.append({
                        'texto': token.text,
                        'tipo': 'CONCEPTO',
                        'inicio': token.idx,
                        'fin': token.idx + len(token.text)
                    })
            
            if not elementos_modificables:
                return None
            
            # Seleccionar elemento a modificar
            elemento = random.choice(elementos_modificables)
            
            # Generar reemplazo usando T5
            input_text = f"generate false replacement: {elemento['texto']} | type: {elemento['tipo']}"
            inputs = self.t5_tokenizer(
                input_text,
                max_length=512,
                truncation=True,
                return_tensors="pt"
            )
            
            outputs = self.t5_model.generate(
                **inputs,
                max_length=64,
                num_beams=4,
                temperature=0.8,
                do_sample=True
            )
            
            reemplazo = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Crear texto modificado
            texto_modificado = (
                doc.text[:elemento['inicio']] + 
                reemplazo + 
                doc.text[elemento['fin']:]
            )
            
            return texto_modificado
            
        except Exception as e:
            print(f"Error modificando texto V/F: {str(e)}")
            return None
    
    def generar_cuestionario(self, texto: str, num_preguntas: int = 10) -> List[Pregunta]:
        """
        Genera un cuestionario completo a partir de un texto
        """
        try:
            preguntas = []
            info_relevante = self._extraer_informacion_relevante(texto)
            
            # Distribuir preguntas (60% alternativas, 40% V/F)
            num_alternativas = int(num_preguntas * 0.6)
            num_vf = num_preguntas - num_alternativas
            
            # Generar preguntas de alternativas
            with tqdm(total=num_alternativas, desc="Generando preguntas de alternativas") as pbar:
                intentos = 0
                while len([p for p in preguntas if p.tipo == 'alternativas']) < num_alternativas and intentos < 20:
                    for info in info_relevante:
                        if len([p for p in preguntas if p.tipo == 'alternativas']) >= num_alternativas:
                            break
                            
                        pregunta_qa = self._generar_pregunta_qa(info['texto'])
                        if pregunta_qa and pregunta_qa['score'] > 0.5:
                            alternativas = self._generar_alternativas(
                                pregunta_qa['pregunta'],
                                pregunta_qa['respuesta'],
                                info['texto']
                            )
                            
                            if len(alternativas) == 4:  # Verificamos que tengamos las 4 alternativas
                                preguntas.append(Pregunta(
                                    enunciado=pregunta_qa['pregunta'],
                                    tipo='alternativas',
                                    respuesta_correcta=pregunta_qa['respuesta'],
                                    opciones=alternativas
                                ))
                                pbar.update(1)
                                
                    intentos += 1
            
            # Generar preguntas de verdadero/falso
            with tqdm(total=num_vf, desc="Generando preguntas de verdadero/falso") as pbar:
                intentos = 0
                while len([p for p in preguntas if p.tipo == 'verdadero_falso']) < num_vf and intentos < 20:
                    for info in info_relevante:
                        if len([p for p in preguntas if p.tipo == 'verdadero_falso']) >= num_vf:
                            break
                            
                        pregunta_vf = self._generar_pregunta_vf(info)
                        if pregunta_vf:
                            preguntas.append(Pregunta(
                                enunciado=pregunta_vf['pregunta'],
                                tipo='verdadero_falso',
                                respuesta_correcta=pregunta_vf['respuesta'],
                                explicacion=pregunta_vf['explicacion']
                            ))
                            pbar.update(1)
                            
                    intentos += 1
            
            # Verificar que tengamos exactamente el número de preguntas solicitado
            if len(preguntas) < num_preguntas:
                print(f"Advertencia: Solo se pudieron generar {len(preguntas)} preguntas de {num_preguntas}")
            
            return preguntas[:num_preguntas]  # Asegurar que devolvemos exactamente num_preguntas
            
        except Exception as e:
            print(f"Error generando cuestionario: {str(e)}")
            return []

    def procesar_archivo_json(self, ruta_archivo: str, ruta_salida: str) -> None:
        """
        Procesa un archivo JSON con textos y genera cuestionarios
        """
        try:
            # Cargar datos
            with open(ruta_archivo, 'r', encoding='utf-8') as f:
                datos = json.load(f)
            
            resultados = {'quiz': []}
            
            # Procesar cada texto
            for item in tqdm(datos['quiz'], desc="Procesando textos"):
                if not item.get('texto'):
                    continue
                
                # Generar cuestionario
                preguntas = self.generar_cuestionario(item['texto'])
                
                if preguntas:
                    resultados['quiz'].append({
                        'texto': item['texto'],
                        'fuente': item.get('fuente', ''),
                        'materia': item.get('materia', ''),
                        'preguntas': [pregunta.__dict__ for pregunta in preguntas]
                    })
            
            # Guardar resultados
            with open(ruta_salida, 'w', encoding='utf-8') as f:
                json.dump(resultados, f, ensure_ascii=False, indent=2)
            
            print(f"\nResultados guardados en: {ruta_salida}")
            
        except Exception as e:
            print(f"Error procesando archivo: {str(e)}")