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
from typing import List, Dict, Tuple

class GeneradorCuestionariosIA:
    def __init__(self):
        # Modelo T5 para generación de preguntas en español
        self.modelo_preguntas = "google/flan-t5-large"
        self.tokenizer_preguntas = AutoTokenizer.from_pretrained(self.modelo_preguntas)
        self.modelo_gen_preguntas = AutoModelForSeq2SeqLM.from_pretrained(self.modelo_preguntas)
        
        # Modelo para question-answering en español (usando PyTorch)
        self.qa_pipeline = pipeline(
            "question-answering",
            model="PlanTL-GOB-ES/roberta-large-bne-sqac",
            tokenizer="PlanTL-GOB-ES/roberta-large-bne-sqac",
            framework="pt"  # Especificamos PyTorch como framework
        )
        
        # Modelo BERT multilingüe para procesamiento de texto
        self.tokenizer_bert = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
        
        # Modelo spaCy para análisis de texto en español
        self.nlp = spacy.load("es_core_news_lg")

    def generar_pregunta_analisis(self, texto: str) -> Tuple[str, List[str], str]:
        """Genera una pregunta de análisis con alternativas"""
        doc = self.nlp(texto)
        
        # Extraer conceptos clave y entidades
        entidades = [ent.text for ent in doc.ents]
        frases_clave = [chunk.text for chunk in doc.noun_chunks]
        
        # Generar la pregunta
        prompt = f"Genera una pregunta de análisis sobre: {texto[:500]}"
        inputs = self.tokenizer_preguntas(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        outputs = self.modelo_gen_preguntas.generate(
            inputs["input_ids"],
            max_length=128,
            num_beams=4,
            temperature=0.7
        )
        
        pregunta = self.tokenizer_preguntas.decode(outputs[0], skip_special_tokens=True)
        
        # Obtener respuesta usando QA
        try:
            qa_result = self.qa_pipeline(
                question=pregunta,
                context=texto,
                max_answer_length=50
            )
            respuesta_correcta = qa_result['answer']
            
            # Generar alternativas
            alternativas = self._generar_alternativas(
                pregunta, 
                respuesta_correcta, 
                texto, 
                entidades, 
                frases_clave
            )
            
            return pregunta, alternativas, respuesta_correcta
            
        except Exception as e:
            print(f"Error en generación de pregunta: {e}")
            return None, None, None

    def _generar_alternativas(self, pregunta: str, respuesta_correcta: str, 
                            texto: str, entidades: List[str], frases_clave: List[str]) -> List[str]:
        """Genera alternativas coherentes para una pregunta"""
        alternativas = [respuesta_correcta]
        
        # Analizar el tipo de respuesta esperada
        resp_doc = self.nlp(respuesta_correcta)
        tipo_respuesta = self._determinar_tipo_respuesta(resp_doc)
        
        # Seleccionar candidatos según el tipo de respuesta
        candidatos = self._seleccionar_candidatos(
            tipo_respuesta, 
            texto, 
            entidades, 
            frases_clave, 
            respuesta_correcta
        )
        
        # Filtrar y procesar candidatos
        for candidato in candidatos:
            if len(alternativas) >= 4:
                break
                
            if self._es_alternativa_valida(candidato, respuesta_correcta, pregunta):
                alternativas.append(candidato)
        
        # Si no hay suficientes alternativas, generar algunas
        while len(alternativas) < 4:
            nueva_alt = self._generar_alternativa_similar(
                respuesta_correcta, 
                tipo_respuesta
            )
            if nueva_alt and nueva_alt not in alternativas:
                alternativas.append(nueva_alt)
        
        random.shuffle(alternativas)
        return alternativas[:4]

    def _determinar_tipo_respuesta(self, doc) -> str:
        """Determina el tipo de respuesta esperada"""
        if any(ent.label_ in ['PER', 'ORG'] for ent in doc.ents):
            return 'ENTIDAD'
        elif any(token.pos_ == 'NUM' for token in doc):
            return 'NUMERICO'
        elif len(doc) <= 3:
            return 'CONCEPTO'
        else:
            return 'EXPLICACION'

    def _seleccionar_candidatos(self, tipo: str, texto: str, 
                              entidades: List[str], frases: List[str], 
                              respuesta: str) -> List[str]:
        """Selecciona candidatos para alternativas según el tipo"""
        doc = self.nlp(texto)
        candidatos = []
        
        if tipo == 'ENTIDAD':
            candidatos = [ent.text for ent in doc.ents 
                         if ent.text != respuesta and len(ent.text.split()) <= 3]
        elif tipo == 'NUMERICO':
            # Extraer números y modificarlos ligeramente
            nums = [token.text for token in doc if token.like_num]
            candidatos = [self._modificar_numero(num) for num in nums]
        elif tipo == 'CONCEPTO':
            candidatos = [chunk.text for chunk in doc.noun_chunks 
                         if len(chunk.text.split()) <= 3]
        else:
            candidatos = [sent.text for sent in doc.sents 
                         if len(sent.text.split()) <= 10]
        
        return list(set(candidatos))

    def _modificar_numero(self, num_str: str) -> str:
        """Modifica un número para crear una alternativa plausible"""
        try:
            num = float(num_str)
            factor = random.uniform(0.8, 1.2)
            return str(round(num * factor, 2))
        except:
            return num_str

    def _es_alternativa_valida(self, alternativa: str, respuesta: str, pregunta: str) -> bool:
        """Verifica si una alternativa es válida y coherente"""
        if not alternativa or alternativa.strip() == respuesta.strip():
            return False
        
        alt_doc = self.nlp(alternativa)
        resp_doc = self.nlp(respuesta)
        
        # Verificar similitud semántica
        similitud = alt_doc.similarity(resp_doc)
        if similitud > 0.9 or similitud < 0.1:
            return False
        
        # Verificar longitud y complejidad
        if len(alternativa.split()) > 2 * len(respuesta.split()):
            return False
        
        return True

    def _generar_alternativa_similar(self, respuesta: str, tipo: str) -> str:
        """Genera una alternativa similar pero incorrecta"""
        doc = self.nlp(respuesta)
        
        if tipo == 'NUMERICO':
            nums = [token.text for token in doc if token.like_num]
            if nums:
                return self._modificar_numero(nums[0])
        
        palabras = respuesta.split()
        if len(palabras) <= 3:
            return ' '.join(random.sample(palabras, len(palabras)))
        
        return None

    def generar_pregunta_vf(self, texto: str) -> Tuple[str, str]:
        """Genera una pregunta de verdadero/falso"""
        doc = self.nlp(texto)
        
        # Seleccionar una oración informativa
        oraciones = [sent.text for sent in doc.sents 
                    if len(sent.text.split()) > 5 and len(sent.text.split()) < 20]
        
        if not oraciones:
            return None, None
        
        oracion = random.choice(oraciones)
        es_verdadero = random.choice([True, False])
        
        if es_verdadero:
            pregunta = f"¿Es correcto afirmar que {oracion}?"
            return pregunta, "Verdadero"
        else:
            oracion_modificada = self._modificar_oracion(oracion)
            pregunta = f"¿Es correcto afirmar que {oracion_modificada}?"
            return pregunta, "Falso"

    def _modificar_oracion(self, oracion: str) -> str:
        """Modifica una oración para hacerla falsa pero plausible"""
        doc = self.nlp(oracion)
        
        # Estrategias de modificación
        modificaciones = [
            self._cambiar_sujeto,
            self._cambiar_verbo,
            self._agregar_negacion,
            self._cambiar_objeto
        ]
        
        modificacion = random.choice(modificaciones)
        oracion_modificada = modificacion(doc)
        
        return oracion_modificada if oracion_modificada else oracion

    def _cambiar_sujeto(self, doc) -> str:
        """Cambia el sujeto de la oración"""
        for token in doc:
            if token.dep_ == 'nsubj':
                return doc.text.replace(token.text, random.choice(['alguien', 'nadie', 'otro']))
        return None

    def _cambiar_verbo(self, doc) -> str:
        """Cambia el verbo principal por su opuesto o similar"""
        verbos_opuestos = {
            'es': 'no es',
            'puede': 'no puede',
            'tiene': 'carece de',
            'está': 'no está'
        }
        
        for token in doc:
            if token.pos_ == 'VERB':
                return doc.text.replace(token.text, verbos_opuestos.get(token.text, f"no {token.text}"))
        return None

    def _agregar_negacion(self, doc) -> str:
        """Agrega o quita una negación"""
        texto = doc.text
        if 'no' in texto.lower():
            return texto.replace('no ', '').replace('No ', '')
        return f"no {texto}"

    def _cambiar_objeto(self, doc) -> str:
        """Cambia el objeto directo de la oración"""
        for token in doc:
            if token.dep_ == 'obj':
                return doc.text.replace(token.text, 'otra cosa')
        return None

def generar_cuestionario(ruta_archivo: str) -> Dict:
    """Genera un cuestionario completo a partir del archivo JSON"""
    try:
        with open(ruta_archivo, 'r', encoding='utf-8') as archivo:
            datos = json.load(archivo)
    except Exception as e:
        print(f"Error al cargar el archivo JSON: {e}")
        return None

    generador = GeneradorCuestionariosIA()
    cuestionarios = {}

    for item in datos['quiz']:
        materia = item['materia']
        if materia not in cuestionarios:
            cuestionarios[materia] = []

        texto = item['texto']
        preguntas = []
        
        # Generar preguntas de alternativas
        for _ in range(7):  # Intentar generar 7 preguntas de alternativas
            pregunta, alternativas, respuesta = generador.generar_pregunta_analisis(texto)
            if pregunta and alternativas and respuesta:
                preguntas.append({
                    "pregunta": pregunta,
                    "opciones": alternativas,
                    "respuesta_correcta": respuesta,
                    "tipo": "alternativas"
                })

        # Generar preguntas de verdadero/falso
        for _ in range(3):  # Intentar generar 3 preguntas de V/F
            pregunta, respuesta = generador.generar_pregunta_vf(texto)
            if pregunta and respuesta:
                preguntas.append({
                    "pregunta": pregunta,
                    "opciones": ["Verdadero", "Falso"],
                    "respuesta_correcta": respuesta,
                    "tipo": "verdadero_falso"
                })

        if preguntas:
            cuestionarios[materia].append({
                "texto": texto,
                "fuente": item['fuente'],
                "preguntas": preguntas
            })

    return cuestionarios