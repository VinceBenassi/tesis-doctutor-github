# Generador de Quizzes
# por Franco Benassi
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    pipeline
)
import spacy
import json
import random
from typing import List, Dict, Tuple
import numpy as np

class GeneradorPreguntas:
    def __init__(self):
        # Usar modelos que están garantizados disponibles
        model_name = "xlm-roberta-base"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Pipeline de question-answering multilingüe
        self.qa_pipeline = pipeline(
            "question-answering",
            model="xlm-roberta-base",
            tokenizer="xlm-roberta-base"
        )
        
        # Modelo para clasificación
        self.classifier = pipeline(
            "text-classification",
            model="xlm-roberta-base",
            tokenizer="xlm-roberta-base"
        )
        
        # Spacy para procesamiento de texto en español
        self.nlp = spacy.load("es_core_news_sm")
        
        # Plantillas para preguntas de análisis
        self.plantillas_preguntas = [
            "¿Cuál es el principal concepto relacionado con {}?",
            "¿Qué característica define mejor a {}?",
            "¿Cómo se relaciona {} con el tema central?",
            "¿Cuál es la importancia de {} en este contexto?",
            "¿Qué papel desempeña {} en esta situación?",
            "¿Qué efecto tiene {} en el desarrollo del tema?",
            "¿Qué conclusión podemos extraer sobre {}?",
            "¿Cuál es la función principal de {}?",
            "¿Qué implica {} en este contexto?",
            "¿Cómo influye {} en el desarrollo del tema?"
        ]
    
    def extraer_temas_principales(self, texto: str) -> List[str]:
        """Extrae los temas principales del texto usando análisis NLP"""
        doc = self.nlp(texto)
        temas = []
        
        # Extraer frases nominales importantes
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 4:
                temas.append(chunk.text)
        
        # Extraer entidades relevantes
        for ent in doc.ents:
            if ent.label_ in ['PER', 'ORG', 'LOC', 'MISC']:
                temas.append(ent.text)
        
        return list(set(temas))
    
    def generar_pregunta_analisis(self, tema: str, contexto: str) -> Tuple[str, str]:
        """Genera una pregunta de análisis sobre un tema específico"""
        # Seleccionar plantilla aleatoria
        plantilla = random.choice(self.plantillas_preguntas)
        pregunta = plantilla.format(tema)
        
        # Obtener respuesta usando el contexto
        try:
            respuesta = self.qa_pipeline(
                question=pregunta,
                context=contexto,
                max_length=512,
                truncation=True
            )
            return pregunta, respuesta['answer']
        except Exception as e:
            print(f"Error generando respuesta: {e}")
            return None, None
    
    def generar_alternativas_coherentes(self, respuesta_correcta: str, contexto: str) -> Tuple[List[str], int]:
        """Genera alternativas coherentes pero incorrectas"""
        doc = self.nlp(contexto)
        alternativas = [respuesta_correcta]
        
        # Generar alternativas basadas en el contexto
        frases_candidatas = []
        for sent in doc.sents:
            if respuesta_correcta not in sent.text:
                frases_candidatas.append(sent.text)
        
        # Filtrar y refinar alternativas
        for frase in frases_candidatas:
            if len(alternativas) >= 4:
                break
                
            alternativa = self.refinar_alternativa(frase)
            if alternativa and self.es_alternativa_valida(alternativa, alternativas):
                alternativas.append(alternativa)
        
        # Si no hay suficientes alternativas, generar algunas
        while len(alternativas) < 4:
            alternativa = self.generar_alternativa_generica(respuesta_correcta)
            if alternativa and self.es_alternativa_valida(alternativa, alternativas):
                alternativas.append(alternativa)
        
        # Mezclar alternativas y guardar índice de la correcta
        indice_correcto = alternativas.index(respuesta_correcta)
        random.shuffle(alternativas)
        return alternativas, alternativas.index(respuesta_correcta)
    
    def es_alternativa_valida(self, candidato: str, existentes: List[str]) -> bool:
        """Verifica si una alternativa es válida y coherente"""
        if not candidato or len(candidato) < 3 or len(candidato.split()) > 15:
            return False
            
        # Verificar que no sea muy similar a las existentes
        doc_candidato = self.nlp(candidato)
        for existente in existentes:
            similitud = doc_candidato.similarity(self.nlp(existente))
            if similitud > 0.8:
                return False
        
        # Verificar estructura gramatical básica
        return any(token.pos_ in ['NOUN', 'VERB', 'ADJ'] for token in doc_candidato)
    
    def refinar_alternativa(self, texto: str) -> str:
        """Refina una alternativa para hacerla más concisa y coherente"""
        doc = self.nlp(texto)
        
        # Extraer la parte más relevante si es muy larga
        if len(doc) > 10:
            # Buscar la frase nominal más relevante
            chunks = list(doc.noun_chunks)
            if chunks:
                return max(chunks, key=lambda x: len(x.text)).text
        
        return texto if len(texto.split()) <= 10 else ' '.join(texto.split()[:10])
    
    def generar_alternativa_generica(self, texto_base: str) -> str:
        """Genera una alternativa genérica pero plausible"""
        doc = self.nlp(texto_base)
        
        # Estrategias para generar alternativas
        estrategias = [
            self._modificar_sustantivos,
            self._cambiar_adjetivos,
            self._alterar_verbos
        ]
        
        return random.choice(estrategias)(doc)
    
    def _modificar_sustantivos(self, doc) -> str:
        """Modifica sustantivos manteniendo coherencia"""
        sustantivos_comunes = {
            'problema': ['dificultad', 'situación', 'circunstancia'],
            'persona': ['individuo', 'sujeto', 'ciudadano'],
            'tiempo': ['momento', 'período', 'etapa'],
            'forma': ['manera', 'modo', 'método'],
            'caso': ['ejemplo', 'situación', 'instancia']
        }
        
        texto = doc.text
        for token in doc:
            if token.pos_ == 'NOUN' and token.text.lower() in sustantivos_comunes:
                reemplazo = random.choice(sustantivos_comunes[token.text.lower()])
                texto = texto.replace(token.text, reemplazo)
                break
        
        return texto
    
    def _cambiar_adjetivos(self, doc) -> str:
        """Cambia adjetivos por sinónimos o antónimos"""
        adjetivos_mapping = {
            'bueno': ['regular', 'aceptable', 'decente'],
            'malo': ['deficiente', 'inadecuado', 'imperfecto'],
            'grande': ['considerable', 'significativo', 'notable'],
            'pequeño': ['reducido', 'limitado', 'escaso']
        }
        
        texto = doc.text
        for token in doc:
            if token.pos_ == 'ADJ' and token.text.lower() in adjetivos_mapping:
                reemplazo = random.choice(adjetivos_mapping[token.text.lower()])
                texto = texto.replace(token.text, reemplazo)
                break
        
        return texto
    
    def _alterar_verbos(self, doc) -> str:
        """Altera verbos manteniendo coherencia"""
        verbos_mapping = {
            'ser': ['parecer', 'resultar', 'manifestarse'],
            'estar': ['encontrarse', 'hallarse', 'permanecer'],
            'hacer': ['realizar', 'ejecutar', 'efectuar'],
            'tener': ['poseer', 'contar con', 'disponer']
        }
        
        texto = doc.text
        for token in doc:
            if token.pos_ == 'VERB' and token.text.lower() in verbos_mapping:
                reemplazo = random.choice(verbos_mapping[token.text.lower()])
                texto = texto.replace(token.text, reemplazo)
                break
        
        return texto

    def generar_pregunta_verdadero_falso(self, texto: str) -> Tuple[str, str]:
        """Genera una pregunta de verdadero/falso"""
        doc = self.nlp(texto)
        oraciones = list(doc.sents)
        
        if not oraciones:
            return None, None
        
        # Seleccionar una oración informativa
        oracion = max(oraciones, key=lambda x: len([t for t in x if t.pos_ in ['NOUN', 'VERB']]))
        es_verdadero = random.choice([True, False])
        
        if es_verdadero:
            return f"¿Es correcto afirmar que {oracion.text.strip()}?", "Verdadero"
        else:
            oracion_modificada = self.modificar_oracion_para_falso(oracion.text)
            return f"¿Es correcto afirmar que {oracion_modificada}?", "Falso"
    
    def modificar_oracion_para_falso(self, oracion: str) -> str:
        """Modifica una oración para hacerla falsa pero plausible"""
        doc = self.nlp(oracion)
        
        estrategias = [
            self._negar_afirmacion,
            self._cambiar_entidad,
            self._modificar_cantidad,
            self._alterar_tiempo_verbal
        ]
        
        return random.choice(estrategias)(doc)
    
    def _negar_afirmacion(self, doc) -> str:
        """Niega la afirmación principal"""
        texto = doc.text
        for token in doc:
            if token.pos_ == "VERB" and token.dep_ == "ROOT":
                return texto.replace(token.text, f"no {token.text}")
        return texto
    
    def _cambiar_entidad(self, doc) -> str:
        """Cambia una entidad por otra del mismo tipo"""
        entidades_por_tipo = {
            'PER': ['Juan', 'María', 'Pedro', 'Ana'],
            'ORG': ['empresa', 'organización', 'institución', 'compañía'],
            'LOC': ['Madrid', 'Barcelona', 'Sevilla', 'Valencia']
        }
        
        texto = doc.text
        for ent in doc.ents:
            if ent.label_ in entidades_por_tipo:
                reemplazo = random.choice(entidades_por_tipo[ent.label_])
                return texto.replace(ent.text, reemplazo)
        return texto
    
    def _modificar_cantidad(self, doc) -> str:
        """Modifica números o cantidades"""
        texto = doc.text
        for token in doc:
            if token.like_num:
                try:
                    num = int(token.text)
                    nuevo_num = num * 2 if num < 100 else num // 2
                    return texto.replace(token.text, str(nuevo_num))
                except ValueError:
                    pass
        return texto
    
    def _alterar_tiempo_verbal(self, doc) -> str:
        """Cambia el tiempo verbal de la oración"""
        cambios_tiempo = {
            'presente': 'futuro',
            'pasado': 'presente',
            'futuro': 'pasado'
        }
        
        texto = doc.text
        for token in doc:
            if token.pos_ == "VERB" and token.dep_ == "ROOT":
                if "pasado" in token.morph.get("Tense"):
                    return texto.replace(token.text, f"será {token.text}")
                elif "futuro" in token.morph.get("Tense"):
                    return texto.replace(token.text, f"fue {token.text}")
                else:
                    return texto.replace(token.text, f"fue {token.text}")
        return texto

def generar_cuestionario(ruta_archivo: str) -> Dict:
    """Genera un cuestionario completo a partir del archivo JSON"""
    try:
        with open(ruta_archivo, 'r', encoding='utf-8') as archivo:
            datos = json.load(archivo)
    except Exception as e:
        print(f"Error al cargar el archivo JSON: {e}")
        return None

    generador = GeneradorPreguntas()
    cuestionarios = {}

    for item in datos['quiz']:
        materia = item['materia']
        if materia not in cuestionarios:
            cuestionarios[materia] = []

        texto = item['texto']
        preguntas = []
        
        # Extraer temas principales
        temas = generador.extraer_temas_principales(texto)
        
        # Generar preguntas de análisis
        for _ in range(7):
            if temas:
                tema = random.choice(temas)
                pregunta, respuesta = generador.generar_pregunta_analisis(tema, texto)
                if pregunta and respuesta:
                    alternativas, indice_correcto = generador.generar_alternativas_coherentes(
                        respuesta, texto
                    )
                    preguntas.append({
                        "pregunta": pregunta,
                        "opciones": alternativas,
                        "respuesta_correcta": alternativas[indice_correcto],
                        "tipo": "alternativas"
                    })

        # Generar preguntas de verdadero/falso
        for _ in range(3):
            pregunta, respuesta = generador.generar_pregunta_verdadero_falso(texto)
            if pregunta and respuesta:
                preguntas.append({
                    "pregunta": pregunta,
                    "opciones": ["Verdadero", "Falso"],
                    "respuesta_correcta": respuesta,
                    "tipo": "verdadero_falso"
                })

        cuestionarios[materia].append({
            "texto": texto,
            "fuente": item['fuente'],
            "preguntas": preguntas
        })

    return cuestionarios