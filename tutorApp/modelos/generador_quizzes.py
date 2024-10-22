# Generador de Quizzes
# por Franco Benassi
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
    pipeline
)
import spacy
import json
import random
from typing import List, Dict, Any

class GeneradorCuestionarios:
    def __init__(self):
        print("Inicializando modelos...")
        
        # Modelo FLAN-T5 para generación de preguntas analíticas
        self.qa_model_name = "google/flan-t5-xl"
        self.qa_tokenizer = AutoTokenizer.from_pretrained(self.qa_model_name)
        self.qa_model = T5ForConditionalGeneration.from_pretrained(self.qa_model_name)
        
        # DeBERTa multilingüe para verificación semántica
        self.verifier = pipeline(
            "question-answering",
            model="microsoft/mdeberta-v3-base",
            tokenizer="microsoft/mdeberta-v3-base"
        )
        
        # SpaCy para análisis semántico en español
        self.nlp = spacy.load("es_core_news_lg")

    def _extraer_contextos(self, texto: str) -> List[Dict]:
        """Extrae contextos relevantes del texto para generar preguntas"""
        doc = self.nlp(texto)
        contextos = []
        
        # Identificar párrafos relevantes
        parrafos = texto.split('\n\n')
        for parrafo in parrafos:
            if len(parrafo.split()) > 20:  # Párrafos sustanciales
                doc_parrafo = self.nlp(parrafo)
                
                # Extraer entidades y conceptos clave
                entidades = []
                conceptos = []
                relaciones = []
                
                for ent in doc_parrafo.ents:
                    if ent.label_ in ['PER', 'ORG', 'LOC', 'MISC']:
                        entidades.append({
                            'texto': ent.text,
                            'tipo': ent.label_
                        })
                
                for token in doc_parrafo:
                    if token.pos_ in ['NOUN', 'VERB', 'ADJ'] and not token.is_stop:
                        conceptos.append({
                            'texto': token.text,
                            'tipo': token.pos_,
                            'importancia': token.vector_norm
                        })
                    
                    if token.dep_ in ['nsubj', 'dobj', 'pobj']:
                        relaciones.append({
                            'origen': token.head.text,
                            'tipo': token.dep_,
                            'destino': token.text
                        })
                
                if entidades or conceptos:
                    contextos.append({
                        'texto': parrafo,
                        'entidades': entidades,
                        'conceptos': conceptos,
                        'relaciones': relaciones
                    })
        
        return contextos

    def _generar_pregunta_analitica(self, contexto: Dict) -> Dict:
        """Genera una pregunta analítica con alternativas coherentes"""
        try:
            # Construir prompt para generar pregunta
            conceptos_str = ", ".join([c['texto'] for c in contexto['conceptos']])
            prompt = f"""
            Genera una pregunta de análisis en español sobre este texto:
            {contexto['texto']}
            
            La pregunta debe:
            1. Requerir comprensión y análisis
            2. Relacionar estos conceptos: {conceptos_str}
            3. Tener una respuesta específica pero no obvia
            4. Ser clara y bien formulada
            
            Formato: Pregunta?|Respuesta correcta
            """
            
            # Generar pregunta y respuesta correcta
            inputs = self.qa_tokenizer(prompt, return_tensors="pt", max_length=512, 
                                   truncation=True, padding=True)
            outputs = self.qa_model.generate(
                inputs.input_ids,
                max_length=150,
                num_beams=5,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            resultado = self.qa_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Separar pregunta y respuesta
            partes = resultado.split('|')
            if len(partes) != 2:
                return None
                
            pregunta, respuesta_correcta = partes
            
            # Generar alternativas incorrectas pero plausibles
            alternativas = [respuesta_correcta]
            
            # Prompt para alternativas
            prompt_alt = f"""
            Pregunta: {pregunta}
            Respuesta correcta: {respuesta_correcta}
            
            Genera una respuesta incorrecta pero plausible que:
            1. Use conceptos relacionados: {conceptos_str}
            2. Tenga un nivel similar de detalle
            3. Pueda parecer correcta a primera vista
            4. Sea diferente a: {', '.join(alternativas)}
            """
            
            for _ in range(4):  # Generar más alternativas de las necesarias
                inputs = self.qa_tokenizer(prompt_alt, return_tensors="pt", 
                                       max_length=512, truncation=True)
                outputs = self.qa_model.generate(
                    inputs.input_ids,
                    max_length=100,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    num_beams=4
                )
                
                alternativa = self.qa_tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Validar calidad de la alternativa
                if self._validar_alternativa(alternativa, alternativas, pregunta, contexto):
                    alternativas.append(alternativa)
                
                if len(alternativas) >= 4:
                    break
            
            if len(alternativas) >= 4:
                random.shuffle(alternativas)
                return {
                    "pregunta": pregunta.strip(),
                    "opciones": alternativas,
                    "respuesta_correcta": respuesta_correcta,
                    "tipo": "alternativas"
                }
            
            return None
            
        except Exception as e:
            print(f"Error al generar pregunta analítica: {str(e)}")
            return None

    def _validar_alternativa(
        self,
        alternativa: str,
        alternativas_existentes: List[str],
        pregunta: str,
        contexto: Dict
    ) -> bool:
        """Valida que una alternativa sea coherente y única"""
        if not alternativa or len(alternativa.split()) < 4:
            return False
            
        doc_alt = self.nlp(alternativa)
        
        # Verificar coherencia gramatical
        tiene_sentido = False
        for token in doc_alt:
            if token.pos_ == 'VERB':
                tiene_sentido = True
                break
                
        if not tiene_sentido:
            return False
            
        # Verificar similitud con otras alternativas
        for existente in alternativas_existentes:
            doc_existente = self.nlp(existente)
            similitud = doc_alt.similarity(doc_existente)
            if similitud > 0.7:  # Muy similar
                return False
        
        # Verificar relación con el contexto
        conceptos_texto = " ".join([c['texto'] for c in contexto['conceptos']])
        similitud_contexto = self.nlp(conceptos_texto).similarity(doc_alt)
        if similitud_contexto < 0.3:  # Poco relacionada
            return False
            
        return True

    def _generar_pregunta_vf(self, contexto: Dict) -> Dict:
        """Genera una pregunta de verdadero/falso analítica"""
        try:
            # Construir prompt para generar afirmación
            prompt = f"""
            Basado en este texto:
            {contexto['texto']}
            
            Genera una afirmación en español que:
            1. Sea específica y verificable
            2. Requiera comprensión del texto
            3. No sea obviamente verdadera o falsa
            4. Use estos conceptos: {', '.join([c['texto'] for c in contexto['conceptos']])}
            
            Formato: Afirmación|Verdadero/Falso|Explicación
            """
            
            inputs = self.qa_tokenizer(prompt, return_tensors="pt", max_length=512, 
                                   truncation=True)
            outputs = self.qa_model.generate(
                inputs.input_ids,
                max_length=150,
                do_sample=True,
                temperature=0.7,
                num_beams=4
            )
            
            resultado = self.qa_tokenizer.decode(outputs[0], skip_special_tokens=True)
            partes = resultado.split('|')
            
            if len(partes) != 3:
                return None
                
            afirmacion, es_verdadero, explicacion = partes
            
            # Validar con el verificador
            verificacion = self.verifier(
                question=afirmacion,
                context=contexto['texto']
            )
            
            # Solo usar si hay alta confianza
            if verificacion['score'] > 0.7:
                return {
                    "pregunta": afirmacion.strip(),
                    "opciones": ["Verdadero", "Falso"],
                    "respuesta_correcta": es_verdadero.strip(),
                    "explicacion": explicacion.strip(),
                    "tipo": "verdadero_falso"
                }
            
            return None
            
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
            
            # Extraer contextos relevantes
            contextos = self._extraer_contextos(texto)
            if not contextos:
                print(f"No se pudieron extraer contextos para {materia}")
                continue
            
            preguntas = []
            intentos = 0
            max_intentos = 20
            
            # Generar preguntas de alternativas (7)
            while len([p for p in preguntas if p['tipo'] == 'alternativas']) < 7 and intentos < max_intentos:
                contexto = random.choice(contextos)
                pregunta = self._generar_pregunta_analitica(contexto)
                if pregunta and self._es_pregunta_valida(pregunta, preguntas):
                    preguntas.append(pregunta)
                intentos += 1
            
            # Generar preguntas de verdadero/falso (3)
            intentos = 0
            while len([p for p in preguntas if p['tipo'] == 'verdadero_falso']) < 3 and intentos < max_intentos:
                contexto = random.choice(contextos)
                pregunta = self._generar_pregunta_vf(contexto)
                if pregunta and self._es_pregunta_valida(pregunta, preguntas):
                    preguntas.append(pregunta)
                intentos += 1
            
            # Validar y seleccionar mejores preguntas
            if len(preguntas) >= 8:
                preguntas_validadas = self._validar_cuestionario(preguntas)
                if len(preguntas_validadas) >= 10:
                    cuestionarios[materia].extend(preguntas_validadas[:10])
                else:
                    print(f"No se generaron suficientes preguntas válidas para {materia}")
            else:
                print(f"No se generaron suficientes preguntas para {materia}")
        
        return cuestionarios

    def _es_pregunta_valida(self, pregunta: Dict, preguntas_existentes: List[Dict]) -> bool:
        """Verifica si una pregunta es válida y única"""
        if not pregunta:
            return False
            
        # Verificar duplicados aproximados
        doc_pregunta = self.nlp(pregunta['pregunta'])
        for p in preguntas_existentes:
            similitud = self.nlp(p['pregunta']).similarity(doc_pregunta)
            if similitud > 0.6:
                return False
        
        # Verificar estructura gramatical
        doc = self.nlp(pregunta['pregunta'])
        tiene_estructura = False
        for token in doc:
            if token.dep_ == 'ROOT' and token.pos_ == 'VERB':
                tiene_estructura = True
                break
                
        if not tiene_estructura:
            return False
            
        # Verificar longitud mínima
        if len(pregunta['pregunta'].split()) < 5:
            return False
            
        # Verificar ortografía básica
        if not self._verificar_ortografia(pregunta['pregunta']):
            return False
            
        return True

    def _validar_cuestionario(self, preguntas: List[Dict]) -> List[Dict]:
        """Valida y selecciona las mejores preguntas del cuestionario"""
        preguntas_validadas = []
        
        for pregunta in preguntas:
            # Validar formato y coherencia
            if not self._verificar_ortografia(pregunta['pregunta']):
                continue
                
            # Validar opciones
            if pregunta['tipo'] == 'alternativas':
                opciones_validas = True
                for opcion in pregunta['opciones']:
                    if not self._verificar_ortografia(opcion) or len(opcion.split()) < 3:
                        opciones_validas = False
                        break
                        
                if not opciones_validas:
                    continue
                    
                # Verificar respuesta correcta
                if pregunta['respuesta_correcta'] not in pregunta['opciones']:
                    continue
            
            preguntas_validadas.append(pregunta)
        
        # Ordenar por calidad (podría implementarse un scoring más complejo)
        random.shuffle(preguntas_validadas)
        return preguntas_validadas

    def _verificar_ortografia(self, texto: str) -> bool:
        """Realiza verificaciones básicas de ortografía y formato"""
        if not texto:
            return False
            
        # Verificar caracteres básicos
        if not all(c.isalnum() or c.isspace() or c in '.,;:¿?¡!()[]' for c in texto):
            return False
            
        # Verificar signos de puntuación balanceados
        if texto.count('(') != texto.count(')'):
            return False
        if texto.count('[') != texto.count(']'):
            return False
        if texto.count('¿') != texto.count('?'):
            return False
        if texto.count('¡') != texto.count('!'):
            return False
            
        # Verificar espacios después de signos de puntuación
        for signo in ['.', ',', ';', ':']:
            for i, c in enumerate(texto[:-1]):
                if c == signo and texto[i + 1] != ' ':
                    return False
                    
        # Verificar mayúsculas después de punto
        for i, c in enumerate(texto[:-2]):
            if c == '.' and texto[i + 1] == ' ' and not texto[i + 2].isupper():
                return False
                
        # Verificar espacios múltiples
        if '  ' in texto:
            return False
            
        # Verificar mayúscula inicial
        if not texto[0].isupper():
            return False
            
        # Verificar terminación adecuada
        if not texto[-1] in '.!?':
            return False
            
        # Verificar longitud mínima de palabras
        palabras = texto.split()
        if any(len(palabra) < 2 for palabra in palabras):
            return False
            
        return True

    def _validar_respuesta(self, respuesta: str, pregunta: str, contexto: str) -> bool:
        """Valida que una respuesta sea coherente con la pregunta y el contexto"""
        try:
            # Verificar coherencia semántica
            doc_respuesta = self.nlp(respuesta)
            doc_pregunta = self.nlp(pregunta)
            
            # Verificar relación entre pregunta y respuesta
            similitud = doc_respuesta.similarity(doc_pregunta)
            if similitud < 0.3:  # Umbral mínimo de relación
                return False
                
            # Verificar que la respuesta use conceptos del contexto
            doc_contexto = self.nlp(contexto)
            similitud_contexto = doc_respuesta.similarity(doc_contexto)
            if similitud_contexto < 0.4:  # Debe estar relacionada con el contexto
                return False
                
            # Verificar estructura gramatical básica
            tiene_verbo = False
            tiene_sustantivo = False
            for token in doc_respuesta:
                if token.pos_ == 'VERB':
                    tiene_verbo = True
                elif token.pos_ in ['NOUN', 'PROPN']:
                    tiene_sustantivo = True
                    
            if not (tiene_verbo and tiene_sustantivo):
                return False
                
            # Verificar longitud adecuada
            palabras = respuesta.split()
            if len(palabras) < 4 or len(palabras) > 30:
                return False
                
            return True
            
        except Exception as e:
            print(f"Error al validar respuesta: {str(e)}")
            return False

    def _validar_coherencia_alternativas(self, opciones: List[str], respuesta_correcta: str) -> bool:
        """Valida que las alternativas sean coherentes entre sí"""
        try:
            if len(opciones) < 4:
                return False
                
            # Verificar que todas las alternativas sean del mismo tipo
            tipos_gramaticales = set()
            for opcion in opciones:
                doc = self.nlp(opcion)
                tipo = [token.pos_ for token in doc]
                tipos_gramaticales.add(tuple(tipo))
                
            if len(tipos_gramaticales) > 2:  # Permite cierta variación
                return False
                
            # Verificar longitud similar
            longitudes = [len(opcion.split()) for opcion in opciones]
            promedio = sum(longitudes) / len(longitudes)
            if any(abs(l - promedio) > 5 for l in longitudes):
                return False
                
            # Verificar que la respuesta correcta no destaque
            doc_correcta = self.nlp(respuesta_correcta)
            similitudes = []
            for opcion in opciones:
                if opcion != respuesta_correcta:
                    doc_opcion = self.nlp(opcion)
                    similitud = doc_opcion.similarity(doc_correcta)
                    similitudes.append(similitud)
                    
            # Las alternativas incorrectas deben ser suficientemente similares
            if min(similitudes) < 0.3:
                return False
                
            return True
            
        except Exception as e:
            print(f"Error al validar coherencia de alternativas: {str(e)}")
            return False