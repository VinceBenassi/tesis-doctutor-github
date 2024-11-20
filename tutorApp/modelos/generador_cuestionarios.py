# Generador de Cuestionarios
# Por Franco Benassi
import torch
import json
import os
import random
import spacy
import re
from typing import List, Dict, Any
from collections import defaultdict
from transformers import pipeline

class GeneradorCuestionarios:
    def __init__(self):
        # Añadir stopwords en español
        self.stopwords = set(['el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'y', 'o', 'pero', 'si',
                            'bien', 'mal', 'que', 'cual', 'quien', 'donde', 'cuando', 'como', 'para', 'por',
                            'sin', 'sobre', 'entre', 'detrás', 'después', 'ante', 'antes', 'desde', 'hacia',
                            'hasta', 'en', 'con', 'de', 'del', 'al', 'este', 'esta', 'estos', 'estas'])
        
        # Expandir diccionario de contrarios con más pares relevantes
        self.contrarios = {
            "aumenta": "disminuye",
            "mejora": "empeora",
            "incrementa": "reduce",
            "positivo": "negativo",
            "beneficioso": "perjudicial",
            "siempre": "nunca",
            "todos": "ninguno",
            "más": "menos",
            "mayor": "menor",
            "mejor": "peor",
            "superior": "inferior",
            "alto": "bajo",
            "mucho": "poco",
            "rápido": "lento",
            "fácil": "difícil",
            "correcto": "incorrecto",
            "verdadero": "falso",
            "claro": "confuso",
            "importante": "insignificante",
            "efectivo": "inefectivo",
            "útil": "inútil",
            "necesario": "innecesario",
            "simple": "complejo",
            "directo": "indirecto",
            "favorable": "desfavorable"
        }
        
        # Inicializar modelos con configuración mejorada
        try:
            self.nlp = spacy.load("es_core_news_lg")  # Intentar cargar modelo grande primero
        except OSError:
            try:
                self.nlp = spacy.load("es_core_news_md")  # Intentar modelo mediano como fallback
            except OSError:
                print("Warning: Using small model. For better results, install a larger model:")
                print("python -m spacy download es_core_news_lg")
                self.nlp = spacy.load("es_core_news_sm")
        
        self.qa_model = pipeline("question-answering", 
                            model="dccuchile/bert-base-spanish-wwm-cased",
                            device="cuda" if torch.cuda.is_available() else "cpu")

    def _limpiar_oracion(self, oracion: str) -> str:
        """Limpia y normaliza una oración de manera más robusta."""
        # Eliminar espacios y caracteres innecesarios al inicio
        oracion = re.sub(r'^\s*[,\s]*', '', oracion.strip())
        
        # Eliminar espacios múltiples y normalizar puntuación
        oracion = re.sub(r'\s+', ' ', oracion)
        oracion = re.sub(r'([.,;:¿?¡!])\s*([.,;:¿?¡!])', r'\1', oracion)
        
        # Corregir espacios alrededor de puntuación
        oracion = re.sub(r'\s*([.,;:¿?¡!])', r'\1', oracion)
        oracion = re.sub(r'([¿¡])\s*', r'\1', oracion)
        
        # Asegurar que empiece con mayúscula
        if oracion and oracion[0].isalpha():
            oracion = oracion[0].upper() + oracion[1:]
            
        # Asegurar punto final
        if not oracion.endswith('.'):
            oracion = oracion.rstrip('.,') + '.'
            
        return oracion

    def _generar_pregunta_especifica(self, concepto: str, oracion: str) -> str:
        """Genera una pregunta específica sobre un concepto."""
        plantillas = [
            f"¿Cuál de las siguientes afirmaciones describe mejor {concepto}?",
            f"En relación con {concepto}, ¿qué afirmación es correcta?",
            f"¿Qué característica es verdadera sobre {concepto}?",
            f"Respecto a {concepto}, ¿cuál de las siguientes opciones es acertada?"
        ]
        return random.choice(plantillas)

    def _extraer_informacion_relevante(self, oracion: str, concepto: str) -> str:
        """Extrae la información relevante de una oración sobre un concepto."""
        # Si la oración es lo suficientemente corta, usarla completa
        if len(oracion.split()) < 30:
            return oracion
            
        # Intentar extraer la parte más relevante usando el modelo QA
        try:
            respuesta = self.qa_model(
                question=f"¿Qué se dice sobre {concepto}?",
                context=oracion
            )
            if respuesta['score'] > 0.5:  # Si la confianza es suficiente
                return respuesta['answer']
        except:
            pass
            
        # Si no se puede usar QA, retornar la oración completa
        return oracion

    def _es_distractor_valido(self, distractor: str, respuesta_correcta: str) -> bool:
        """Verifica si un distractor es válido."""
        if not distractor or not respuesta_correcta:
            return False
            
        # Verificar longitud mínima
        if len(distractor.split()) < 5:
            return False
            
        # Verificar que no sea muy similar a la respuesta correcta
        if self._calcular_similitud(distractor, respuesta_correcta) > 0.8:
            return False
            
        # Verificar que no contenga patrones inválidos
        patrones_invalidos = [
            r'no\s+no',
            r'\b(el|la|los|las)\s+\1\b',
            r'\s{2,}',
        ]
        
        for patron in patrones_invalidos:
            if re.search(patron, distractor.lower()):
                return False
                
        return True

    def _es_oracion_valida(self, oracion: str) -> bool:
        """Verifica si una oración es válida."""
        if not oracion:
            return False
            
        # Verificar longitud mínima
        if len(oracion.split()) < 5:
            return False
            
        # Verificar que no tenga caracteres o patrones inválidos
        patrones_invalidos = [
            r'[^\w\s.,;:¿?¡!áéíóúñÁÉÍÓÚÑ]',
            r'\s{2,}',
            r'^[^A-ZÁÉÍÓÚÑ]'  # Debe empezar con mayúscula
        ]
        
        for patron in patrones_invalidos:
            if re.search(patron, oracion):
                return False
                
        return True

    def _generar_explicacion_vf_mejorada(self, oracion_base: str, pregunta: str, es_verdadero: bool, texto: str) -> str:
        """Genera una explicación mejorada para preguntas de verdadero/falso."""
        doc = self.nlp(texto)
        
        # Encontrar oraciones relacionadas
        oraciones_relacionadas = []
        for sent in doc.sents:
            if sent.text != oracion_base and self._tienen_palabras_comunes(sent.text, oracion_base):
                oraciones_relacionadas.append(sent.text)
        
        # Construir explicación
        if es_verdadero:
            explicacion = f"Esta afirmación es correcta porque {oracion_base.lower()}"
            if oraciones_relacionadas:
                explicacion += f", lo cual se evidencia también cuando el texto menciona que {oraciones_relacionadas[0].lower()}"
        else:
            explicacion = f"Esta afirmación es incorrecta. Lo correcto es que {oracion_base.lower()}"
            if oraciones_relacionadas:
                explicacion += f". Esto se confirma cuando el texto menciona que {oraciones_relacionadas[0].lower()}"
        
        return explicacion

    def _modificar_sujeto(self, oracion: str) -> str:
        """Modifica el sujeto de una oración."""
        doc = self.nlp(oracion)
        for token in doc:
            if token.dep_ == 'nsubj':
                # Intentar reemplazar con un sinónimo o relacionado
                sinonimos = self._obtener_sinonimos_o_relacionados(token.text)
                if sinonimos:
                    return oracion.replace(token.text, random.choice(sinonimos))
        return None

    def _modificar_verbo(self, oracion: str) -> str:
        """Modifica el verbo principal de una oración."""
        doc = self.nlp(oracion)
        for token in doc:
            if token.pos_ == 'VERB' and token.dep_ == 'ROOT':
                modificadores = ['puede', 'debe', 'suele', 'tiende a']
                return oracion.replace(token.text, f"{random.choice(modificadores)} {token.text}")
        return None

    def _modificar_objeto(self, oracion: str) -> str:
        """Modifica el objeto de una oración."""
        doc = self.nlp(oracion)
        for token in doc:
            if token.dep_ == 'dobj':
                sinonimos = self._obtener_sinonimos_o_relacionados(token.text)
                if sinonimos:
                    return oracion.replace(token.text, random.choice(sinonimos))
        return None

    def _invertir_significado(self, oracion: str) -> str:
        """Invierte el significado principal de una oración."""
        for palabra, contrario in self.contrarios.items():
            if palabra in oracion.lower():
                return oracion.replace(palabra, contrario)
        return None

    def _cambiar_cantidad(self, oracion: str) -> str:
        """Cambia las cantidades numéricas en una oración."""
        def modificar_numero(match):
            num = int(match.group())
            return str(num * 2 if num < 100 else num // 2)
        
        return re.sub(r'\d+', modificar_numero, oracion)

    def _calcular_similitud(self, texto1: str, texto2: str) -> float:
        """Calcula la similitud entre dos textos."""
        doc1 = self.nlp(texto1.lower())
        doc2 = self.nlp(texto2.lower())
        return doc1.similarity(doc2)

    def _tienen_palabras_comunes(self, texto1: str, texto2: str) -> bool:
        """Verifica si dos textos tienen palabras significativas en común."""
        palabras1 = set(token.text.lower() for token in self.nlp(texto1) 
                        if token.pos_ in ['NOUN', 'VERB', 'ADJ'] and len(token.text) > 3)
        palabras2 = set(token.text.lower() for token in self.nlp(texto2) 
                        if token.pos_ in ['NOUN', 'VERB', 'ADJ'] and len(token.text) > 3)
        return bool(palabras1 & palabras2)

    def analizar_texto(self, texto: str) -> Dict[str, Any]:
        """Analiza el texto para identificar conceptos clave y estructuras importantes."""
        doc = self.nlp(texto)
        
        # Mejorar la extracción de oraciones
        oraciones = [sent.text.strip() for sent in doc.sents 
                    if len(sent.text.split()) > 10 and not sent.text.startswith('?')]
        
        # Mejorar la extracción de entidades
        entidades = []
        for ent in doc.ents:
            if (ent.label_ in ["PER", "ORG", "CONCEPT", "EVENT", "MISC"] and 
                len(ent.text.split()) <= 4 and 
                ent.text.strip() not in entidades):
                entidades.append(ent.text.strip())
        
        # Mejorar la extracción de conceptos
        conceptos = []
        for chunk in doc.noun_chunks:
            if (2 <= len(chunk.text.split()) <= 4 and 
                not any(char in chunk.text for char in '?¿!¡') and
                chunk.text.strip() not in conceptos):
                conceptos.append(chunk.text.strip())
        
        # Extraer palabras clave adicionales
        keywords = [token.text for token in doc 
                if token.pos_ in ['NOUN', 'PROPN'] 
                and len(token.text) > 4 
                and token.text.lower() not in self.stopwords]  # Usar self.stopwords
        
        return {
            "oraciones": oraciones,
            "entidades": entidades,
            "conceptos": conceptos,
            "keywords": list(set(keywords))
        }

    def _extraer_conceptos(self, texto: str) -> List[str]:
        """Extrae conceptos clave del texto de manera más precisa."""
        doc = self.nlp(texto)
        conceptos = []
        
        # Buscar sustantivos compuestos y entidades nombradas
        for chunk in doc.noun_chunks:
            if 2 <= len(chunk.text.split()) <= 4:
                concepto = self._limpiar_oracion(chunk.text)
                if self._es_concepto_valido(concepto):
                    conceptos.append(concepto)
                    
        # Añadir entidades nombradas relevantes
        for ent in doc.ents:
            if ent.label_ in ["CONCEPT", "PRODUCT", "ORG", "PERSON"]:
                concepto = self._limpiar_oracion(ent.text)
                if self._es_concepto_valido(concepto):
                    conceptos.append(concepto)
        
        return list(set(conceptos))

    def _extraer_definiciones(self, oraciones: List[str]) -> List[Dict[str, str]]:
        """Extrae definiciones del texto"""
        definiciones = []
        patrones_definicion = [
            r'([^.]+)\s+es\s+([^.]+)',
            r'([^.]+)\s+se define como\s+([^.]+)',
            r'([^.]+)\s+significa\s+([^.]+)',
            r'([^.]+)\s+se refiere a\s+([^.]+)'
        ]

        for oracion in oraciones:
            for patron in patrones_definicion:
                matches = re.findall(patron, oracion, re.IGNORECASE)
                for match in matches:
                    if len(match) == 2:
                        definiciones.append({
                            'concepto': match[0].strip(),
                            'definicion': match[1].strip()
                        })

        return definiciones

    def _extraer_relaciones(self, oraciones: List[str]) -> List[Dict[str, Any]]:
        """Extrae relaciones entre conceptos"""
        relaciones = []
        patrones_relacion = [
            r'([^.]+)\s+depende de\s+([^.]+)',
            r'([^.]+)\s+influye en\s+([^.]+)',
            r'([^.]+)\s+afecta a\s+([^.]+)',
            r'([^.]+)\s+se relaciona con\s+([^.]+)'
        ]

        for oracion in oraciones:
            for patron in patrones_relacion:
                matches = re.findall(patron, oracion, re.IGNORECASE)
                for match in matches:
                    if len(match) == 2:
                        relaciones.append({
                            'concepto1': match[0].strip(),
                            'concepto2': match[1].strip(),
                            'tipo_relacion': 'asociación'
                        })

        return relaciones

    def _extraer_procesos(self, oraciones: List[str]) -> List[Dict[str, List[str]]]:
        """Extrae procesos o secuencias del texto"""
        procesos = []
        inicio_proceso = [
            'primero', 'inicialmente', 'para comenzar',
            'el proceso comienza', 'el primer paso'
        ]

        proceso_actual = None
        pasos = []

        for oracion in oraciones:
            # Detectar inicio de proceso
            if any(inicio in oracion.lower() for inicio in inicio_proceso):
                if proceso_actual and pasos:
                    procesos.append({
                        'nombre': proceso_actual,
                        'pasos': pasos
                    })
                proceso_actual = oracion
                pasos = []
            # Continuar proceso actual
            elif proceso_actual and any(conector in oracion.lower() for conector in ['luego', 'después', 'entonces', 'finalmente']):
                pasos.append(oracion)

        # Agregar último proceso si existe
        if proceso_actual and pasos:
            procesos.append({
                'nombre': proceso_actual,
                'pasos': pasos
            })

        return procesos

    def _generar_pregunta_definicion(self, analisis: Dict[str, Any]) -> Dict[str, Any]:
        """Genera una pregunta sobre definición de conceptos"""
        if not analisis['definiciones']:
            return None

        definicion = random.choice(analisis['definiciones'])
        concepto = definicion['concepto']
        def_correcta = definicion['definicion']

        # Generar distractores modificando la definición correcta
        distractores = []
        palabras = def_correcta.split()
        for _ in range(3):
            distractor = palabras.copy()
            # Modificar algunas palabras al azar
            for i in range(min(3, len(distractor))):
                pos = random.randint(0, len(distractor)-1)
                distractor[pos] = random.choice(['diferente', 'similar', 'específico', 'general', 'único', 'especial'])
            distractores.append(' '.join(distractor))

        opciones = [def_correcta] + distractores
        random.shuffle(opciones)

        return {
            "tipo": "alternativas",
            "pregunta": f"¿Cuál es la definición correcta de {concepto}?",
            "opciones": opciones,
            "respuesta_correcta": def_correcta,
            "explicacion": f"La definición correcta de {concepto} es '{def_correcta}'. Esta definición captura la esencia del concepto y sus características principales."
        }

    def _generar_pregunta_concepto(self, analisis: Dict[str, Any]) -> Dict[str, Any]:
        """Genera una pregunta sobre características de conceptos"""
        if not analisis['conceptos']:
            return None

        concepto = random.choice(analisis['conceptos'])
        # Buscar oraciones que mencionan el concepto
        oraciones_relacionadas = [
            o for o in analisis['oraciones'] 
            if concepto.lower() in o.lower()
        ]

        if not oraciones_relacionadas:
            return None

        oracion_principal = random.choice(oraciones_relacionadas)
        caracteristica_correcta = oracion_principal.replace(concepto, "").strip()

        return {
            "tipo": "verdadero_falso",
            "pregunta": f"{concepto} {caracteristica_correcta}",
            "respuesta_correcta": "Verdadero",
            "explicacion": f"Esta afirmación es verdadera porque {oracion_principal.lower()}"
        }

    def _generar_pregunta_relacion(self, analisis: Dict[str, Any]) -> Dict[str, Any]:
        """Genera una pregunta sobre relaciones entre conceptos"""
        if not analisis['relaciones']:
            return None

        relacion = random.choice(analisis['relaciones'])
        concepto1 = relacion['concepto1']
        concepto2 = relacion['concepto2']

        # Generar opciones de respuesta
        opciones = [
            f"{concepto1} está directamente relacionado con {concepto2}",
            f"{concepto1} es independiente de {concepto2}",
            f"{concepto1} no tiene relación con {concepto2}",
            f"{concepto2} es opuesto a {concepto1}"
        ]

        return {
            "tipo": "alternativas",
            "pregunta": f"¿Cuál es la relación correcta entre {concepto1} y {concepto2}?",
            "opciones": opciones,
            "respuesta_correcta": opciones[0],
            "explicacion": f"Existe una relación directa entre {concepto1} y {concepto2} porque {relacion.get('tipo_relacion', 'están conectados en el contexto dado')}"
        }

    def _generar_pregunta_proceso(self, analisis: Dict[str, Any]) -> Dict[str, Any]:
        """Genera una pregunta sobre procesos o secuencias"""
        if not analisis['procesos']:
            return None

        proceso = random.choice(analisis['procesos'])
        pasos = proceso['pasos']

        if len(pasos) < 2:
            return None

        # Crear una afirmación sobre el orden de los pasos
        paso1 = random.choice(pasos)
        paso2 = random.choice([p for p in pasos if p != paso1])
        orden_correcto = pasos.index(paso1) < pasos.index(paso2)

        afirmacion = f"En el proceso de {proceso['nombre']}, {paso1} ocurre antes que {paso2}"

        return {
            "tipo": "verdadero_falso",
            "pregunta": afirmacion,
            "respuesta_correcta": "Verdadero" if orden_correcto else "Falso",
            "explicacion": f"Esta afirmación es {'correcta' if orden_correcto else 'incorrecta'} porque en el proceso descrito, {'' if orden_correcto else 'no'} se sigue esta secuencia específica."
        }

    def _generar_pregunta_aplicacion(self, analisis: Dict[str, Any]) -> Dict[str, Any]:
        """Genera una pregunta sobre aplicación práctica de conceptos"""
        if not analisis['conceptos']:
            return None

        concepto = random.choice(analisis['conceptos'])
        
        # Generar opciones de aplicación
        opciones = [
            f"Aplicar {concepto} en situaciones prácticas",
            f"Memorizar la definición de {concepto}",
            f"Ignorar {concepto} en la práctica",
            f"Evitar el uso de {concepto}"
        ]

        return {
            "tipo": "alternativas",
            "pregunta": f"¿Cuál es la mejor manera de utilizar el conocimiento sobre {concepto}?",
            "opciones": opciones,
            "respuesta_correcta": opciones[0],
            "explicacion": f"La mejor manera de utilizar el conocimiento sobre {concepto} es aplicándolo en situaciones prácticas, ya que esto permite comprender su utilidad real y desarrollar competencias efectivas."
        }
    
    def generar_preguntas(self, texto: str, num_preguntas: int = 10) -> List[Dict[str, Any]]:
        """Genera múltiples preguntas dinámicamente."""
        analisis = self.analizar_texto(texto)
        preguntas = []
        preguntas_hash = set()  # Control de duplicados
        intentos = 0
        max_intentos = num_preguntas * 3  # Limitar intentos para evitar bucles infinitos

        # Añadir información del análisis
        analisis['definiciones'] = self._extraer_definiciones(analisis['oraciones'])
        analisis['relaciones'] = self._extraer_relaciones(analisis['oraciones'])
        analisis['procesos'] = self._extraer_procesos(analisis['oraciones'])

        # Lista de generadores de preguntas disponibles
        generadores = [
            ('alternativas', self.generar_pregunta_alternativas),
            ('verdadero_falso', self.generar_pregunta_verdadero_falso),
            ('definicion', self._generar_pregunta_definicion),
            ('concepto', self._generar_pregunta_concepto),
            ('relacion', self._generar_pregunta_relacion),
            ('proceso', self._generar_pregunta_proceso),
            ('aplicacion', self._generar_pregunta_aplicacion)
        ]

        while len(preguntas) < num_preguntas and intentos < max_intentos:
            # Seleccionar tipo de pregunta y generador
            tipo, generador = random.choice(generadores)
            
            try:
                pregunta = generador(analisis, texto) if tipo in ['alternativas', 'verdadero_falso'] else generador(analisis)
                
                if pregunta and self._validar_pregunta(pregunta):
                    # Crear hash de la pregunta para control de duplicados
                    pregunta_hash = hash(f"{pregunta['pregunta']}_{pregunta['respuesta_correcta']}")
                    
                    if pregunta_hash not in preguntas_hash:
                        preguntas_hash.add(pregunta_hash)
                        preguntas.append(pregunta)
            except Exception as e:
                print(f"Error generando pregunta tipo {tipo}: {str(e)}")
                
            intentos += 1

        return preguntas

    def generar_pregunta_alternativas(self, analisis: Dict[str, Any], texto: str) -> Dict[str, Any]:
        """Genera preguntas de opción múltiple más naturales y coherentes."""
        if not analisis["conceptos"] or not analisis["oraciones"]:
            return None
            
        # 1. Selección mejorada de concepto y contexto
        conceptos_relevantes = [
            c for c in analisis["conceptos"]
            if sum(1 for o in analisis["oraciones"] if c.lower() in o.lower()) >= 2
        ]
        
        if not conceptos_relevantes:
            return None
            
        concepto = random.choice(conceptos_relevantes)
        
        # 2. Búsqueda de oraciones relacionadas mejorada
        oraciones_relacionadas = [
            o for o in analisis["oraciones"]
            if concepto.lower() in o.lower() 
            and len(o.split()) >= 15  # Asegurar oraciones sustanciales
            and not any(p in o.lower() for p in ['?', '¿', '!', '¡'])
            and self._es_oracion_informativa(o)
        ]

        if not oraciones_relacionadas:
            return None

        # 3. Selección y preparación de la oración principal
        oracion_principal = self._seleccionar_mejor_oracion(oraciones_relacionadas)
        oracion_principal = self._limpiar_oracion(oracion_principal)
        
        # 4. Generación de pregunta contextual
        pregunta = self._generar_pregunta_contextual(concepto, oracion_principal)
        
        # 5. Generación de respuesta correcta mejorada
        respuesta_correcta = self._extraer_respuesta_correcta(oracion_principal, concepto)
        
        # 6. Generación de distractores más realistas
        distractores = self._generar_distractores_mejorados(
            texto, concepto, respuesta_correcta, oraciones_relacionadas
        )

        if len(distractores) < 3:
            return None

        # 7. Validación y formato final
        opciones = [respuesta_correcta] + distractores[:3]
        random.shuffle(opciones)

        # 8. Generación de explicación contextualizada
        explicacion = self._generar_explicacion_detallada(
            concepto, respuesta_correcta, oracion_principal, texto
        )

        return {
            "tipo": "alternativas",
            "pregunta": pregunta,
            "opciones": opciones,
            "respuesta_correcta": respuesta_correcta,
            "explicacion": explicacion
        }

    def _es_oracion_informativa(self, oracion: str) -> bool:
        """Verifica si una oración contiene información sustancial."""
        doc = self.nlp(oracion)
        
        # Verificar presencia de elementos informativos
        tiene_verbo = any(token.pos_ == "VERB" for token in doc)
        tiene_sustantivo = any(token.pos_ == "NOUN" for token in doc)
        tiene_contexto = len([token for token in doc if token.pos_ in ["ADJ", "ADV"]]) >= 1
        
        return tiene_verbo and tiene_sustantivo and tiene_contexto

    def _seleccionar_mejor_oracion(self, oraciones: List[str]) -> str:
        """Selecciona la oración más apropiada para generar una pregunta."""
        oraciones_puntuadas = []
        
        for oracion in oraciones:
            puntuacion = 0
            doc = self.nlp(oracion)
            
            # Criterios de puntuación
            if len(oracion.split()) >= 15 and len(oracion.split()) <= 40:
                puntuacion += 3
            
            if any(token.pos_ == "VERB" for token in doc):
                puntuacion += 2
                
            if any(token.dep_ == "nsubj" for token in doc):
                puntuacion += 2
                
            if not any(token.text.lower() in self.stopwords for token in doc):
                puntuacion += 1
                
            oraciones_puntuadas.append((oracion, puntuacion))
        
        return max(oraciones_puntuadas, key=lambda x: x[1])[0]

    def _generar_pregunta_contextual(self, concepto: str, oracion: str) -> str:
        """Genera una pregunta más natural y contextualizada."""
        doc = self.nlp(oracion)
        
        # Análisis más profundo del contexto
        contexto = {
            "definicion": any(token.text.lower() in ["es", "son", "significa", "consiste", "representa", "constituye"] 
                            for token in doc),
            "proceso": any(token.text.lower() in ["primero", "luego", "después", "finalmente", "siguiente", "entonces", 
                                                "posteriormente", "antes", "durante"] for token in doc),
            "caracteristica": any(token.pos_ in ["ADJ", "ADV"] for token in doc),
            "comparacion": any(token.text.lower() in ["más", "menos", "mayor", "menor", "mejor", "peor", "como"] 
                            for token in doc),
            "causa_efecto": any(token.text.lower() in ["porque", "debido", "causa", "produce", "genera", "resulta"] 
                            for token in doc)
        }
        
        # Plantillas específicas por tipo de contenido
        plantillas = {
            "definicion": [
                f"¿Cuál de las siguientes afirmaciones define mejor {concepto}?",
                f"¿Qué descripción caracteriza más precisamente a {concepto}?",
                f"En el contexto presentado, ¿qué se entiende por {concepto}?",
                f"¿Cuál es la definición más precisa de {concepto}?",
                f"¿Cómo se define mejor {concepto} en este contexto?"
            ],
            "proceso": [
                f"¿Cómo se desarrolla el proceso relacionado con {concepto}?",
                f"¿Cuál es la secuencia correcta que describe {concepto}?",
                f"¿Qué afirmación explica mejor el funcionamiento de {concepto}?",
                f"En relación con {concepto}, ¿qué pasos se siguen?",
                f"¿Cómo se lleva a cabo el proceso de {concepto}?"
            ],
            "caracteristica": [
                f"¿Qué característica define mejor a {concepto}?",
                f"¿Cuál es el aspecto más relevante de {concepto}?",
                f"¿Qué rasgo distintivo presenta {concepto}?",
                f"¿Cuál es la cualidad más importante de {concepto}?",
                f"¿Qué caracteriza principalmente a {concepto}?"
            ],
            "comparacion": [
                f"¿Qué diferencia a {concepto} de otros elementos similares?",
                f"En comparación con otros aspectos, ¿qué distingue a {concepto}?",
                f"¿Cuál es la principal ventaja de {concepto}?",
                f"¿Qué hace único a {concepto}?",
                f"¿Cómo se diferencia {concepto} de otras alternativas?"
            ],
            "causa_efecto": [
                f"¿Qué efectos produce {concepto}?",
                f"¿Cuál es la consecuencia principal de {concepto}?",
                f"¿Qué relación causal existe con {concepto}?",
                f"¿Cómo influye {concepto} en su contexto?",
                f"¿Qué impacto tiene {concepto}?"
            ],
            "general": [
                f"¿Qué aspecto de {concepto} se destaca en el texto?",
                f"¿Cuál es la idea principal sobre {concepto}?",
                f"¿Qué se puede afirmar con certeza sobre {concepto}?",
                f"¿Qué plantea el texto acerca de {concepto}?",
                f"¿Cuál es el punto más relevante sobre {concepto}?"
            ]
        }
        
        # Seleccionar tipo de pregunta basado en el contexto
        tipo_pregunta = max(contexto.items(), key=lambda x: x[1])[0] if any(contexto.values()) else "general"
        
        return random.choice(plantillas[tipo_pregunta])

    def _extraer_respuesta_correcta(self, oracion: str, concepto: str) -> str:
        """Extrae una respuesta correcta más precisa y natural."""
        doc = self.nlp(oracion)
        
        # Identificar la parte más relevante de la oración
        inicio_respuesta = None
        fin_respuesta = None
        
        for i, token in enumerate(doc):
            # Buscar el inicio después del concepto
            if concepto.lower() in token.text.lower():
                inicio_respuesta = i + 1
                continue
                
            # Buscar un punto natural de corte
            if inicio_respuesta and token.dep_ in ["punct", "cc"]:
                fin_respuesta = i
                break
        
        if inicio_respuesta:
            if not fin_respuesta:
                fin_respuesta = len(doc)
                
            respuesta = doc[inicio_respuesta:fin_respuesta].text
            return self._limpiar_oracion(respuesta)
            
        return self._limpiar_oracion(oracion)

    def generar_pregunta_verdadero_falso(self, analisis: Dict[str, Any], texto: str) -> Dict[str, Any]:
        """Genera preguntas de verdadero/falso más sofisticadas."""
        if not analisis["oraciones"]:
            return None

        # Filtrar y seleccionar oraciones válidas
        oraciones_validas = [
            o for o in analisis["oraciones"]
            if (len(o.split()) > 12 and
                not any(p in o.lower() for p in ['?', '¿', '!', '¡']) and
                any(c in o.lower() for c in analisis["conceptos"]))
        ]

        if not oraciones_validas:
            return None

        # Seleccionar oración base y determinar veracidad
        oracion_base = random.choice(oraciones_validas)
        es_verdadero = random.choice([True, False])

        if es_verdadero:
            pregunta = self._limpiar_oracion(oracion_base)
        else:
            pregunta = self._generar_version_falsa_mejorada(oracion_base)

        # Verificar que la pregunta sea válida
        if not self._es_pregunta_valida(pregunta):
            return None

        # Generar explicación contextualizada
        explicacion = self._generar_explicacion_vf_mejorada(
            oracion_base, pregunta, es_verdadero, texto
        )

        return {
            "tipo": "verdadero_falso",
            "pregunta": pregunta,
            "respuesta_correcta": "Verdadero" if es_verdadero else "Falso",
            "explicacion": explicacion
        }
    
    def _modificar_tiempo_verbal(self, oracion: str) -> str:
        """Modifica el tiempo verbal de una oración."""
        doc = self.nlp(oracion)
        modificaciones = {
            'presente': {'es': 'era', 'son': 'eran', 'está': 'estaba', 'están': 'estaban'},
            'pasado': {'era': 'es', 'eran': 'son', 'estaba': 'está', 'estaban': 'están'},
            'futuro': {'es': 'será', 'son': 'serán', 'está': 'estará', 'están': 'estarán'}
        }
        
        for token in doc:
            if token.pos_ == 'VERB':
                for tiempo, cambios in modificaciones.items():
                    if token.text.lower() in cambios:
                        return oracion.replace(token.text, cambios[token.text.lower()])
        return None

    def _generar_version_falsa_mejorada(self, oracion: str) -> str:
        """Genera una versión falsa más sofisticada de una oración."""
        doc = self.nlp(oracion)
        
        # Lista de estrategias ordenadas por prioridad
        estrategias = [
            self._invertir_significado,        # Cambiar el significado usando contrarios
            self._modificar_argumento,         # Modificar argumentos específicos
            self._cambiar_cantidad,            # Cambiar números/cantidades
            self._cambiar_alcance,             # Modificar el alcance/generalización
            self._agregar_excepcion,           # Añadir una excepción
            self._negar_verbo                  # Última opción: negar el verbo
        ]
        
        # Intentar cada estrategia hasta encontrar una modificación válida
        for estrategia in [e for e in estrategias if hasattr(self, e.__name__)]:
            try:
                mod = estrategia(oracion) if estrategia.__name__ != '_negar_verbo' else estrategia(oracion, None)
                if mod and mod != oracion and self._es_oracion_valida(mod):
                    return mod
            except Exception as e:
                continue
        
        # Si ninguna estrategia funcionó, retornar None
        return None
    
    def _es_pregunta_valida(self, pregunta: str) -> bool:
        """Valida que una pregunta cumpla con los criterios de calidad."""
        if not pregunta:
            return False
            
        # Verificar longitud mínima
        if len(pregunta.split()) < 8:
            return False
            
        # Evitar preguntas que empiezan con negación
        if pregunta.lower().startswith(('no ', 'no es')):
            return False
            
        # Evitar preguntas que contienen ciertos patrones problemáticos
        patrones_invalidos = [
            r'no\s+no',
            r'\b(el|la|los|las)\s+\1\b',
            r'\b(y|o)\s+\1\b',
            r'\s{2,}',
        ]
        
        for patron in patrones_invalidos:
            if re.search(patron, pregunta.lower()):
                return False
                
        return True
    
    def _negar_verbo(self, oracion: str, token) -> str:
        """Niega un verbo de manera más natural."""
        palabras = oracion.split()
        pos = token.i
        
        negaciones = {
            'es': 'no es',
            'son': 'no son',
            'está': 'no está',
            'están': 'no están',
            'puede': 'no puede',
            'tienen': 'no tienen',
            'ha': 'no ha',
            'han': 'no han'
        }
        
        if token.text.lower() in negaciones:
            palabras[pos] = negaciones[token.text.lower()]
        else:
            palabras.insert(pos, 'no')
        
        return ' '.join(palabras)

    def _obtener_sinonimos_o_relacionados(self, palabra: str) -> List[str]:
        """Retorna sinónimos o palabras relacionadas para generar distractores."""
        relacionados = {
            'aumentar': ['incrementar', 'crecer', 'subir'],
            'reducir': ['disminuir', 'bajar', 'decrecer'],
            'mejorar': ['optimizar', 'perfeccionar', 'desarrollar'],
            'importante': ['relevante', 'significativo', 'esencial'],
            'necesario': ['requerido', 'indispensable', 'fundamental'],
            'actual': ['presente', 'vigente', 'contemporáneo']
        }
        
        # Buscar en el diccionario de palabras relacionadas
        for key, values in relacionados.items():
            if palabra.lower().startswith(key):
                return values
        
        return []

    def _generar_explicacion_detallada(self, concepto: str, respuesta: str, 
                                 oracion: str, texto: str, es_correcta: bool = True) -> str:
        """Genera una explicación más clara y variada."""
        doc = self.nlp(texto)
        evidencias = []
        
        # Encontrar evidencias relevantes
        for sent in doc.sents:
            if concepto.lower() in sent.text.lower() and sent.text != oracion:
                similitud = self._calcular_similitud_semantica(sent.text, respuesta)
                if similitud > 0.3:
                    evidencias.append((sent.text, similitud))
        
        # Ordenar y seleccionar evidencias
        evidencias.sort(key=lambda x: x[1], reverse=True)
        evidencias = [ev[0] for ev in evidencias]
        
        # Construir explicación
        if es_correcta:
            plantillas_inicio = [
                f"Esta respuesta es correcta ya que {respuesta.lower()}.",
                f"La afirmación es acertada porque {respuesta.lower()}.",
                f"Es correcto dado que {respuesta.lower()}."
            ]
            explicacion = [random.choice(plantillas_inicio)]
            
            if evidencias:
                plantillas_evidencia = [
                    f"El texto lo confirma al mencionar que {evidencias[0].lower()}",
                    f"Esto se respalda cuando se indica que {evidencias[0].lower()}",
                    f"Se evidencia en el texto cuando señala que {evidencias[0].lower()}"
                ]
                explicacion.append(random.choice(plantillas_evidencia))
        else:
            plantillas_inicio = [
                f"Esta respuesta es incorrecta.",
                f"La afirmación no es precisa.",
                f"El planteamiento es erróneo."
            ]
            explicacion = [random.choice(plantillas_inicio)]
            explicacion.append(f"Lo correcto es que {oracion.lower()}")
            
            if evidencias:
                explicacion.append(f"Esto se demuestra cuando el texto menciona que {evidencias[0].lower()}")
        
        return " ".join(explicacion)

    def _calcular_similitud_semantica(self, texto1: str, texto2: str) -> float:
        """Calcula una similitud semántica más precisa entre textos."""
        doc1 = self.nlp(texto1.lower())
        doc2 = self.nlp(texto2.lower())
        
        # Extraer palabras clave
        palabras_clave1 = set(token.text for token in doc1 
                            if token.pos_ in ['NOUN', 'VERB', 'ADJ'] 
                            and not token.is_stop)
        palabras_clave2 = set(token.text for token in doc2 
                            if token.pos_ in ['NOUN', 'VERB', 'ADJ'] 
                            and not token.is_stop)
        
        # Calcular similitud de Jaccard de palabras clave
        similitud_jaccard = len(palabras_clave1 & palabras_clave2) / len(palabras_clave1 | palabras_clave2) if palabras_clave1 | palabras_clave2 else 0
        
        # Combinar con similitud de spaCy si está disponible
        try:
            similitud_spacy = doc1.similarity(doc2)
            return (similitud_jaccard + similitud_spacy) / 2
        except:
            return similitud_jaccard
    
    def _modificar_argumento(self, oracion: str) -> str:
        """Modifica argumentos específicos en la oración."""
        doc = self.nlp(oracion)
        for token in doc:
            if token.dep_ in ['dobj', 'pobj', 'attr']:
                # Buscar argumentos específicos para modificar
                if any(palabra in token.text.lower() for palabra in ['todos', 'siempre', 'nunca', 'nadie']):
                    return oracion.replace(token.text, random.choice(['algunos', 'ocasionalmente', 'raramente']))
                # Modificar cantidades o medidas
                if token.like_num:
                    num = int(token.text)
                    nuevo_num = num * 2 if num < 100 else num // 2
                    return oracion.replace(token.text, str(nuevo_num))
        return None

    def _cambiar_alcance(self, oracion: str) -> str:
        """Cambia el alcance o la generalización de una afirmación."""
        cuantificadores = {
            'todos': 'algunos',
            'siempre': 'a veces',
            'nunca': 'raramente',
            'completamente': 'parcialmente',
            'absolutamente': 'relativamente',
            'globalmente': 'localmente',
            'mundialmente': 'regionalmente'
        }
        
        for original, modificado in cuantificadores.items():
            if original in oracion.lower():
                return oracion.replace(original, modificado)
        return None

    def _agregar_excepcion(self, oracion: str) -> str:
        """Añade una excepción que invalida la afirmación principal."""
        excepciones = [
            ', excepto en casos específicos',
            ', aunque no siempre es así',
            ', pero con importantes limitaciones',
            ', si se cumplen ciertas condiciones',
            ', dependiendo del contexto'
        ]
        
        # Verificar que no termine en punto
        if oracion.endswith('.'):
            oracion = oracion[:-1]
        
        return oracion + random.choice(excepciones) + '.'


    def _generar_distractores_mejorados(self, texto: str, concepto: str, 
                                  respuesta_correcta: str, 
                                  oraciones_relacionadas: List[str]) -> List[str]:
        """Genera distractores más naturales y coherentes."""
        distractores = set()
        doc_respuesta = self.nlp(respuesta_correcta)
        
        # 1. Modificación basada en contrarios semánticos
        for token in doc_respuesta:
            if token.pos_ in ['VERB', 'ADJ', 'ADV'] and token.text.lower() in self.contrarios:
                nueva_oracion = respuesta_correcta.replace(
                    token.text, 
                    self.contrarios[token.text.lower()]
                )
                if self._es_distractor_valido(nueva_oracion, respuesta_correcta):
                    distractores.add(self._limpiar_oracion(nueva_oracion))
        
        # 2. Uso de información relacionada pero incorrecta
        for oracion in oraciones_relacionadas:
            if oracion != respuesta_correcta:
                doc_oracion = self.nlp(oracion)
                for chunk in doc_oracion.noun_chunks:
                    if len(chunk.text.split()) > 2 and concepto.lower() not in chunk.text.lower():
                        mod = self._modificar_chunk_semanticamente(chunk.text, oracion)
                        if mod and self._es_distractor_valido(mod, respuesta_correcta):
                            distractores.add(self._limpiar_oracion(mod))
        
        # 3. Modificación de estructura manteniendo significado similar
        modificaciones = [
            self._cambiar_orden_componentes(respuesta_correcta),
            self._agregar_modificador(respuesta_correcta),
            self._cambiar_tiempo_verbal(respuesta_correcta),
            self._modificar_determinantes(respuesta_correcta)
        ]
        
        for mod in modificaciones:
            if mod and self._es_distractor_valido(mod, respuesta_correcta):
                distractores.add(self._limpiar_oracion(mod))
        
        # Filtrar y seleccionar los mejores distractores
        distractores = list(distractores)
        distractores.sort(key=lambda x: self._evaluar_calidad_distractor(x, respuesta_correcta))
        
        return [d for d in distractores[:3] if self._es_oracion_coherente(d)]

    def _cambiar_tiempo_verbal(self, oracion: str) -> str:
        """Modifica el tiempo verbal de una oración de manera más robusta."""
        doc = self.nlp(oracion)
        palabras = oracion.split()
        
        # Mapeo más completo de tiempos verbales
        modificaciones = {
            # Presente a pasado
            'es': 'era',
            'son': 'eran',
            'está': 'estaba',
            'están': 'estaban',
            'tiene': 'tenía',
            'tienen': 'tenían',
            'puede': 'podía',
            'pueden': 'podían',
            'hace': 'hacía',
            'hacen': 'hacían',
            'va': 'iba',
            'van': 'iban',
            
            # Pasado a presente
            'era': 'es',
            'eran': 'son',
            'estaba': 'está',
            'estaban': 'están',
            'tenía': 'tiene',
            'tenían': 'tienen',
            'podía': 'puede',
            'podían': 'pueden',
            'hacía': 'hace',
            'hacían': 'hacen',
            'iba': 'va',
            'iban': 'van',
            
            # Presente a futuro
            'será': 'es',
            'serán': 'son',
            'estará': 'está',
            'estarán': 'están',
            'tendrá': 'tiene',
            'tendrán': 'tienen',
            'podrá': 'puede',
            'podrán': 'pueden',
            'hará': 'hace',
            'harán': 'hacen',
            'irá': 'va',
            'irán': 'van'
        }
        
        # Buscar y reemplazar verbos
        for token in doc:
            if token.pos_ == 'VERB' or token.pos_ == 'AUX':
                if token.text.lower() in modificaciones:
                    palabras[token.i] = modificaciones[token.text.lower()]
                    # Ajustar el resto de la oración si es necesario
                    if token.dep_ == 'ROOT':
                        # Modificar tiempos de verbos dependientes
                        for child in token.children:
                            if child.pos_ == 'VERB' and child.text.lower() in modificaciones:
                                palabras[child.i] = modificaciones[child.text.lower()]
        
        nueva_oracion = ' '.join(palabras)
        # Asegurar que la oración sigue siendo válida
        if self._es_oracion_valida(nueva_oracion):
            return nueva_oracion
        return None

    def _es_verbo_conjugado(self, token) -> bool:
        """Verifica si un token es un verbo conjugado."""
        return (token.pos_ in ['VERB', 'AUX'] and 
                not token.tag_.startswith('VB') and  # No infinitivo
                not token.tag_.startswith('VBG') and  # No gerundio
                not token.tag_.startswith('VBN'))     # No participio

    def _ajustar_concordancia(self, oracion: str) -> str:
        """Ajusta la concordancia gramatical después de modificar verbos."""
        doc = self.nlp(oracion)
        palabras = oracion.split()
        
        for token in doc:
            if token.pos_ == 'VERB':
                # Buscar sujeto
                sujeto = None
                for child in token.children:
                    if child.dep_ == 'nsubj':
                        sujeto = child
                        break
                
                if sujeto:
                    # Ajustar artículos y determinantes
                    for det in sujeto.children:
                        if det.pos_ == 'DET':
                            # Implementar ajustes de concordancia específicos
                            pass
        
        return ' '.join(palabras)


    def _modificar_chunk_semanticamente(self, chunk: str, oracion_contexto: str) -> str:
        """Modifica un fragmento de texto manteniendo coherencia semántica."""
        doc = self.nlp(chunk)
        palabras = chunk.split()
        
        # 1. Modificación basada en partes del discurso
        for token in doc:
            if token.pos_ in ['VERB', 'ADJ', 'ADV']:
                # Modificar verbos y adjetivos
                if token.text.lower() in self.contrarios:
                    chunk = chunk.replace(token.text, self.contrarios[token.text.lower()])
                    return chunk
        
        # 2. Agregar modificadores
        for token in doc:
            if token.pos_ == 'NOUN':
                modificadores = ['algunos', 'ciertos', 'diversos', 'diferentes', 'varios']
                return f"{random.choice(modificadores)} {chunk}"
        
        return chunk

    def _cambiar_orden_componentes(self, oracion: str) -> str:
        """Reorganiza los componentes de la oración manteniendo gramaticalidad."""
        doc = self.nlp(oracion)
        componentes = []
        
        # Identificar sujeto, verbo y complementos
        sujeto = None
        verbo = None
        complementos = []
        
        for token in doc:
            if token.dep_ == 'nsubj':
                sujeto = token.text
            elif token.pos_ == 'VERB':
                verbo = token.text
            elif token.dep_ in ['dobj', 'pobj', 'attr']:
                span = self._obtener_span_complemento(token)
                if span:
                    complementos.append(span)
        
        # Si no se identificaron los componentes principales, retornar None
        if not (sujeto and verbo):
            return None
        
        # Generar variantes de orden
        if complementos:
            comp = ' '.join(complementos)
            variantes = [
                f"{sujeto} {verbo} {comp}",
                f"{comp}, {sujeto} {verbo}",
                f"En cuanto a {sujeto}, {verbo} {comp}"
            ]
            return random.choice(variantes)
        
        return None

    def _agregar_modificador(self, oracion: str) -> str:
        """Agrega modificadores para cambiar sutilmente el significado."""
        modificadores = {
            'frecuencia': ['generalmente', 'usualmente', 'frecuentemente', 'a menudo'],
            'intensidad': ['considerablemente', 'significativamente', 'notablemente', 'marcadamente'],
            'modo': ['gradualmente', 'progresivamente', 'sistemáticamente', 'metodológicamente'],
            'certeza': ['probablemente', 'posiblemente', 'presumiblemente', 'aparentemente']
        }
        
        # Seleccionar tipo de modificador al azar
        tipo = random.choice(list(modificadores.keys()))
        modificador = random.choice(modificadores[tipo])
        
        # Insertar el modificador en una posición apropiada
        doc = self.nlp(oracion)
        for token in doc:
            if token.pos_ == 'VERB':
                return oracion.replace(token.text, f"{modificador} {token.text}")
        
        # Si no se encuentra una posición adecuada, agregar al inicio
        return f"{modificador}, {oracion}"

    def _modificar_determinantes(self, oracion: str) -> str:
        """Modifica los determinantes para cambiar el alcance."""
        doc = self.nlp(oracion)
        palabras = oracion.split()
        
        determinantes = {
            'el': ['cada', 'todo', 'cualquier'],
            'la': ['cada', 'toda', 'cualquier'],
            'los': ['algunos', 'varios', 'diversos'],
            'las': ['algunas', 'varias', 'diversas'],
            'un': ['algún', 'cualquier', 'cada'],
            'una': ['alguna', 'cualquier', 'cada']
        }
        
        # Buscar y reemplazar determinantes
        for i, token in enumerate(doc):
            if token.text.lower() in determinantes:
                nuevo_det = random.choice(determinantes[token.text.lower()])
                palabras[token.i] = nuevo_det
                return ' '.join(palabras)
        
        return None

    def _obtener_span_complemento(self, token) -> str:
        """Obtiene el span completo de un complemento."""
        inicio = token.i
        fin = token.i + 1
        
        # Buscar el inicio real del complemento
        while inicio > 0 and token.doc[inicio - 1].dep_ in ['det', 'amod', 'compound']:
            inicio -= 1
        
        # Buscar el final real del complemento
        while fin < len(token.doc) and token.doc[fin].dep_ in ['prep', 'pobj', 'acl', 'amod']:
            fin += 1
        
        return token.doc[inicio:fin].text

    def _es_concepto_valido(self, concepto: str) -> bool:
        """Verifica si un concepto es válido para generar preguntas."""
        # Verificar longitud
        if len(concepto.split()) < 2 or len(concepto.split()) > 4:
            return False
        
        # Verificar que no sea solo stopwords
        palabras_contenido = [p for p in concepto.split() 
                            if p.lower() not in self.stopwords]
        if not palabras_contenido:
            return False
        
        # Verificar estructura básica
        doc = self.nlp(concepto)
        tiene_sustantivo = any(token.pos_ == 'NOUN' for token in doc)
        tiene_determinante = any(token.pos_ == 'DET' for token in doc)
        
        return tiene_sustantivo and tiene_determinante

    def _es_oracion_coherente(self, oracion: str) -> bool:
        """Verifica que una oración tenga estructura y sentido coherente."""
        doc = self.nlp(oracion)
        
        # Verificar estructura básica
        tiene_sujeto = any(token.dep_ == "nsubj" for token in doc)
        tiene_verbo = any(token.pos_ == "VERB" for token in doc)
        tiene_predicado = any(token.dep_ in ["dobj", "attr", "acomp"] for token in doc)
        
        if not (tiene_sujeto and tiene_verbo):
            return False
            
        # Verificar coherencia semántica
        tiene_sentido = all(not token.is_oov for token in doc)
        
        # Verificar gramática
        es_gramatical = not any(token.dep_ == "dep" for token in doc)
        
        return tiene_sentido and es_gramatical

    def _evaluar_calidad_distractor(self, distractor: str, respuesta_correcta: str) -> float:
        """Evalúa la calidad de un distractor."""
        # Inicializar puntuación
        puntuacion = 0.0
        
        # Verificar longitud similar
        ratio_longitud = len(distractor) / len(respuesta_correcta)
        if 0.7 <= ratio_longitud <= 1.3:
            puntuacion += 0.3
            
        # Verificar estructura gramatical
        doc_distractor = self.nlp(distractor)
        doc_respuesta = self.nlp(respuesta_correcta)
        
        tiene_verbo_distractor = any(token.pos_ == "VERB" for token in doc_distractor)
        tiene_verbo_respuesta = any(token.pos_ == "VERB" for token in doc_respuesta)
        
        if tiene_verbo_distractor == tiene_verbo_respuesta:
            puntuacion += 0.3
            
        # Verificar palabras clave similares pero no idénticas
        palabras_distractor = set(token.text.lower() for token in doc_distractor 
                                if token.pos_ in ["NOUN", "ADJ", "VERB"])
        palabras_respuesta = set(token.text.lower() for token in doc_respuesta 
                               if token.pos_ in ["NOUN", "ADJ", "VERB"])
        
        palabras_comunes = len(palabras_distractor & palabras_respuesta)
        if 1 <= palabras_comunes <= 3:
            puntuacion += 0.4
            
        return puntuacion
    
    def _modificar_oracion_semanticamente(self, oracion: str) -> str:
        """Modifica una oración manteniendo su estructura pero cambiando su significado."""
        doc = self.nlp(oracion)
        palabras = oracion.split()
        
        # Identificar y modificar verbos o adjetivos
        for token in doc:
            if token.pos_ in ['VERB', 'ADJ']:
                modificadores = {
                    'VERB': ['puede', 'debe', 'suele', 'tiende a'],
                    'ADJ': ['parcialmente', 'ocasionalmente', 'raramente', 'potencialmente']
                }
                modificador = random.choice(modificadores[token.pos_])
                palabras.insert(token.i, modificador)
        
        return ' '.join(palabras)

    def alterar_oracion(self, oracion: str) -> str:
        """Modifica una oración para crear una versión falsa."""
        palabras = oracion.split()
        if len(palabras) > 2:
            palabras[random.randint(0, len(palabras) - 1)] = "no"
        return " ".join(palabras)

    def generar_cuestionario(self, texto: str, materia: str, fuente: str, num_preguntas: int = 10) -> Dict[str, Any]:
        """Genera un cuestionario completo basado en un texto."""
        preguntas = self.generar_preguntas(texto, num_preguntas)
        return {
            "materia": materia,
            "fuente": fuente,
            "preguntas": preguntas
        }

    def _validar_pregunta(self, pregunta: Dict[str, Any]) -> bool:
        """Valida que una pregunta tenga el formato correcto y contenido válido"""
        try:
            # Verificar campos requeridos
            campos_requeridos = ['tipo', 'pregunta', 'respuesta_correcta', 'explicacion']
            if not all(campo in pregunta for campo in campos_requeridos):
                return False

            # Validar que los campos no estén vacíos
            if not all(pregunta[campo] for campo in campos_requeridos):
                return False

            # Validar tipo de pregunta
            if pregunta['tipo'] not in ['alternativas', 'verdadero_falso']:
                return False

            # Validar pregunta de alternativas
            if pregunta['tipo'] == 'alternativas':
                if 'opciones' not in pregunta:
                    return False
                if not isinstance(pregunta['opciones'], list):
                    return False
                if len(pregunta['opciones']) != 4:
                    return False
                if pregunta['respuesta_correcta'] not in pregunta['opciones']:
                    return False
                # Verificar que las opciones son únicas
                if len(set(pregunta['opciones'])) != 4:
                    return False

            # Validar pregunta de verdadero/falso
            if pregunta['tipo'] == 'verdadero_falso':
                if pregunta['respuesta_correcta'] not in ['Verdadero', 'Falso']:
                    return False

            # Validar longitud mínima de la pregunta y explicación
            if len(pregunta['pregunta'].split()) < 5:
                return False
            if len(pregunta['explicacion'].split()) < 10:
                return False

            return True

        except Exception:
            return False

    def procesar_json_entrada(self, ruta_json: str) -> List[Dict[str, Any]]:
        """Procesa el archivo quiz.json y genera cuestionarios para cada texto"""
        try:
            # Verificar que el archivo existe
            if not os.path.exists(ruta_json):
                raise FileNotFoundError(f"No se encontró el archivo: {ruta_json}")

            # Leer el archivo JSON
            with open(ruta_json, 'r', encoding='utf-8') as f:
                datos = json.load(f)

            # Verificar estructura del JSON
            if 'quiz' not in datos or not isinstance(datos['quiz'], list):
                raise ValueError("Formato de JSON inválido: debe contener una lista 'quiz'")

            cuestionarios = []
            for item in datos['quiz']:
                try:
                    # Verificar campos requeridos en cada item
                    if not all(campo in item for campo in ['texto', 'materia', 'fuente']):
                        print(f"Advertencia: item ignorado por falta de campos requeridos: {item}")
                        continue

                    # Generar cuestionario
                    cuestionario = self.generar_cuestionario(
                        texto=item['texto'],
                        materia=item['materia'],
                        fuente=item['fuente']
                    )
                    
                    # Verificar que el cuestionario tiene preguntas válidas
                    if cuestionario and cuestionario['preguntas']:
                        print(f"Cuestionario generado exitosamente para {item['materia']}")
                        print(f"Preguntas generadas: {len(cuestionario['preguntas'])}")
                        cuestionarios.append(cuestionario)
                    else:
                        print(f"No se pudieron generar preguntas para {item['materia']}")

                except Exception as e:
                    print(f"Error procesando item: {str(e)}")
                    continue

            if not cuestionarios:
                print("Advertencia: No se generaron cuestionarios")
            else:
                print(f"Total de cuestionarios generados: {len(cuestionarios)}")

            return cuestionarios

        except json.JSONDecodeError as e:
            print(f"Error decodificando JSON: {str(e)}")
            return []
        except Exception as e:
            print(f"Error procesando JSON de entrada: {str(e)}")
            return []

    def guardar_cuestionarios(self, cuestionarios: List[Dict[str, Any]], ruta_salida: str):
        """Guarda los cuestionarios generados en un archivo JSON"""
        try:
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)

            # Verificar que hay cuestionarios para guardar
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