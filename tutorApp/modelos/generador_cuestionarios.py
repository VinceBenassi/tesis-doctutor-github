# Generador de Cuestionarios
# Por Franco Benassi
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
        
        # Agregar diccionario de contrarios
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
            "fácil": "difícil"
        }
        
        # Inicializar modelos
        self.qa_model = pipeline("question-answering", model="dccuchile/bert-base-spanish-wwm-cased")
        self.nlp = spacy.load("es_core_news_sm")

    def _limpiar_oracion(self, oracion: str) -> str:
        """Limpia y normaliza una oración."""
        # Eliminar espacios extra
        oracion = re.sub(r'\s+', ' ', oracion.strip())
        
        # Eliminar caracteres no deseados
        oracion = re.sub(r'[^\w\s.,;:¿?¡!áéíóúñÁÉÍÓÚÑ]', '', oracion)
        
        # Asegurar que empiece con mayúscula
        if oracion and oracion[0].isalpha():
            oracion = oracion[0].upper() + oracion[1:]
            
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
        """Genera una explicación mejorada sin referencias directas al texto."""
        doc = self.nlp(texto)
        oracion_limpia = re.sub(r'\s+', ' ', oracion_base.strip())
        
        if es_verdadero:
            explicacion = f"Esta afirmación es correcta. {self._extraer_razon_principal(oracion_limpia)}"
            
            # Agregar contexto adicional
            contexto = self._extraer_contexto_relacionado(doc, oracion_limpia)
            if contexto:
                explicacion += f" Además, {self._generalizar_evidencia(contexto)}"
        else:
            version_correcta = self._obtener_version_correcta(oracion_limpia)
            explicacion = f"Esta afirmación es incorrecta. {version_correcta}"
        
        return explicacion
    
    def _extraer_razon_principal(self, oracion: str) -> str:
        """Extrae la razón principal de una afirmación."""
        doc = self.nlp(oracion)
        
        # Identificar el verbo principal y su objeto
        verbo_principal = None
        objeto = None
        
        for token in doc:
            if token.pos_ == 'VERB' and token.dep_ == 'ROOT':
                verbo_principal = token
            if token.dep_ in ['dobj', 'attr', 'iobj'] and not objeto:
                objeto = token
        
        if verbo_principal and objeto:
            return f"Esto se debe a que {verbo_principal.text} {objeto.text}"
        
        return "La evidencia respalda esta afirmación"

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
    
    def _invertir_significado_principal(self, oracion: str) -> str:
        """Invierte el significado principal de forma más natural."""
        doc = self.nlp(oracion)
        for token in doc:
            if token.pos_ == 'VERB' and token.dep_ == 'ROOT':
                # Buscar negaciones existentes
                tiene_negacion = any(t.dep_ == 'neg' for t in token.children)
                if tiene_negacion:
                    # Quitar negación
                    return re.sub(r'no\s+' + token.text, token.text, oracion)
                else:
                    # Agregar negación
                    return oracion.replace(token.text, f"no {token.text}")
        return None


    def _cambiar_cantidad(self, oracion: str) -> str:
        """Cambia las cantidades numéricas en una oración."""
        def modificar_numero(match):
            num = int(match.group())
            return str(num * 2 if num < 100 else num // 2)
        
        return re.sub(r'\d+', modificar_numero, oracion)

    def _calcular_similitud(self, texto1: str, texto2: str) -> float:
        """Calcula la similitud entre textos usando una métrica más robusta."""
        # Tokenizar y normalizar
        tokens1 = set(token.lower() for token in texto1.split() if token.lower() not in self.stopwords)
        tokens2 = set(token.lower() for token in texto2.split() if token.lower() not in self.stopwords)
        
        # Calcular coeficiente de Jaccard
        interseccion = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        return interseccion / union if union > 0 else 0.0

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
        """Extrae conceptos clave del texto"""
        # Buscar frases con mayúsculas iniciales y términos técnicos
        conceptos = re.findall(r'\b[A-Z][a-zA-Z\s]+\b|\b[a-zA-Z]+(?:\s+[a-zA-Z]+)*\b', texto)
        
        # Filtrar y limpiar conceptos
        conceptos = [c.strip() for c in conceptos if len(c.split()) <= 4 and len(c) > 3]
        
        # Eliminar duplicados y ordenar por longitud
        return sorted(list(set(conceptos)), key=len, reverse=True)[:10]

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
        """Genera una mezcla balanceada de preguntas."""
        analisis = self.analizar_texto(texto)
        preguntas = []
        preguntas_hash = set()  # Para evitar duplicados
        
        # Intentar generar igual número de cada tipo
        num_cada_tipo = num_preguntas // 2
        
        # Generar preguntas de alternativas
        intentos_alt = 0
        while len([p for p in preguntas if p['tipo'] == 'alternativas']) < num_cada_tipo and intentos_alt < num_cada_tipo * 3:
            pregunta = self.generar_pregunta_alternativas(analisis, texto)
            if pregunta and self._es_pregunta_valida_mejorada(pregunta):
                hash_pregunta = hash(pregunta['pregunta'])
                if hash_pregunta not in preguntas_hash:
                    preguntas.append(pregunta)
                    preguntas_hash.add(hash_pregunta)
            intentos_alt += 1
        
        # Generar preguntas V/F
        intentos_vf = 0
        while len([p for p in preguntas if p['tipo'] == 'verdadero_falso']) < num_cada_tipo and intentos_vf < num_cada_tipo * 3:
            pregunta = self.generar_pregunta_verdadero_falso(analisis, texto)
            if pregunta and self._es_pregunta_valida_mejorada(pregunta):
                hash_pregunta = hash(pregunta['pregunta'])
                if hash_pregunta not in preguntas_hash:
                    preguntas.append(pregunta)
                    preguntas_hash.add(hash_pregunta)
            intentos_vf += 1
        
        # Generar el tipo que falte si no se llegó al número deseado
        while len(preguntas) < num_preguntas:
            if random.random() < 0.5:
                tipo_faltante = "alternativas"
                generador = self.generar_pregunta_alternativas
            else:
                tipo_faltante = "verdadero_falso"
                generador = self.generar_pregunta_verdadero_falso
            
            pregunta = generador(analisis, texto)
            if pregunta and self._es_pregunta_valida_mejorada(pregunta):
                hash_pregunta = hash(pregunta['pregunta'])
                if hash_pregunta not in preguntas_hash:
                    preguntas.append(pregunta)
                    preguntas_hash.add(hash_pregunta)
        
        random.shuffle(preguntas)
        return preguntas[:num_preguntas]
    
    def _generar_hash_pregunta(self, pregunta: Dict[str, Any]) -> int:
        """Genera un hash más robusto para detectar duplicados."""
        elementos = [
            pregunta['pregunta'],
            pregunta['respuesta_correcta'],
            pregunta['tipo']
        ]
        if pregunta['tipo'] == 'alternativas':
            elementos.extend(pregunta['opciones'])
        
        return hash('||'.join(str(e) for e in elementos))
    
    def _seleccionar_mejores_preguntas(self, candidatas: List[Dict[str, Any]], 
                                  num_requeridas: int) -> List[Dict[str, Any]]:
        """Selecciona las mejores preguntas basado en criterios de calidad."""
        if not candidatas:
            return []
            
        # Calcular puntuación para cada pregunta
        puntuaciones = []
        for pregunta in candidatas:
            puntuacion = 0
            # Longitud adecuada
            palabras = len(pregunta['pregunta'].split())
            if 10 <= palabras <= 30:
                puntuacion += 2
            # Calidad de explicación
            if len(pregunta['explicacion'].split()) > 20:
                puntuacion += 2
            # Diversidad de opciones (para alternativas)
            if pregunta['tipo'] == 'alternativas':
                similitudes = []
                for i, opcion1 in enumerate(pregunta['opciones']):
                    for j, opcion2 in enumerate(pregunta['opciones'][i+1:], i+1):
                        similitud = self._calcular_similitud(opcion1, opcion2)
                        similitudes.append(similitud)
                if max(similitudes) < 0.7:  # Opciones suficientemente diferentes
                    puntuacion += 3
            
            puntuaciones.append((puntuacion, pregunta))
        
        # Ordenar por puntuación y seleccionar las mejores
        puntuaciones.sort(reverse=True)
        return [p[1] for p in puntuaciones[:num_requeridas]]

    def generar_pregunta_alternativas(self, analisis: Dict[str, Any], texto: str) -> Dict[str, Any]:
        """Genera preguntas de alternativas mejoradas."""
        if not analisis["conceptos"] or not analisis["oraciones"]:
            return None
            
        # Seleccionar concepto y contexto
        concepto = random.choice(analisis["conceptos"])
        contexto = self._extraer_contexto_concepto(concepto, texto)
        
        # Generar pregunta más natural
        pregunta = self._generar_pregunta_natural(concepto, contexto)
        
        # Generar opciones
        respuesta_correcta = self._extraer_respuesta_correcta(contexto, concepto)
        distractores = self._generar_distractores_mejorados(texto, concepto, respuesta_correcta, [])
        
        if not respuesta_correcta or len(distractores) < 3:
            return None
            
        opciones = [respuesta_correcta] + distractores[:3]
        random.shuffle(opciones)
        
        # Generar explicación sin referencias al texto
        explicacion = self._generar_explicacion_conceptual(concepto, respuesta_correcta)
        
        return {
            "tipo": "alternativas",
            "pregunta": pregunta,
            "opciones": opciones,
            "respuesta_correcta": respuesta_correcta,
            "explicacion": explicacion
        }
    
    def _extraer_respuesta_correcta(self, contexto: str, concepto: str) -> str:
        """Extrae una respuesta correcta del contexto sobre un concepto."""
        # Intentar extraer con QA
        try:
            respuesta = self.qa_model(
                question=f"¿Qué característica principal tiene {concepto}?",
                context=contexto
            )
            if respuesta['score'] > 0.5:
                return self._limpiar_oracion(respuesta['answer'])
        except Exception:
            pass
        
        # Si falla QA, extraer la oración más relevante
        doc = self.nlp(contexto)
        oraciones_relevantes = []
        for sent in doc.sents:
            if concepto.lower() in sent.text.lower():
                oraciones_relevantes.append(sent.text)
        
        if oraciones_relevantes:
            return self._limpiar_oracion(max(oraciones_relevantes, key=len))
        
        return None

    def generar_pregunta_verdadero_falso(self, analisis: Dict[str, Any], texto: str) -> Dict[str, Any]:
        """Genera una pregunta de verdadero/falso mejorada."""
        if not analisis["oraciones"]:
            return None
        
        # Filtrar oraciones válidas
        oraciones_validas = [
            o for o in analisis["oraciones"]
            if len(o.split()) >= 10
            and not any(p in o.lower() for p in ['?', '¿', '!', '¡'])
            and self._es_oracion_valida(o)
        ]
        
        if not oraciones_validas:
            return None
        
        # Seleccionar una oración al azar
        oracion_base = random.choice(oraciones_validas)
        es_verdadero = random.choice([True, False])
        
        if es_verdadero:
            pregunta = self._limpiar_oracion(oracion_base)
        else:
            # Generar una versión falsa de la oración
            pregunta = self._generar_version_falsa_mejorada(oracion_base)
            if not pregunta:
                return None
        
        # Generar explicación contextualizada
        if es_verdadero:
            explicacion = f"Esta afirmación es correcta porque {oracion_base.lower()}"
        else:
            explicacion = f"Esta afirmación es incorrecta. Lo correcto es que {oracion_base.lower()}"
        
        return {
            "tipo": "verdadero_falso",
            "pregunta": pregunta,
            "respuesta_correcta": "Verdadero" if es_verdadero else "Falso",
            "explicacion": explicacion
        }
    
    def _generar_pregunta_natural(self, concepto: str, contexto: str) -> str:
        """Genera una pregunta más natural sobre un concepto."""
        plantillas = [
            f"¿Cuál es la principal característica de {concepto}?",
            f"¿Qué aspecto es fundamental para entender {concepto}?",
            f"¿Qué caracteriza mejor a {concepto}?",
            f"¿Cuál es el papel principal de {concepto}?",
            f"¿Qué define mejor a {concepto}?"
        ]
        return random.choice(plantillas)

    def _extraer_contexto_concepto(self, concepto: str, texto: str) -> str:
        """Extrae el contexto relevante para un concepto sin depender del texto original."""
        doc = self.nlp(texto)
        contexto_relevante = []
        
        for sent in doc.sents:
            if concepto.lower() in sent.text.lower():
                contexto_relevante.append(sent.text)
        
        return " ".join(contexto_relevante[:2])

    def _generar_explicacion_conceptual(self, concepto: str, respuesta: str) -> str:
        """Genera una explicación basada en conceptos sin referencias al texto."""
        return f"Esta respuesta es correcta porque representa la característica fundamental de {concepto}. " \
            f"{respuesta.capitalize()} es un aspecto esencial que define cómo funciona y se aplica {concepto}."

    def _identificar_dominio(self, concepto: str) -> str:
        """Identifica el dominio del concepto basado en palabras clave."""
        dominios = {
            "programacion": ["algoritmo", "código", "programación", "desarrollo", "software"],
            "medicina": ["tratamiento", "diagnóstico", "paciente", "clínico", "médico"],
            "pedagogia": ["aprendizaje", "enseñanza", "educación", "didáctica", "pedagógico"]
        }
        
        for dominio, palabras_clave in dominios.items():
            if any(palabra in concepto.lower() for palabra in palabras_clave):
                return dominio
        return "general"
    
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
    
    def _modificar_actor(self, oracion: str) -> str:
        """Modifica el actor principal de la oración."""
        doc = self.nlp(oracion)
        for token in doc:
            if token.dep_ == 'nsubj':
                # Identificar el tipo de actor
                if token.text.lower() in ['el', 'la', 'los', 'las']:
                    continue
                    
                actores_alternativos = {
                    'sistema': ['proceso', 'método', 'mecanismo'],
                    'persona': ['individuo', 'sujeto', 'participante'],
                    'grupo': ['equipo', 'conjunto', 'colectivo'],
                    'profesional': ['especialista', 'experto', 'técnico']
                }
                
                # Encontrar categoría más apropiada
                for categoria, alternativas in actores_alternativos.items():
                    if any(palabra in token.text.lower() for palabra in [categoria] + alternativas):
                        return oracion.replace(token.text, random.choice(alternativas))
        
        return None
    
    def _cambiar_accion(self, oracion: str) -> str:
        """Cambia la acción principal por una alternativa."""
        doc = self.nlp(oracion)
        for token in doc:
            if token.pos_ == 'VERB' and token.dep_ == 'ROOT':
                acciones_alternativas = {
                    'aumentar': ['reducir', 'disminuir', 'decrecer'],
                    'mejorar': ['empeorar', 'degradar', 'deteriorar'],
                    'facilitar': ['dificultar', 'complicar', 'obstaculizar'],
                    'permitir': ['impedir', 'prohibir', 'restringir'],
                    'incluir': ['excluir', 'omitir', 'descartar']
                }
                
                for base, alternativas in acciones_alternativas.items():
                    if token.lemma_.lower() == base:
                        return oracion.replace(token.text, random.choice(alternativas))
        
        return None

    def _generar_version_falsa_mejorada(self, oracion: str) -> str:
        """Genera una versión falsa de manera más significativa."""
        doc = self.nlp(oracion)
        estrategias = [
            self._invertir_significado_principal,
            self._cambiar_cuantificadores,
            self._modificar_actor,
            self._cambiar_accion
        ]
        
        modificaciones = []
        for estrategia in estrategias:
            try:
                modificacion = estrategia(oracion)
                if modificacion and modificacion != oracion:
                    modificaciones.append(modificacion)
            except Exception:
                continue
        
        if modificaciones:
            return random.choice(modificaciones)
            
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
    
    def _es_pregunta_valida_mejorada(self, pregunta: Dict[str, Any]) -> bool:
        """Validación mejorada de preguntas."""
        if not pregunta or not isinstance(pregunta, dict):
            return False
            
        # Verificar campos requeridos
        campos_requeridos = ['tipo', 'pregunta', 'respuesta_correcta', 'explicacion']
        if not all(campo in pregunta for campo in campos_requeridos):
            return False
        
        # Validar longitud de la pregunta
        if len(pregunta['pregunta'].split()) < 8:
            return False
        
        # Validar tipo específico
        if pregunta['tipo'] == 'alternativas':
            if 'opciones' not in pregunta or len(pregunta['opciones']) != 4:
                return False
            if pregunta['respuesta_correcta'] not in pregunta['opciones']:
                return False
            
            # Verificar que las opciones son suficientemente diferentes
            for i, opcion1 in enumerate(pregunta['opciones']):
                for opcion2 in pregunta['opciones'][i+1:]:
                    if self._calcular_similitud(opcion1, opcion2) > 0.8:
                        return False
        
        elif pregunta['tipo'] == 'verdadero_falso':
            if pregunta['respuesta_correcta'] not in ['Verdadero', 'Falso']:
                return False
        
        else:
            return False
        
        # Validar explicación
        if len(pregunta['explicacion'].split()) < 10:
            return False
        
        return True


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

    def _generar_explicacion_detallada(self, concepto: str, respuesta: str, oracion: str, texto: str) -> str:
        """Genera una explicación más detallada y contextualizada sin referencias al texto."""
        explicacion = f"La respuesta es correcta porque {respuesta.lower()}. "
        
        # Agregar contexto adicional sin referencias al texto
        doc = self.nlp(texto)
        evidencias = []
        for sent in doc.sents:
            if concepto.lower() in sent.text.lower() and sent.text != oracion:
                similitud = self._calcular_similitud(sent.text, respuesta)
                if similitud > 0.3:
                    evidencias.append(self._generalizar_evidencia(sent.text))
        
        if evidencias:
            explicacion += f"Esto se fundamenta en que {evidencias[0].lower()}"
            if len(evidencias) > 1:
                explicacion += f". Adicionalmente, {evidencias[1].lower()}"
        
        return explicacion.strip()
    
    def _generalizar_evidencia(self, evidencia: str) -> str:
        """Convierte una evidencia específica en una declaración más general."""
        # Eliminar referencias específicas
        evidencia = re.sub(r'como se menciona|según el texto|como dice|como se indica', '', evidencia)
        # Convertir a presente simple
        evidencia = re.sub(r'mencionó|indicó|mostró', 'muestra', evidencia)
        return evidencia.strip()
    
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
    
    def _invertir_afirmacion(self, oracion: str) -> str:
        """Invierte el significado de una afirmación."""
        # Negaciones simples
        if not oracion:
            return None
            
        # Patrones comunes de negación
        patrones_negacion = [
            (r'\bes\b', 'no es'),
            (r'\bson\b', 'no son'),
            (r'\bhay\b', 'no hay'),
            (r'\btiene\b', 'no tiene'),
            (r'\bpuede\b', 'no puede'),
            (r'\bsiempre\b', 'nunca'),
            (r'\btodos\b', 'ninguno'),
            (r'\bnadie\b', 'todos'),
            (r'\bnunca\b', 'siempre'),
        ]
        
        # Intentar cada patrón de negación
        for patron, reemplazo in patrones_negacion:
            if re.search(patron, oracion.lower()):
                return re.sub(patron, reemplazo, oracion, flags=re.IGNORECASE)
                
        # Si no se encuentra un patrón específico, agregar "no" al principio
        # de la primera cláusula verbal
        doc = self.nlp(oracion)
        for token in doc:
            if token.pos_ == 'VERB':
                index = token.i
                partes = list(token.text for token in doc)
                partes.insert(index, "no")
                return " ".join(partes)
                
        return oracion

    
    def _generar_distractores_mejorados(self, texto: str, concepto: str, 
                                respuesta_correcta: str, oraciones_relacionadas: List[str]) -> List[str]:
        """Genera distractores más naturales y variados."""
        distractores = set()
        
        # 1. Modificación semántica
        modificaciones = [
            (self._invertir_significado, "opuesto"),
            (self._modificar_alcance, "parcial"),
            (self._cambiar_contexto, "diferente contexto"),
            (self._agregar_condicion, "condicional")
        ]
        
        for modificar, tipo in modificaciones:
            distractor = modificar(respuesta_correcta)
            if distractor and self._es_distractor_valido(distractor, respuesta_correcta):
                distractores.add(distractor)
        
        # 2. Uso de conocimiento del dominio
        patrones_dominio = self._obtener_patrones_dominio(concepto)
        for patron in patrones_dominio:
            distractor = patron.format(concepto=concepto)
            if self._es_distractor_valido(distractor, respuesta_correcta):
                distractores.add(distractor)
        
        # 3. Generación basada en contexto
        for oracion in oraciones_relacionadas:
            if oracion != respuesta_correcta:
                version_modificada = self._modificar_oracion_semanticamente(oracion)
                if version_modificada and self._es_distractor_valido(version_modificada, respuesta_correcta):
                    distractores.add(version_modificada)
        
        distractores = list(distractores)[:3]
        return distractores
    
    def _modificar_alcance(self, oracion: str) -> str:
        """Modifica el alcance o amplitud de una afirmación."""
        modificadores = {
            'todos': 'algunos',
            'siempre': 'a veces',
            'nunca': 'raramente',
            'completamente': 'parcialmente',
            'absolutamente': 'relativamente',
            'totalmente': 'parcialmente',
            'globalmente': 'localmente'
        }
        
        for palabra, modificador in modificadores.items():
            if palabra in oracion.lower():
                return oracion.replace(palabra, modificador)
        
        # Si no encuentra palabras específicas, agregar un modificador de alcance
        doc = self.nlp(oracion)
        for token in doc:
            if token.pos_ in ['VERB', 'ADJ']:
                return oracion.replace(token.text, f"parcialmente {token.text}")
        
        return None
    
    def _cambiar_contexto(self, oracion: str) -> str:
        """Cambia el contexto de aplicación de una afirmación."""
        # Agregar limitadores de contexto
        limitadores = [
            "solo en ambientes controlados",
            "únicamente en casos específicos",
            "exclusivamente en ciertos contextos",
            "bajo ciertas condiciones",
            "en situaciones particulares"
        ]
        
        if oracion.endswith('.'):
            oracion = oracion[:-1]
        
        return f"{oracion} {random.choice(limitadores)}."

    def _agregar_condicion(self, oracion: str) -> str:
        """Agrega una condición que limita la validez de la afirmación."""
        condiciones = [
            "siempre y cuando se cumplan los requisitos necesarios",
            "cuando se dispone de los recursos adecuados",
            "si se siguen los procedimientos correctos",
            "dependiendo de las circunstancias específicas",
            "sujeto a validación caso por caso"
        ]
        
        if oracion.endswith('.'):
            oracion = oracion[:-1]
        
        return f"{oracion} {random.choice(condiciones)}."

    def _cambiar_cuantificador(self, oracion: str) -> str:
        """Cambia los cuantificadores en una oración."""
        cuantificadores = {
            'todos': 'pocos',
            'muchos': 'algunos',
            'siempre': 'ocasionalmente',
            'nunca': 'raramente',
            'cada': 'algún',
            'cualquier': 'cierto',
            'ningún': 'algún',
            'todo': 'parte'
        }
        
        for cuant, reemplazo in cuantificadores.items():
            if cuant in oracion.lower():
                return oracion.replace(cuant, reemplazo)
        
        return None
    
    def _cambiar_cuantificadores(self, oracion: str) -> str:
        """Cambia los cuantificadores de forma más sofisticada."""
        cambios = {
            r'\btodos\b': 'pocos',
            r'\bsiempre\b': 'raramente',
            r'\bnunca\b': 'frecuentemente',
            r'\bcada\b': 'algunos',
            r'\bmucho[s]?\b': 'poco',
            r'\bpoco[s]?\b': 'mucho',
            r'\bla mayoría\b': 'la minoría',
            r'\bcompletamente\b': 'parcialmente'
        }
        
        oracion_modificada = oracion
        for patron, reemplazo in cambios.items():
            if re.search(patron, oracion_modificada, re.IGNORECASE):
                oracion_modificada = re.sub(patron, reemplazo, oracion_modificada, flags=re.IGNORECASE)
                return oracion_modificada
        
        return None

    def _modificar_tiempo(self, oracion: str) -> str:
        """Cambia el tiempo verbal de las acciones en la oración."""
        # Mapeo de tiempos verbales comunes
        tiempos = {
            'es': 'será',
            'son': 'serán',
            'está': 'estará',
            'están': 'estarán',
            'ha': 'habrá',
            'han': 'habrán',
            'puede': 'podrá',
            'pueden': 'podrán'
        }
        
        for tiempo, futuro in tiempos.items():
            if f" {tiempo} " in oracion:
                return oracion.replace(f" {tiempo} ", f" {futuro} ")
        
        return None

    def _cambiar_agente(self, oracion: str) -> str:
        """Cambia el agente o sujeto principal de la oración."""
        doc = self.nlp(oracion)
        for token in doc:
            if token.dep_ == 'nsubj':
                # Intentar generalizar o especificar el sujeto
                if token.text.lower() in ['el', 'la', 'los', 'las']:
                    continue
                agentes_alternativos = [
                    'algunos expertos',
                    'ciertos especialistas',
                    'diversos estudios',
                    'investigaciones recientes',
                    'análisis preliminares'
                ]
                return oracion.replace(token.text, random.choice(agentes_alternativos))
        return None
    
    def _obtener_patrones_dominio(self, concepto: str) -> List[str]:
        """Obtiene patrones de respuesta específicos del dominio."""
        patrones_generales = {
            "programacion": [
                "La {concepto} es una técnica obsoleta que ha sido reemplazada",
                "La {concepto} solo funciona en casos muy específicos",
                "La {concepto} requiere recursos computacionales excesivos"
            ],
            "medicina": [
                "El {concepto} solo se aplica en casos de emergencia",
                "El {concepto} no tiene efectos significativos en el tratamiento",
                "El {concepto} presenta más riesgos que beneficios"
            ],
            # Agregar más dominios según sea necesario
        }
        
        # Determinar el dominio basado en el concepto
        dominio = self._identificar_dominio(concepto)
        return patrones_generales.get(dominio, [])

    def _generar_distractor_generico(self, concepto: str, respuesta_correcta: str) -> str:
        """Genera un distractor genérico cuando otras estrategias fallan."""
        plantillas = [
            f"El {concepto} no tiene relación con este tema",
            f"El {concepto} funciona de manera opuesta",
            f"No existe evidencia sobre {concepto}",
            f"Los estudios no han demostrado efectos de {concepto}"
        ]
        return random.choice(plantillas)
    
    def _generar_contradiccion(self, oracion: str) -> str:
        """Genera una contradicción lógica de la oración."""
        doc = self.nlp(oracion)
        
        # Intentar diferentes estrategias de contradicción
        estrategias = [
            self._invertir_afirmacion,
            self._cambiar_cuantificador,
            self._modificar_tiempo,
            self._cambiar_agente
        ]
        
        for estrategia in estrategias:
            resultado = estrategia(oracion)
            if resultado and resultado != oracion:
                return resultado
                
        return None
    
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
        """Valida que una pregunta tenga el formato correcto"""
        try:
            # Verificar campos requeridos
            campos_requeridos = ['tipo', 'pregunta', 'respuesta_correcta', 'explicacion']
            if not all(campo in pregunta for campo in campos_requeridos):
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

            # Validar pregunta de verdadero/falso
            if pregunta['tipo'] == 'verdadero_falso':
                if pregunta['respuesta_correcta'] not in ['Verdadero', 'Falso']:
                    return False

            return True

        except Exception:
            return False
        
    def _validar_pregunta_mejorada(self, pregunta: Dict[str, Any]) -> bool:
        """Validación mejorada de preguntas."""
        try:
            # Validaciones básicas
            if not self._validar_pregunta(pregunta):
                return False

            # Validación de longitud y contenido
            if len(pregunta['pregunta'].split()) < 8:
                return False
                
            if pregunta['tipo'] == 'alternativas':
                # Verificar que las opciones son suficientemente diferentes
                opciones = pregunta['opciones']
                for i, opcion1 in enumerate(opciones):
                    for opcion2 in opciones[i+1:]:
                        if self._calcular_similitud(opcion1, opcion2) > 0.8:
                            return False
                            
                # Verificar que la respuesta correcta es válida
                if pregunta['respuesta_correcta'] not in opciones:
                    return False
                    
            # Verificar que la explicación es relevante
            if self._calcular_similitud(pregunta['pregunta'], pregunta['explicacion']) < 0.2:
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
                    cuestionarios.append(cuestionario)

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