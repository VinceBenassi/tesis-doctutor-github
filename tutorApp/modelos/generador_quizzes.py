# Generador de Quizzes
# por Franco Benassi
from transformers import pipeline, AutoTokenizer
import spacy
import json
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import random
import numpy as np
import re

class GeneradorQA:
    def __init__(self):
        print("Inicializando modelos...")
        try:
            # Pipeline QA principal
            self.qa_model = pipeline(
                "question-answering",
                model="mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"
            )
            
            # SpaCy para análisis semántico
            self.nlp = spacy.load("es_core_news_lg")
            
            # Tokenizer para procesamiento
            self.tokenizer = AutoTokenizer.from_pretrained(
                "mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"
            )
            
            print("Modelos cargados correctamente")
            
        except Exception as e:
            print(f"Error inicializando modelos: {str(e)}")
            raise

    def _extraer_conceptos_clave(self, texto: str) -> List[Dict]:
        """Extrae conceptos clave y sus relaciones de forma dinámica"""
        doc = self.nlp(texto)
        conceptos = []
        
        # Analizar cada oración para encontrar conceptos significativos
        for sent in doc.sents:
            # Identificar términos principales usando análisis de dependencias
            for token in sent:
                if self._es_termino_significativo(token):
                    # Obtener el contexto semántico del término
                    contexto = self._obtener_contexto_semantico(token, sent)
                    if contexto:
                        concepto = {
                            'texto': token.text,
                            'contexto': contexto,
                            'relaciones': self._analizar_relaciones(token, sent)
                        }
                        conceptos.append(concepto)

            # Analizar grupos nominales para conceptos compuestos
            for chunk in sent.noun_chunks:
                if self._es_concepto_compuesto_valido(chunk):
                    contexto = self._obtener_contexto_semantico(chunk.root, sent)
                    if contexto:
                        concepto = {
                            'texto': chunk.text,
                            'contexto': contexto,
                            'relaciones': self._analizar_relaciones(chunk.root, sent)
                        }
                        conceptos.append(concepto)
        
        return conceptos

    def _es_termino_significativo(self, token) -> bool:
        """Determina si un término es significativo para generar preguntas"""
        # Verificar características lingüísticas
        es_significativo = (
            not token.is_stop and
            not token.is_punct and
            token.pos_ in ['NOUN', 'PROPN', 'VERB'] and
            len(token.text) > 2
        )
        
        if es_significativo:
            # Verificar importancia semántica
            doc = self.nlp(token.sent.text)
            importancia = sum(token.similarity(t) for t in doc if t != token) / len(doc)
            return importancia > 0.3
            
        return False

    def _es_concepto_compuesto_valido(self, chunk) -> bool:
        """Valida si un grupo nominal es un concepto válido para preguntas"""
        # Verificar estructura y longitud
        if len(chunk) < 2:
            return False
            
        # Verificar que no sea una frase común o sin significado
        doc = self.nlp(chunk.text)
        tiene_contenido = any(not t.is_stop for t in doc)
        tiene_estructura = any(t.pos_ in ['NOUN', 'PROPN'] for t in doc)
        
        return tiene_contenido and tiene_estructura

    def _analizar_relaciones(self, token, sent) -> Dict:
        """Analiza las relaciones semánticas y sintácticas de un término"""
        relaciones = {
            'acciones': [],
            'propiedades': [],
            'asociaciones': []
        }
        
        for t in sent:
            if t.head == token or token in [c for c in t.children]:
                if t.pos_ == 'VERB' and not t.is_stop:
                    relaciones['acciones'].append(t.text)
                elif t.pos_ == 'ADJ':
                    relaciones['propiedades'].append(t.text)
                elif t.pos_ in ['NOUN', 'PROPN'] and t != token:
                    relaciones['asociaciones'].append(t.text)
                    
        return relaciones

    def _obtener_contexto_semantico(self, chunk, sent) -> Optional[str]:
        """Extrae el contexto semántico relevante de una frase"""
        contexto = []
        
        # Analizar dependencias sintácticas
        for token in sent:
            if token.head == chunk.root or chunk.root in [child for child in token.children]:
                if not token.is_stop and token.pos_ in ['VERB', 'NOUN', 'ADJ']:
                    contexto.append(token.text)

        return ' '.join(contexto) if contexto else None

    def _generar_pregunta_analitica(self, concepto: Dict) -> Optional[Dict]:
        """Genera una pregunta analítica basada en el análisis semántico"""
        try:
            # Analizar el concepto para determinar el tipo de pregunta más apropiado
            tipo_pregunta = self._determinar_tipo_pregunta_dinamico(concepto)
            if not tipo_pregunta:
                return None

            # Generar la pregunta
            pregunta = self._construir_pregunta_dinamica(concepto, tipo_pregunta)
            if not pregunta:
                return None

            # Generar respuesta y alternativas
            respuesta, alternativas = self._generar_opciones_semanticas(concepto, tipo_pregunta)
            if not respuesta or len(alternativas) < 3:
                return None

            # Asegurar coherencia y variedad
            alternativas_finales = self._ajustar_alternativas(alternativas, respuesta)
            
            return {
                'pregunta': pregunta,
                'opciones': alternativas_finales,
                'respuesta_correcta': respuesta,
                'tipo': 'alternativas'
            }

        except Exception as e:
            print(f"Error en generación de pregunta: {str(e)}")
            return None

    def _determinar_tipo_pregunta_dinamico(self, concepto: Dict) -> Optional[str]:
        """Determina el tipo de pregunta basado en el análisis del concepto"""
        relaciones = concepto['relaciones']
        
        if relaciones['acciones']:
            return 'FUNCION'
        elif relaciones['propiedades']:
            return 'CARACTERISTICA'
        elif relaciones['asociaciones']:
            return 'RELACION'
        
        # Análisis semántico para determinar el tipo
        doc = self.nlp(concepto['texto'])
        vector_concepto = doc.vector
        
        # Calcular similitud con diferentes tipos de preguntas
        similitudes = {
            'PROCESO': self.nlp("cómo funciona método procedimiento").similarity(doc),
            'PROPÓSITO': self.nlp("para qué sirve objetivo finalidad").similarity(doc),
            'COMPARACION': self.nlp("diferencia distinción contraste").similarity(doc)
        }
        
        tipo = max(similitudes.items(), key=lambda x: x[1])
        return tipo[0] if tipo[1] > 0.3 else None

    def _construir_pregunta_dinamica(self, concepto: Dict, tipo: str) -> Optional[str]:
        """Construye una pregunta de forma dinámica según el análisis"""
        texto_concepto = concepto['texto'].lower()
        
        # Generar estructura de pregunta basada en análisis lingüístico
        doc = self.nlp(texto_concepto)
        if doc[0].pos_ in ['NOUN', 'PROPN']:
            texto_concepto = f"el {texto_concepto}" if not texto_concepto.startswith('el ') else texto_concepto

        # Construir pregunta según el tipo y contexto
        patrones_base = {
            'FUNCION': [
                "¿Cuál es el principal propósito de",
                "¿Para qué se utiliza",
                "¿Qué función cumple"
            ],
            'CARACTERISTICA': [
                "¿Qué característica define a",
                "¿Qué distingue a",
                "¿Cuál es el rasgo principal de"
            ],
            'RELACION': [
                "¿Cómo se relaciona",
                "¿Qué conexión tiene",
                "¿Qué vincula a"
            ],
            'PROCESO': [
                "¿Cómo funciona",
                "¿Cuál es el mecanismo de",
                "¿De qué manera opera"
            ],
            'PROPÓSITO': [
                "¿Cuál es la finalidad de",
                "¿Qué busca lograr",
                "¿Cuál es el objetivo de"
            ],
            'COMPARACION': [
                "¿Qué diferencia a",
                "¿Cómo se distingue",
                "¿Qué hace único a"
            ]
        }
        
        patron = random.choice(patrones_base[tipo])
        return f"{patron} {texto_concepto}?"

    def _generar_opciones_semanticas(self, concepto: Dict, tipo: str) -> Tuple[Optional[str], List[str]]:
        """Genera opciones semánticamente coherentes"""
        # Obtener la respuesta base de las relaciones del concepto
        respuesta = self._obtener_respuesta_base(concepto, tipo)
        if not respuesta:
            return None, []

        # Generar alternativas mediante análisis semántico
        alternativas = set()
        doc_respuesta = self.nlp(respuesta)
        
        # Analizar el contexto para generar alternativas relacionadas
        doc_contexto = self.nlp(concepto['contexto'])
        
        for token in doc_contexto:
            if self._es_termino_significativo(token):
                # Generar alternativa basada en similitud semántica
                alt = self._generar_alternativa_semantica(token, doc_respuesta, tipo)
                if alt and alt != respuesta:
                    alternativas.add(alt)

        # Asegurar suficientes alternativas únicas y coherentes
        alternativas = list(alternativas)[:3]
        if len(alternativas) < 3:
            # Generar alternativas adicionales si es necesario
            extras = self._generar_alternativas_adicionales(respuesta, tipo)
            alternativas.extend(extras)
            alternativas = list(set(alternativas))[:3]

        return respuesta, alternativas

    def _obtener_respuesta_base(self, concepto: Dict, tipo: str) -> Optional[str]:
        """Obtiene una respuesta base según el tipo de pregunta"""
        relaciones = concepto['relaciones']
        
        if tipo == 'FUNCION' and relaciones['acciones']:
            return f"{random.choice(relaciones['acciones'])} {concepto['texto']}"
        elif tipo == 'CARACTERISTICA' and relaciones['propiedades']:
            return f"{random.choice(relaciones['propiedades'])} {concepto['texto']}"
        elif tipo == 'RELACION' and relaciones['asociaciones']:
            return f"Se relaciona con {random.choice(relaciones['asociaciones'])}"
        
        # Generar respuesta basada en análisis semántico si no hay relaciones directas
        doc = self.nlp(concepto['contexto'])
        vectores_relevantes = [t for t in doc if self._es_termino_significativo(t)]
        if vectores_relevantes:
            return random.choice(vectores_relevantes).text
        
        return None

    def _generar_alternativa_semantica(self, token, doc_respuesta, tipo: str) -> Optional[str]:
        """Genera una alternativa semánticamente coherente"""
        if not token.has_vector or not doc_respuesta.has_vector:
            return None
            
        similitud = token.similarity(doc_respuesta)
        
        # Generar alternativa si la similitud está en un rango apropiado
        if 0.3 <= similitud <= 0.7:
            if tipo == 'FUNCION':
                return f"{token.text} el elemento"
            elif tipo == 'CARACTERISTICA':
                return f"Es {token.text}"
            elif tipo == 'RELACION':
                return f"Se relaciona mediante {token.text}"
                
        return None

    def _generar_alternativas_adicionales(self, respuesta: str, tipo: str) -> List[str]:
        """Genera alternativas adicionales cuando es necesario"""
        alternativas = []
        doc_respuesta = self.nlp(respuesta)
        
        # Buscar términos semánticamente relacionados
        for token in self.nlp.vocab:
            if len(alternativas) >= 3:
                break
                
            if token.has_vector and token.text.lower() != respuesta.lower():
                similitud = token.similarity(doc_respuesta)
                if 0.3 <= similitud <= 0.7:
                    alternativa = self._formatear_alternativa(token.text, tipo)
                    if alternativa:
                        alternativas.append(alternativa)
                        
        return alternativas

    def _formatear_alternativa(self, texto: str, tipo: str) -> Optional[str]:
        """Formatea una alternativa según el tipo de pregunta"""
        if not texto or len(texto.split()) < 2:
            return None
            
        doc = self.nlp(texto)
        if not any(self._es_termino_significativo(t) for t in doc):
            return None
            
        return texto

    def _determinar_tipo_analisis(self, concepto: Dict) -> Optional[str]:
        """Determina el tipo de análisis más apropiado para el concepto"""
        if 'accion' in concepto:
            # Análisis de función o propósito
            return 'funcion'
        elif 'tema' in concepto:
            # Análisis de características o propiedades
            return 'caracteristica'
        elif concepto['tipo'] in ['ORG', 'PRODUCT']:
            # Análisis de rol o impacto
            return 'impacto'
        return None

    def _construir_pregunta(self, concepto: Dict, tipo_analisis: str) -> Tuple[Optional[str], Optional[str]]:
        """Construye una pregunta analítica coherente"""
        doc = self.nlp(concepto['contexto'])
        
        # Asegurar que el concepto está en el caso gramatical correcto
        concepto_texto = concepto['texto']
        if concepto_texto.lower() != concepto_texto:
            concepto_texto = f"el {concepto_texto}"

        if tipo_analisis == 'funcion':
            # Analizar el propósito o función
            verbo = concepto.get('accion', '')
            if verbo:
                return (
                    f"¿Cuál es la función principal de {concepto_texto} en relación con {verbo}?",
                    concepto['contexto']
                )
        
        elif tipo_analisis == 'caracteristica':
            # Analizar características distintivas
            tema = concepto.get('tema', '')
            if tema:
                return (
                    f"¿Qué característica distintiva presenta {concepto_texto} en el contexto de {tema}?",
                    concepto['contexto']
                )
        
        elif tipo_analisis == 'impacto':
            # Analizar el impacto o importancia
            return (
                f"¿Cuál es el impacto más significativo de {concepto_texto} según el texto?",
                concepto['contexto']
            )

        return None, None

    def _generar_alternativas_semanticas(self, respuesta: str, concepto: Dict, tipo_analisis: str) -> List[str]:
        """Genera alternativas semánticamente coherentes"""
        doc_respuesta = self.nlp(respuesta)
        doc_contexto = self.nlp(concepto['contexto'])
        alternativas = set()

        # Identificar elementos semánticos clave en la respuesta
        elementos_clave = [token for token in doc_respuesta if not token.is_stop and token.pos_ in ['NOUN', 'VERB', 'ADJ']]

        # Generar alternativas basadas en similitud semántica
        for sent in doc_contexto.sents:
            for token in sent:
                if token.pos_ in ['NOUN', 'VERB', 'ADJ'] and not token.is_stop:
                    # Verificar relevancia semántica
                    if any(elem.similarity(token) > 0.5 for elem in elementos_clave):
                        alternativa = self._construir_alternativa_coherente(token, sent, tipo_analisis)
                        if alternativa and alternativa != respuesta:
                            alternativas.add(alternativa)

        # Asegurar alternativas distintas y coherentes
        alternativas_filtradas = []
        for alt in alternativas:
            if self._es_alternativa_valida(alt, respuesta, alternativas_filtradas):
                alternativas_filtradas.append(alt)

        return list(alternativas_filtradas)[:3]

    def _construir_alternativa_coherente(self, token, sent, tipo_analisis: str) -> Optional[str]:
        """Construye una alternativa coherente basada en el contexto"""
        if tipo_analisis == 'funcion':
            # Buscar verbos y objetos relacionados
            for tok in sent:
                if tok.dep_ == "dobj" and tok.head == token:
                    return f"{token.text} {tok.text}"
        
        elif tipo_analisis == 'caracteristica':
            # Buscar adjetivos y sustantivos relacionados
            for tok in sent:
                if tok.dep_ == "amod" and tok.head == token:
                    return f"{tok.text} {token.text}"
        
        elif tipo_analisis == 'impacto':
            # Buscar efectos o consecuencias
            for tok in sent:
                if tok.dep_ in ["prep", "agent"] and tok.head == token:
                    return f"{token.text} {tok.text} {next(tok.children).text}"

        return None

    def _ajustar_alternativas(self, alternativas: List[str], respuesta: str) -> List[str]:
        """Asegura que la respuesta correcta esté entre las alternativas y mantiene coherencia"""
        if respuesta not in alternativas:
            alternativas = alternativas[:3] + [respuesta]
        random.shuffle(alternativas)
        return alternativas

    def generar_preguntas(self, texto: str) -> List[Dict]:
        """Genera preguntas analíticas a partir del texto"""
        conceptos = self._extraer_conceptos_clave(texto)
        if not conceptos:
            return []

        preguntas = []
        for concepto in conceptos:
            pregunta = self._generar_pregunta_analitica(concepto)
            if pregunta and self._es_pregunta_valida(pregunta, preguntas):
                preguntas.append(pregunta)

        return preguntas[:10]  # Limitar a 10 preguntas

    def _es_pregunta_valida(self, nueva_pregunta: Dict, preguntas: List[Dict]) -> bool:
        """Valida que la pregunta sea única y tenga sentido"""
        # Verificar coherencia gramatical
        if not self._validar_gramatica_pregunta(nueva_pregunta['pregunta']):
            return False

        # Evitar preguntas duplicadas o muy similares
        for pregunta in preguntas:
            if self.nlp(pregunta['pregunta']).similarity(self.nlp(nueva_pregunta['pregunta'])) > 0.7:
                return False

        return True

    def _validar_gramatica_pregunta(self, pregunta: str) -> bool:
        """Valida la coherencia gramatical de la pregunta"""
        doc = self.nlp(pregunta)
        
        # Verificar estructura básica de pregunta
        tiene_pronombre = False
        tiene_verbo = False
        tiene_complemento = False
        
        for token in doc:
            if token.pos_ == "PRON" and token.text.lower() in ["qué", "cuál", "cómo", "dónde", "quién", "cuándo"]:
                tiene_pronombre = True
            elif token.pos_ == "VERB":
                tiene_verbo = True
            elif token.dep_ in ["obj", "obl"]:
                tiene_complemento = True

        return tiene_pronombre and tiene_verbo and tiene_complemento

    def _extraer_hechos(self, texto: str) -> List[Dict]:
        """Extrae hechos relevantes del texto usando análisis semántico"""
        doc = self.nlp(texto)
        hechos = []
        
        # Analizar cada oración
        for sent in doc.sents:
            # Verificar que la oración contiene información factual
            if len(sent) > 5:  # Ignorar oraciones muy cortas
                raiz = None
                sujeto = None
                objeto = None
                
                # Encontrar la estructura básica de la oración
                for token in sent:
                    if token.dep_ == "ROOT":
                        raiz = token
                    elif token.dep_ == "nsubj":
                        sujeto = token
                    elif token.dep_ in ["dobj", "pobj"]:
                        objeto = token
                
                # Si encontramos una estructura válida
                if sujeto and raiz:
                    contexto = sent.text
                    
                    # Extraer información adicional
                    atributos = []
                    for token in sent:
                        if token.dep_ in ["amod", "advmod"] and token.head in [sujeto, objeto]:
                            atributos.append(token.text)
                    
                    hecho = {
                        'sujeto': sujeto.text,
                        'accion': raiz.text,
                        'objeto': objeto.text if objeto else None,
                        'atributos': atributos,
                        'contexto': contexto
                    }
                    
                    # Verificar relevancia del hecho
                    if self._es_hecho_relevante(hecho):
                        hechos.append(hecho)
        
        return hechos

    def _es_hecho_relevante(self, hecho: Dict) -> bool:
        """Determina si un hecho es suficientemente relevante para generar preguntas"""
        # Debe tener al menos sujeto y acción
        if not (hecho['sujeto'] and hecho['accion']):
            return False
            
        # El sujeto debe ser un concepto significativo
        doc_sujeto = self.nlp(hecho['sujeto'])
        if all(token.is_stop for token in doc_sujeto):
            return False
            
        # Debe tener suficiente contenido semántico
        contexto = self.nlp(hecho['contexto'])
        palabras_contenido = [token for token in contexto if not token.is_stop]
        if len(palabras_contenido) < 3:
            return False
            
        return True

    def _validar_texto(self, texto: str) -> bool:
        """Valida que el texto no contenga caracteres especiales aislados ni patrones incoherentes"""
        # Evitar texto vacío o muy corto
        if not texto or len(texto.split()) < 2:
            return False
            
        # Evitar caracteres especiales aislados
        if re.match(r'^[,\.;:]|[,\.;:]$', texto.strip()):
            return False
            
        # Evitar múltiples espacios o caracteres especiales seguidos
        if re.search(r'[,\.;:]{2,}|\s{2,}', texto):
            return False
            
        # Evitar texto muy corto o sin sentido
        if len(texto.strip().split()) < 2:
            return False
            
        # Evitar frases que empiezan con "y", "o", "pero" aislados
        if re.match(r'^(y|o|pero)\s', texto.lower()):
            return False
            
        return True

    def _es_alternativa_valida(self, alternativa: str, respuesta: str, otras_alternativas: List[str]) -> bool:
        """Valida que una alternativa sea coherente y distintiva"""
        doc_alt = self.nlp(alternativa)
        doc_resp = self.nlp(respuesta)

        # Evitar alternativas muy similares a la respuesta
        if doc_alt.similarity(doc_resp) > 0.8:
            return False

        # Evitar alternativas repetidas o muy similares entre sí
        for otra in otras_alternativas:
            doc_otra = self.nlp(otra)
            if doc_alt.similarity(doc_otra) > 0.7:
                return False

        # Verificar coherencia gramatical básica
        tiene_sustantivo = any(token.pos_ == "NOUN" for token in doc_alt)
        tiene_verbo = any(token.pos_ == "VERB" for token in doc_alt)
        
        return tiene_sustantivo or tiene_verbo

    def _generar_pregunta_desde_hecho(self, hecho: Dict) -> Optional[Dict]:
        """Genera una pregunta analítica a partir de un hecho"""
        try:
            doc = self.nlp(hecho['contexto'])
            
            # Extraer entidades nombradas principales
            entidades = []
            for ent in doc.ents:
                if ent.label_ in ['PER', 'ORG', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE']:
                    # Validar que la entidad sea gramaticalmente coherente
                    if self._es_frase_coherente(ent.text):
                        entidades.append((ent.text, ent.label_))

            # Si no hay entidades, buscar sustantivos importantes
            if not entidades:
                for chunk in doc.noun_chunks:
                    if len(chunk.text.split()) >= 2 and not chunk.root.is_stop:
                        if self._es_frase_coherente(chunk.text):
                            entidades.append((chunk.text, 'CONCEPT'))

            if not entidades:
                return None

            # Seleccionar una entidad al azar para preguntar
            entidad, tipo = random.choice(entidades)
            
            # Determinar el tipo de pregunta basado en la entidad
            pregunta, contexto = self._formular_pregunta_entidad(entidad, tipo, doc)
            if not pregunta:
                return None

            # Obtener respuesta usando QA
            respuesta = self.qa_model(
                question=pregunta,
                context=contexto
            )

            if respuesta['score'] < 0.6:
                return None

            # Validar respuesta
            if not self._es_respuesta_valida(respuesta['answer'], tipo):
                return None

            # Generar alternativas relacionadas pero distintas
            alternativas = self._generar_alternativas_coherentes(
                respuesta['answer'],
                tipo,
                doc
            )

            if len(alternativas) < 3:
                return None

            opciones = alternativas + [respuesta['answer']]
            random.shuffle(opciones)

            return {
                'pregunta': pregunta,
                'opciones': opciones,
                'respuesta_correcta': respuesta['answer'],
                'tipo': 'alternativas'
            }

        except Exception as e:
            print(f"Error generando pregunta: {str(e)}")
            return None

    def _es_frase_coherente(self, texto: str) -> bool:
        """Valida que una frase sea gramaticalmente coherente"""
        doc = self.nlp(texto)
        
        # Verificar estructura básica
        if len(doc) < 2:
            return False
            
        # La frase debe tener al menos un sustantivo
        tiene_sustantivo = False
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN']:
                tiene_sustantivo = True
                break
                
        if not tiene_sustantivo:
            return False
            
        # No debe empezar con palabras de conexión
        if doc[0].is_stop and doc[0].text.lower() in ['y', 'o', 'pero', 'sin', 'con']:
            return False
            
        # No debe tener errores gramaticales obvios
        for token in doc:
            if token.dep_ == 'ROOT' and token.head.i != token.i:
                return False
                
        return True

    def _formular_pregunta_entidad(self, entidad: str, tipo: str, doc) -> Tuple[Optional[str], Optional[str]]:
        """Formula una pregunta significativa sobre una entidad"""
        # Encontrar información relevante sobre la entidad en el documento
        contexto_relevante = None
        for sent in doc.sents:
            if entidad in sent.text:
                contexto_relevante = sent.text
                break
                
        if not contexto_relevante:
            return None, None
            
        # Elegir tipo de pregunta según el tipo de entidad
        if tipo == 'PER':
            return f"¿Cuál es el rol de {entidad}?", contexto_relevante
        elif tipo == 'ORG':
            return f"¿Cuál es la función principal de {entidad}?", contexto_relevante
        elif tipo == 'CONCEPT':
            return f"¿Qué caracteriza a {entidad}?", contexto_relevante
        elif tipo in ['PRODUCT', 'WORK_OF_ART']:
            return f"¿Cuál es la importancia de {entidad}?", contexto_relevante
        else:
            return f"¿Qué define a {entidad}?", contexto_relevante

    def _es_respuesta_valida(self, respuesta: str, tipo: str) -> bool:
        """Valida que una respuesta sea apropiada para el tipo de entidad"""
        doc = self.nlp(respuesta)
        
        # La respuesta debe tener una longitud mínima
        if len(doc) < 3:
            return False
            
        # Debe contener al menos un verbo y un sustantivo
        tiene_verbo = any(token.pos_ == 'VERB' for token in doc)
        tiene_sustantivo = any(token.pos_ in ['NOUN', 'PROPN'] for token in doc)
        
        if not (tiene_verbo and tiene_sustantivo):
            return False
            
        # No debe ser una respuesta trivial
        palabras_significativas = len([token for token in doc if not token.is_stop])
        if palabras_significativas < 2:
            return False
            
        return True

    def _generar_alternativas_coherentes(self, respuesta: str, tipo: str, doc) -> List[str]:
        """Genera alternativas coherentes basadas en el tipo de entidad"""
        alternativas = []
        doc_respuesta = self.nlp(respuesta)
        
        # Extraer conceptos similares del documento
        conceptos_similares = []
        for sent in doc.sents:
            # Buscar frases nominales con estructura similar
            for chunk in sent.noun_chunks:
                if chunk.text != respuesta and len(chunk.text.split()) >= 2:
                    # Calcular similitud semántica
                    similitud = chunk.similarity(doc_respuesta)
                    if 0.3 <= similitud <= 0.7:  # Similitud moderada
                        conceptos_similares.append(chunk.text)
                        
        # Asegurar que las alternativas sean diferentes entre sí
        for concepto in conceptos_similares:
            if len(alternativas) >= 3:
                break
                
            # Verificar que el concepto es válido
            if self._es_alternativa_valida(concepto, respuesta, alternativas):
                alternativas.append(concepto)
                
        return alternativas

    def _obtener_contexto_relevante(self, token, doc, window_size=30):
        """Obtiene el contexto más relevante alrededor de un token"""
        start = max(0, token.i - window_size)
        end = min(len(doc), token.i + window_size)
        return doc[start:end].text

    def _es_oracion_completa(self, sent) -> bool:
        """Verifica que una oración tenga estructura completa"""
        tiene_sujeto = False
        tiene_verbo = False
        tiene_objeto = False
        
        for token in sent:
            if token.dep_ == 'nsubj':
                tiene_sujeto = True
            elif token.pos_ == 'VERB':
                tiene_verbo = True
            elif token.dep_ in ['dobj', 'pobj']:
                tiene_objeto = True
                
        return tiene_sujeto and tiene_verbo and tiene_objeto

    def _modificar_oracion_coherentemente(self, oracion: str) -> Optional[str]:
        """Modifica una oración manteniendo coherencia gramatical"""
        doc = self.nlp(oracion)
        tokens = list(doc)
        
        # Identificar el componente principal a modificar
        for i, token in enumerate(tokens):
            if token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop:
                # Buscar un reemplazo semánticamente relacionado
                reemplazo = self._encontrar_reemplazo_semantico(token)
                if reemplazo:
                    nueva_oracion = []
                    for j, t in enumerate(tokens):
                        if j == i:
                            nueva_oracion.append(reemplazo)
                        else:
                            nueva_oracion.append(t.text)
                            
                    return ' '.join(nueva_oracion)
                    
        return None

    def _encontrar_reemplazo_semantico(self, token) -> Optional[str]:
        """Encuentra un reemplazo semánticamente relacionado para un token"""
        reemplazos_potenciales = []
        
        for otro_token in self.nlp.vocab:
            if (otro_token.has_vector and 
                otro_token.pos_ == token.pos_ and 
                otro_token.text.lower() != token.text.lower()):
                similitud = token.similarity(otro_token)
                if 0.3 <= similitud <= 0.7:
                    reemplazos_potenciales.append(otro_token.text)
                    
        return random.choice(reemplazos_potenciales) if reemplazos_potenciales else None

    def generar_preguntas(self, texto: str) -> List[Dict]:
        """Genera preguntas a partir de un texto"""
        # Extraer hechos
        hechos = self._extraer_hechos(texto)
        if not hechos:
            return []
            
        preguntas = []
        
        # Generar preguntas para cada hecho
        for hecho in hechos:
            pregunta = self._generar_pregunta_desde_hecho(hecho)
            if pregunta and self._validar_pregunta(pregunta, preguntas):
                preguntas.append(pregunta)
                
        return preguntas

    def _validar_pregunta(self, nueva_pregunta: Dict, preguntas: List[Dict]) -> bool:
        """Valida que la pregunta sea única y significativa"""
        # Verificar duplicados
        for pregunta in preguntas:
            if pregunta['pregunta'] == nueva_pregunta['pregunta']:
                return False
                
            # Verificar similitud
            similitud = self.nlp(pregunta['pregunta']).similarity(
                self.nlp(nueva_pregunta['pregunta'])
            )
            if similitud > 0.8:
                return False
        
        return True

class GeneradorAlternativas:
    def __init__(self, nlp):
        self.nlp = nlp
        self.cache_semantico = {}

    def _generar_alternativas(self, respuesta: str, hecho: Dict, doc_contexto) -> List[str]:
        return self._generar_alternativas_mejoradas(respuesta, hecho, doc_contexto)

    def _generar_alternativas_mejoradas(self, respuesta: str, hecho: Dict, doc_contexto) -> List[str]:
        """Genera alternativas más coherentes y relacionadas"""
        alternativas = set()
        doc_respuesta = self.nlp(respuesta)
        
        # Identificar la categoría semántica de la respuesta
        categoria = self._identificar_categoria_semantica(doc_respuesta)
        
        if categoria:
            # Buscar elementos de la misma categoría
            for token in doc_contexto:
                if token.has_vector:
                    similitud = token.similarity(doc_respuesta[0])
                    if 0.3 <= similitud <= 0.7:  # Similitud moderada
                        alt_candidata = self._extraer_frase_coherente(token, doc_contexto)
                        if alt_candidata and self._validar_alternativa(alt_candidata, respuesta):
                            alternativas.add(alt_candidata)
        
        # Si no hay suficientes alternativas, buscar en el vocabulario general
        if len(alternativas) < 3:
            for token in self.nlp.vocab:
                if token.has_vector and token.text.lower() != respuesta.lower():
                    similitud = token.similarity(doc_respuesta[0])
                    if 0.3 <= similitud <= 0.7:
                        alternativas.add(token.text)
                        if len(alternativas) >= 5:
                            break
        
        return list(alternativas)[:3]

    def _identificar_categoria_semantica(self, doc) -> Optional[str]:
        """Identifica la categoría semántica principal de un texto"""
        categorias = []
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN', 'VERB']:
                # Buscar hiperónimos
                similares = []
                for otro_token in self.nlp.vocab:
                    if otro_token.has_vector and otro_token.is_lower:
                        similitud = token.similarity(otro_token)
                        if similitud > 0.8:
                            similares.append(otro_token.text)
                if similares:
                    categorias.extend(similares)
        
        return max(categorias, key=categorias.count) if categorias else None

    def _validar_alternativa(self, alternativa: str, respuesta: str) -> bool:
        """Valida que una alternativa sea coherente y diferente de la respuesta"""
        # Evitar alternativas muy similares
        similitud = self.nlp(alternativa).similarity(self.nlp(respuesta))
        if similitud > 0.8:
            return False
            
        # Verificar longitud mínima y estructura
        if len(alternativa.split()) < 2:
            return False
            
        # Verificar coherencia gramatical
        doc = self.nlp(alternativa)
        tiene_sustantivo = any(token.pos_ in ['NOUN', 'PROPN'] for token in doc)
        tiene_verbo = any(token.pos_ == 'VERB' for token in doc)
        
        if not (tiene_sustantivo or tiene_verbo):
            return False
            
        return True

    def _extraer_frase_coherente(self, token, doc) -> Optional[str]:
        """Extrae una frase coherente alrededor de un token"""
        inicio = token.i
        fin = token.i + 1
        
        # Expandir hacia atrás y adelante solo si mantiene coherencia
        while inicio > 0 and doc[inicio - 1].pos_ in ['ADJ', 'DET', 'ADP']:
            if self._validar_texto(doc[inicio-1:fin].text):
                inicio -= 1
            else:
                break
                
        while fin < len(doc) and doc[fin].pos_ in ['ADJ', 'ADP', 'NOUN']:
            if self._validar_texto(doc[inicio:fin+1].text):
                fin += 1
            else:
                break
                
        frase = doc[inicio:fin].text
        return frase if self._validar_texto(frase) else None

    def _identificar_categoria(self, doc_respuesta) -> Optional[str]:
        """Identifica la categoría semántica de la respuesta"""
        # Buscar patrones semánticos
        for token in doc_respuesta:
            if token.pos_ == 'NOUN':
                # Verificar categorías conocidas
                if token.text.lower() in self.cache_semantico:
                    return self.cache_semantico[token.text.lower()]
                
                # Analizar relaciones semánticas
                hipernimos = self._obtener_hipernimos(token)
                if hipernimos:
                    categoria = hipernimos[0]
                    self.cache_semantico[token.text.lower()] = categoria
                    return categoria
                    
        return None

    def _obtener_hipernimos(self, token) -> List[str]:
        """Obtiene los hipernimos (categorías superiores) de un término"""
        hipernimos = []
        
        # Buscar términos más generales
        for otro_token in self.nlp.vocab:
            if otro_token.has_vector and otro_token.is_lower:
                similitud = token.similarity(otro_token)
                if 0.7 <= similitud < 0.9:  # Muy similar pero más general
                    hipernimos.append(otro_token.text)
                    
        return sorted(hipernimos, key=lambda x: len(x))

    def _generar_por_categoria(
        self,
        respuesta: str,
        categoria: str,
        doc_respuesta,
        doc_contexto
    ) -> List[str]:
        """Genera alternativas de la misma categoría semántica"""
        alternativas = set()
        
        # Buscar términos de la misma categoría
        for token in doc_contexto:
            if token.has_vector:
                # Verificar si pertenece a la misma categoría
                hipernimos_token = self._obtener_hipernimos(token)
                if categoria in hipernimos_token:
                    frase = self._extraer_frase(token, doc_contexto)
                    if frase and frase != respuesta:
                        alternativas.add(frase)
        
        # Si no hay suficientes, buscar en el vocabulario general
        if len(alternativas) < 3:
            for token in self.nlp.vocab:
                if token.has_vector and categoria in self._obtener_hipernimos(token):
                    alternativas.add(token.text)
                    if len(alternativas) >= 5:  # Generar más para filtrar después
                        break
        
        return list(alternativas)

    def _generar_por_contexto(
        self,
        respuesta: str,
        doc_respuesta,
        doc_contexto
    ) -> List[str]:
        """Genera alternativas basadas en el contexto cuando no hay categoría clara"""
        alternativas = set()
        
        # Encontrar frases nominales similares
        for chunk in doc_contexto.noun_chunks:
            if chunk.text != respuesta:
                similitud = chunk.similarity(doc_respuesta)
                if 0.3 <= similitud <= 0.7:
                    alternativas.add(chunk.text)
        
        # Generar variaciones sintácticas
        for token in doc_respuesta:
            if token.pos_ in ['NOUN', 'VERB', 'ADJ']:
                similares = []
                for otro_token in doc_contexto:
                    if otro_token.pos_ == token.pos_:
                        similitud = token.similarity(otro_token)
                        if 0.3 <= similitud <= 0.7:
                            similares.append(otro_token.text)
                            
                if similares:
                    nueva_alt = respuesta.replace(token.text, random.choice(similares))
                    alternativas.add(nueva_alt)
        
        return list(alternativas)

    def _extraer_frase(self, token, doc) -> Optional[str]:
        """Extrae una frase completa alrededor de un token"""
        inicio = token.i
        fin = token.i + 1
        
        # Expandir hacia atrás
        while inicio > 0 and doc[inicio - 1].pos_ in ['ADJ', 'DET', 'ADP']:
            inicio -= 1
            
        # Expandir hacia adelante
        while fin < len(doc) and doc[fin].pos_ in ['ADJ', 'ADP', 'NOUN']:
            fin += 1
            
        if fin - inicio > 1:  # Al menos dos tokens
            return doc[inicio:fin].text
        return None

    def _validar_alternativas(
        self,
        alternativas: List[str],
        respuesta: str,
        tipo: str
    ) -> List[str]:
        """Valida y filtra las alternativas generadas"""
        alternativas_validadas = []
        doc_respuesta = self.nlp(respuesta)
        
        for alt in alternativas:
            # Verificar longitud mínima
            if len(alt.split()) < 2:
                continue
                
            # Verificar que no es muy similar a la respuesta
            similitud = self.nlp(alt).similarity(doc_respuesta)
            if similitud > 0.8:
                continue
                
            # Verificar coherencia gramatical
            if not self._es_gramaticalmente_coherente(alt, tipo):
                continue
                
            alternativas_validadas.append(alt)
            
        return alternativas_validadas

    def _es_gramaticalmente_coherente(self, texto: str, tipo: str) -> bool:
        """Verifica la coherencia gramatical de una alternativa"""
        doc = self.nlp(texto)
        
        # Verificar estructura básica
        tiene_sujeto = False
        tiene_verbo = False
        
        for token in doc:
            if token.dep_ == 'nsubj':
                tiene_sujeto = True
            elif token.pos_ == 'VERB':
                tiene_verbo = True
                
        # Para respuestas que son frases nominales
        if tipo == 'caracteristica':
            return any(token.pos_ in ['NOUN', 'ADJ'] for token in doc)
            
        # Para respuestas que son acciones
        elif tipo == 'funcion':
            return tiene_verbo
            
        # Para respuestas que son relaciones
        elif tipo == 'relacion':
            return tiene_sujeto and tiene_verbo
            
        return True

class GeneradorCuestionarios:
    def __init__(self):
        print("Inicializando modelos...")
        try:
            self.qa_model = pipeline(
                "question-answering",
                model="mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"
            )
            self.nlp = spacy.load("es_core_news_lg")
            print("Modelos cargados correctamente")
        except Exception as e:
            print(f"Error inicializando modelos: {str(e)}")
            raise

    def _extraer_oraciones(self, texto: str) -> List[str]:
        """Extrae oraciones del texto"""
        doc = self.nlp(texto)
        return [sent.text for sent in doc.sents if len(sent.text.split()) >= 8]

    def _limpiar_texto(self, texto: str) -> str:
        """Limpia el texto de caracteres especiales y espacios innecesarios"""
        # Eliminar espacios y caracteres especiales al inicio y final
        texto = texto.strip()
        # Eliminar caracteres especiales aislados
        texto = re.sub(r'\s*[,;:]\s*', ' ', texto)
        # Eliminar espacios múltiples
        texto = re.sub(r'\s+', ' ', texto)
        return texto

    def _es_oracion_completa(self, sent) -> bool:
        """Verifica que una oración tenga estructura completa"""
        tiene_sujeto = False
        tiene_verbo = False
        tiene_objeto = False
        
        for token in sent:
            if token.dep_ == 'nsubj':
                tiene_sujeto = True
            elif token.pos_ == 'VERB':
                tiene_verbo = True
            elif token.dep_ in ['dobj', 'pobj']:
                tiene_objeto = True
                
        return tiene_sujeto and tiene_verbo and tiene_objeto

    def _modificar_oracion_coherentemente(self, oracion: str) -> Optional[str]:
        """Modifica una oración manteniendo coherencia gramatical"""
        doc = self.nlp(oracion)
        tokens = list(doc)
        
        # Identificar el componente principal a modificar
        for i, token in enumerate(tokens):
            if token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop:
                # Buscar un reemplazo semánticamente relacionado
                reemplazo = self._encontrar_reemplazo_semantico(token)
                if reemplazo:
                    nueva_oracion = []
                    for j, t in enumerate(tokens):
                        if j == i:
                            nueva_oracion.append(reemplazo)
                        else:
                            nueva_oracion.append(t.text)
                            
                    return ' '.join(nueva_oracion)
                    
        return None

    def _encontrar_reemplazo_semantico(self, token) -> Optional[str]:
        """Encuentra un reemplazo semánticamente relacionado para un token"""
        reemplazos_potenciales = []
        
        for otro_token in self.nlp.vocab:
            if (otro_token.has_vector and 
                otro_token.pos_ == token.pos_ and 
                otro_token.text.lower() != token.text.lower()):
                similitud = token.similarity(otro_token)
                if 0.3 <= similitud <= 0.7:
                    reemplazos_potenciales.append(otro_token.text)
                    
        return random.choice(reemplazos_potenciales) if reemplazos_potenciales else None

    def _es_concepto_valido(self, concepto: str) -> bool:
        """Verifica si un concepto es válido para generar una pregunta"""
        # Limpiar el concepto
        concepto = self._limpiar_texto(concepto)
        
        # Verificar longitud mínima
        if len(concepto.split()) < 2:
            return False
            
        # Evitar números aislados y años
        if re.match(r'^\d+$', concepto) or re.match(r'^\d+\s+(años?|meses)$', concepto):
            return False
            
        # Evitar caracteres especiales
        if re.search(r'[,;:]', concepto):
            return False
            
        # Evitar conceptos sin significado
        doc = self.nlp(concepto)
        if all(token.is_stop or token.is_punct for token in doc):
            return False
            
        return True

    def _generar_pregunta(self, oracion: str) -> Optional[Dict]:
        """Genera una pregunta y su respuesta de forma dinámica"""
        try:
            doc = self.nlp(oracion)
            
            # Analizar estructura semántica de la oración
            elementos_significativos = []
            for token in doc:
                if self._es_elemento_significativo(token):
                    contexto = self._analizar_contexto_elemento(token, doc)
                    if contexto:
                        elementos_significativos.append({
                            'elemento': token.text,
                            'tipo': token.pos_,
                            'accion': contexto.get('accion'),
                            'atributos': contexto.get('atributos', []),
                            'relaciones': contexto.get('relaciones', [])
                        })

            if not elementos_significativos:
                return None

            # Seleccionar elemento para la pregunta
            elemento_elegido = random.choice(elementos_significativos)
            
            # Construir pregunta basada en el análisis semántico
            pregunta, tipo_respuesta = self._construir_pregunta_por_semantica(elemento_elegido)
            if not pregunta:
                return None

            # Obtener respuesta usando el modelo QA
            respuesta = self.qa_model(
                question=pregunta,
                context=oracion
            )

            if respuesta['score'] < 0.3:
                return None

            # Validar y limpiar respuesta
            respuesta_limpia = self._limpiar_texto(respuesta['answer'])
            if not (self._es_respuesta_valida(respuesta_limpia) and 
                    self._validar_estructura_respuesta(respuesta_limpia) and
                    self._validar_respuesta(respuesta_limpia, tipo_respuesta)):
                return None

            # Generar alternativas semánticamente relacionadas
            alternativas = self._generar_alternativas_semanticas(
                respuesta_limpia,
                elemento_elegido,
                tipo_respuesta,
                doc
            )

            if len(alternativas) < 3:
                return None

            # Asegurar coherencia y diversidad
            alternativas = self._refinar_alternativas(alternativas, respuesta_limpia)
            todas_opciones = alternativas[:3] + [respuesta_limpia]
            random.shuffle(todas_opciones)

            return {
                'pregunta': pregunta,
                'opciones': todas_opciones,
                'respuesta_correcta': respuesta_limpia,
                'tipo': 'alternativas'
            }

        except Exception as e:
            print(f"Error en generación de pregunta: {str(e)}")
            return None

    def _es_elemento_significativo(self, token) -> bool:
        """Determina si un elemento es significativo para generar una pregunta"""
        return (
            not token.is_stop and
            token.pos_ in ['NOUN', 'PROPN', 'VERB'] and
            len(token.text) > 2 and
            token.dep_ not in ['aux', 'det', 'punct']
        )

    def _analizar_contexto_elemento(self, token, doc) -> Dict:
        """Analiza el contexto semántico de un elemento"""
        contexto = {
            'accion': None,
            'atributos': [],
            'relaciones': []
        }

        for t in doc:
            if t.head == token:
                if t.pos_ == 'VERB':
                    contexto['accion'] = t.text
                elif t.pos_ == 'ADJ':
                    contexto['atributos'].append(t.text)
            elif token.head == t and t.pos_ == 'VERB':
                contexto['accion'] = t.text
            elif t.dep_ in ['nmod', 'amod'] and t.head == token:
                contexto['atributos'].append(t.text)
            elif t.dep_ in ['conj', 'prep'] and t.head == token:
                contexto['relaciones'].append(t.text)

        return contexto

    def _construir_pregunta_por_semantica(self, elemento: Dict) -> Tuple[Optional[str], Optional[str]]:
        """Construye una pregunta basada en el análisis semántico"""
        texto_elemento = elemento['elemento']
        tipo_elemento = elemento['tipo']
        
        # Analizar características semánticas
        doc = self.nlp(texto_elemento)
        vector_elemento = doc.vector
        
        # Determinar el enfoque de la pregunta basado en el análisis
        if elemento['accion']:
            similitud_accion = self.nlp(elemento['accion']).similarity(doc)
            if similitud_accion > 0.3:
                return f"¿De qué manera {elemento['accion']} {texto_elemento}?", "ACCION"
        
        if elemento['atributos']:
            return f"¿Qué característica define a {texto_elemento}?", "ATRIBUTO"
        
        if elemento['relaciones']:
            return f"¿Cómo se relaciona {texto_elemento} con su entorno?", "RELACION"
        
        # Si no hay contexto específico, generar pregunta por análisis vectorial
        similitud_abstracta = self.nlp("concepto definición significado").similarity(doc)
        similitud_funcional = self.nlp("función propósito objetivo").similarity(doc)
        
        if similitud_abstracta > similitud_funcional:
            return f"¿Qué representa {texto_elemento}?", "CONCEPTO"
        else:
            return f"¿Para qué sirve {texto_elemento}?", "FUNCION"

    def _generar_alternativas_semanticas(self, respuesta: str, elemento: Dict, tipo_respuesta: str, doc) -> List[str]:
        """Genera alternativas semánticamente coherentes"""
        alternativas = set()
        doc_respuesta = self.nlp(respuesta)

        # Generar alternativas basadas en similitud semántica
        for token in doc:
            if self._es_elemento_significativo(token):
                similitud = token.similarity(doc_respuesta[0])
                if 0.3 <= similitud <= 0.7:
                    alt = self._construir_alternativa(token, tipo_respuesta)
                    if alt and alt != respuesta:
                        alternativas.add(alt)

        # Si no hay suficientes alternativas, buscar en el vocabulario general
        if len(alternativas) < 3:
            for token in self.nlp.vocab:
                if len(alternativas) >= 5:
                    break
                if token.has_vector and token.text.lower() != respuesta.lower():
                    similitud = token.similarity(doc_respuesta[0])
                    if 0.3 <= similitud <= 0.7:
                        alt = self._construir_alternativa(token, tipo_respuesta)
                        if alt and alt != respuesta:
                            alternativas.add(alt)

        return list(alternativas)

    def _construir_alternativa(self, token, tipo_respuesta: str) -> Optional[str]:
        """Construye una alternativa coherente según el tipo de respuesta"""
        if tipo_respuesta == "ACCION":
            return f"{token.text} el elemento"
        elif tipo_respuesta == "ATRIBUTO":
            return f"Es {token.text}"
        elif tipo_respuesta == "RELACION":
            return f"Se relaciona con {token.text}"
        elif tipo_respuesta == "CONCEPTO":
            return token.text
        elif tipo_respuesta == "FUNCION":
            return f"Sirve para {token.text}"
        return None

    def _refinar_alternativas(self, alternativas: List[str], respuesta: str) -> List[str]:
        """Refina las alternativas para asegurar coherencia y diversidad"""
        alternativas_refinadas = []
        doc_respuesta = self.nlp(respuesta)
        
        for alt in alternativas:
            doc_alt = self.nlp(alt)
            # Verificar que la alternativa es suficientemente diferente
            if all(doc_alt.similarity(self.nlp(otra)) < 0.7 for otra in alternativas_refinadas):
                # Verificar que mantiene relación temática
                if doc_alt.similarity(doc_respuesta) > 0.3:
                    alternativas_refinadas.append(alt)
                    
        return alternativas_refinadas

    def _validar_respuesta(self, respuesta: str, tipo_respuesta: str) -> bool:
        """Valida que una respuesta sea coherente según su tipo"""
        if not respuesta or len(respuesta.split()) < 2:
            return False
            
        doc = self.nlp(respuesta)
        
        # Verificar estructura básica según el tipo
        if tipo_respuesta == "ACCION":
            tiene_verbo = any(token.pos_ == 'VERB' for token in doc)
            return tiene_verbo
            
        elif tipo_respuesta == "ATRIBUTO":
            tiene_adj = any(token.pos_ == 'ADJ' for token in doc)
            tiene_sustantivo = any(token.pos_ in ['NOUN', 'PROPN'] for token in doc)
            return tiene_adj or tiene_sustantivo
            
        elif tipo_respuesta == "RELACION":
            tiene_prep = any(token.pos_ == 'ADP' for token in doc)
            tiene_sustantivo = any(token.pos_ in ['NOUN', 'PROPN'] for token in doc)
            return tiene_prep and tiene_sustantivo
            
        elif tipo_respuesta == "CONCEPTO":
            tiene_sustantivo = any(token.pos_ in ['NOUN', 'PROPN'] for token in doc)
            return tiene_sustantivo
            
        elif tipo_respuesta == "FUNCION":
            tiene_verbo = any(token.pos_ == 'VERB' for token in doc)
            tiene_complemento = any(token.dep_ in ['dobj', 'pobj'] for token in doc)
            return tiene_verbo and tiene_complemento
            
        return True

    def _validar_estructura_respuesta(self, respuesta: str) -> bool:
        """Valida la estructura gramatical básica de una respuesta"""
        doc = self.nlp(respuesta)
        
        # Verificar longitud mínima y máxima
        if len(doc) < 2 or len(doc) > 15:
            return False
            
        # Verificar que tenga al menos un elemento significativo
        tiene_contenido = False
        for token in doc:
            if not token.is_stop and not token.is_punct:
                tiene_contenido = True
                break
                
        return tiene_contenido

    def _es_respuesta_valida(self, respuesta: str) -> bool:
        """Verifica si una respuesta es válida en términos generales"""
        # Limpiar espacios y caracteres especiales
        respuesta = respuesta.strip()
        if not respuesta:
            return False
            
        # Evitar respuestas muy cortas o muy largas
        palabras = respuesta.split()
        if len(palabras) < 2 or len(palabras) > 10:
            return False
            
        # Verificar que no sea una respuesta trivial
        doc = self.nlp(respuesta)
        palabras_significativas = [token for token in doc if not token.is_stop]
        if len(palabras_significativas) < 2:
            return False
            
        return True

    def _generar_verdadero_falso_mejorado(self, texto: str) -> Optional[Dict]:
        """Genera una pregunta de verdadero/falso coherente"""
        doc = self.nlp(texto)
        
        # Extraer oraciones completas y coherentes
        oraciones_validas = []
        for sent in doc.sents:
            if len(sent.text.split()) >= 8 and self._es_oracion_completa(sent):
                oraciones_validas.append(sent.text)
                
        if not oraciones_validas:
            return None
            
        oracion = random.choice(oraciones_validas)
        es_verdadero = random.choice([True, False])
        
        if es_verdadero:
            return {
                'pregunta': oracion,
                'respuesta_correcta': 'Verdadero',
                'tipo': 'verdadero_falso'
            }
        else:
            # Modificar la oración de manera coherente
            oracion_modificada = self._modificar_oracion_coherentemente(oracion)
            if oracion_modificada:
                return {
                    'pregunta': oracion_modificada,
                    'respuesta_correcta': 'Falso',
                    'tipo': 'verdadero_falso'
                }
                
        return None

    def generar_cuestionario(self, ruta_archivo: str) -> Dict:
        """Genera cuestionarios"""
        try:
            print("Cargando datos...")
            with open(ruta_archivo, 'r', encoding='utf-8') as archivo:
                datos = json.load(archivo)

            cuestionarios = {}

            for item in tqdm(datos['quiz'], desc="Generando cuestionarios"):
                materia = item['materia']
                texto = item['texto']

                if materia not in cuestionarios:
                    cuestionarios[materia] = []

                print(f"\nProcesando {materia}...")

                # Extraer oraciones
                oraciones = self._extraer_oraciones(texto)
                if not oraciones:
                    print(f"No se encontraron oraciones para {materia}")
                    continue

                preguntas = []

                # Generar preguntas de alternativas
                print("Generando preguntas de alternativas...")  # Debug
                for oracion in oraciones:
                    if len([p for p in preguntas if p['tipo'] == 'alternativas']) >= 7:
                        break
                    pregunta = self._generar_pregunta(oracion)
                    if pregunta:
                        preguntas.append(pregunta)

                # Generar preguntas V/F
                print("Generando preguntas V/F...")  # Debug
                for oracion in oraciones:
                    if len([p for p in preguntas if p['tipo'] == 'verdadero_falso']) >= 3:
                        break
                    pregunta = self._generar_verdadero_falso_mejorado(oracion)
                    if pregunta:
                        preguntas.append(pregunta)

                if preguntas:  # Guardar si hay al menos una pregunta
                    print(f"Guardando {len(preguntas)} preguntas para {materia}")  # Debug
                    cuestionarios[materia].append({
                        'texto': texto,
                        'fuente': item['fuente'],
                        'preguntas': preguntas
                    })

            return cuestionarios

        except Exception as e:
            print(f"Error en generación de cuestionario: {str(e)}")
            return None