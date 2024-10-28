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

    def _es_alternativa_valida(self, alternativa: str, pregunta: str, otras_alternativas: List[str]) -> bool:
        """Valida que una alternativa sea coherente y no repetida"""
        # Validar formato básico
        if not self._validar_texto(alternativa):
            return False
            
        # Evitar alternativas idénticas o muy similares
        for otra in otras_alternativas:
            # Comparar textos normalizados
            alt1 = alternativa.lower().strip().strip('.,;:')
            alt2 = otra.lower().strip().strip('.,;:')
            
            if alt1 == alt2:
                return False
                
            # Calcular similitud para evitar alternativas casi idénticas
            similitud = self.nlp(alternativa).similarity(self.nlp(otra))
            if similitud > 0.9:  # Muy similar
                return False
        
        # Verificar relación semántica con la pregunta
        similitud_pregunta = self.nlp(alternativa).similarity(self.nlp(pregunta))
        if similitud_pregunta < 0.2:  # Muy poco relacionada
            return False
            
        return True

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
        """Genera una pregunta y su respuesta"""
        try:
            # Obtener sujetos principales
            doc = self.nlp(oracion)
            sujetos = []
            
            # Extraer conceptos significativos
            for chunk in doc.noun_chunks:
                concepto = self._limpiar_texto(chunk.text)
                if self._es_concepto_valido(concepto):
                    sujetos.append(concepto)

            if not sujetos:
                return None

            # Seleccionar un sujeto y generar pregunta
            sujeto = random.choice(sujetos)
            
            # Patrones de preguntas más naturales
            patrones = [
                f"¿Cuál es el principal objetivo de {sujeto}?",
                f"¿Qué rol desempeña {sujeto}?",
                f"¿Cuál es la importancia de {sujeto}?",
                f"¿Cómo se define {sujeto}?",
                f"¿Qué hace {sujeto}?"
            ]
            
            pregunta_base = random.choice(patrones)

            # Obtener respuesta
            respuesta = self.qa_model(
                question=pregunta_base,
                context=oracion
            )

            if respuesta['score'] < 0.3:
                return None

            # Limpiar la respuesta
            respuesta_limpia = self._limpiar_texto(respuesta['answer'])
            if not self._es_concepto_valido(respuesta_limpia):
                return None

            # Generar alternativas coherentes
            alternativas = []
            for chunk in doc.noun_chunks:
                texto = self._limpiar_texto(chunk.text)
                if (self._es_concepto_valido(texto) and 
                    texto != sujeto and 
                    texto != respuesta_limpia and 
                    texto not in alternativas):
                    alternativas.append(texto)

            if len(alternativas) < 3:
                return None

            # Seleccionar las mejores alternativas
            mejores_alternativas = alternativas[:3]
            todas_opciones = mejores_alternativas + [respuesta_limpia]
            random.shuffle(todas_opciones)

            return {
                'pregunta': pregunta_base,
                'opciones': todas_opciones,
                'respuesta_correcta': respuesta_limpia,
                'tipo': 'alternativas'
            }

        except Exception as e:
            print(f"Error en generación de pregunta: {str(e)}")
            return None

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