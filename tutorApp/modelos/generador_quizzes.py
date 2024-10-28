# Generador de Quizzes
# por Franco Benassi
from transformers import pipeline, AutoTokenizer
import spacy
import json
from typing import List, Dict, Optional
from tqdm import tqdm
import random
import numpy as np

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

    def _generar_pregunta_desde_hecho(self, hecho: Dict) -> Optional[Dict]:
        """Genera una pregunta analítica a partir de un hecho"""
        try:
            # Determinar el aspecto a preguntar
            aspecto = self._determinar_aspecto_pregunta(hecho)
            
            # Generar la pregunta usando QA
            pregunta = self.qa_model(
                question=aspecto['pregunta_base'],
                context=hecho['contexto']
            )
            
            if pregunta['score'] < 0.6:
                return None
                
            # Extraer y validar la respuesta
            respuesta = self._extraer_respuesta(pregunta, hecho)
            if not respuesta:
                return None
                
            return {
                'tipo': aspecto['tipo'],
                'pregunta': aspecto['pregunta_base'],
                'respuesta': respuesta,
                'contexto': hecho['contexto']
            }
            
        except Exception as e:
            print(f"Error generando pregunta: {str(e)}")
            return None

    def _determinar_aspecto_pregunta(self, hecho: Dict) -> Dict:
        """Determina el aspecto más relevante para preguntar sobre el hecho"""
        # Analizar el tipo de información disponible
        if hecho['objeto']:
            # Relación sujeto-objeto
            return {
                'tipo': 'relacion',
                'pregunta_base': f"¿Qué relación existe entre {hecho['sujeto']} y {hecho['objeto']}?"
            }
        elif hecho['atributos']:
            # Características o propiedades
            return {
                'tipo': 'caracteristica',
                'pregunta_base': f"¿Qué característica define a {hecho['sujeto']}?"
            }
        else:
            # Función o propósito
            return {
                'tipo': 'funcion',
                'pregunta_base': f"¿Cuál es la función de {hecho['sujeto']}?"
            }

    def _extraer_respuesta(self, pregunta: Dict, hecho: Dict) -> Optional[str]:
        """Extrae y valida la respuesta del modelo QA"""
        respuesta = pregunta['answer']
        
        # Validar que la respuesta es significativa
        doc_respuesta = self.nlp(respuesta)
        if len(doc_respuesta) < 2 or all(token.is_stop for token in doc_respuesta):
            return None
            
        # Validar coherencia con el contexto
        similitud = doc_respuesta.similarity(self.nlp(hecho['contexto']))
        if similitud < 0.3:
            return None
            
        return respuesta

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

    def generar_alternativas(self, respuesta: str, tipo: str, contexto: str) -> List[str]:
        """Genera alternativas coherentes basadas en la respuesta correcta"""
        doc_respuesta = self.nlp(respuesta)
        doc_contexto = self.nlp(contexto)
        
        # Identificar la categoría semántica de la respuesta
        categoria = self._identificar_categoria(doc_respuesta)
        
        # Generar alternativas según la categoría
        if categoria:
            alternativas = self._generar_por_categoria(
                respuesta,
                categoria,
                doc_respuesta,
                doc_contexto
            )
        else:
            alternativas = self._generar_por_contexto(
                respuesta,
                doc_respuesta,
                doc_contexto
            )
            
        # Filtrar y validar alternativas
        alternativas_validadas = self._validar_alternativas(
            alternativas,
            respuesta,
            tipo
        )
        
        return alternativas_validadas[:3]  # Devolver las 3 mejores

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

    def _generar_pregunta(self, oracion: str) -> Optional[Dict]:
        """Genera una pregunta y su respuesta"""
        try:
            # Obtener sujetos principales
            doc = self.nlp(oracion)
            sujetos = []
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) >= 2:  # Solo frases nominales
                    sujetos.append(chunk.text)

            if not sujetos:
                return None

            # Seleccionar un sujeto y generar pregunta
            sujeto = random.choice(sujetos)
            pregunta_base = f"¿Qué características o función tiene {sujeto}?"

            # Obtener respuesta
            respuesta = self.qa_model(
                question=pregunta_base,
                context=oracion
            )

            if respuesta['score'] < 0.3:  # Umbral bajo para pruebas
                return None

            # Buscar otras frases nominales para alternativas
            otras_frases = [chunk.text for chunk in doc.noun_chunks 
                          if chunk.text != sujeto and 
                          chunk.text != respuesta['answer'] and
                          len(chunk.text.split()) >= 2]

            if len(otras_frases) < 3:
                return None

            # Formar alternativas
            alternativas = random.sample(otras_frases, 3)
            alternativas.append(respuesta['answer'])
            random.shuffle(alternativas)

            print(f"Pregunta generada: {pregunta_base}")  # Debug
            print(f"Respuesta: {respuesta['answer']}")    # Debug
            print(f"Alternativas: {alternativas}")        # Debug

            return {
                'pregunta': pregunta_base,
                'opciones': alternativas,
                'respuesta_correcta': respuesta['answer'],
                'tipo': 'alternativas'
            }

        except Exception as e:
            print(f"Error en generación de pregunta: {str(e)}")
            return None

    def _generar_verdadero_falso(self, oracion: str) -> Optional[Dict]:
        """Genera una pregunta de verdadero/falso"""
        try:
            doc = self.nlp(oracion)
            es_verdadero = random.choice([True, False])

            if es_verdadero:
                print(f"V/F generada (V): {oracion}")  # Debug
                return {
                    'pregunta': oracion,
                    'respuesta_correcta': 'Verdadero',
                    'tipo': 'verdadero_falso'
                }
            else:
                # Modificar un sustantivo o verbo
                tokens = list(doc)
                for i, token in enumerate(tokens):
                    if token.pos_ in ['NOUN', 'VERB']:
                        # Reemplazar con otra palabra
                        reemplazo = random.choice(['diferente', 'otro', 'distinto'])
                        tokens[i] = self.nlp(reemplazo)[0]
                        break

                oracion_modificada = ' '.join(t.text for t in tokens)
                print(f"V/F generada (F): {oracion_modificada}")  # Debug
                return {
                    'pregunta': oracion_modificada,
                    'respuesta_correcta': 'Falso',
                    'tipo': 'verdadero_falso'
                }

        except Exception as e:
            print(f"Error en generación de V/F: {str(e)}")
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
                    pregunta = self._generar_verdadero_falso(oracion)
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