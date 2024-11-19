# Generador de Cuestionarios
# Por Franco Benassi
import spacy
import random
import json
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Tuple

class AnalizadorSemantico:
    def __init__(self):
        self.nlp = spacy.load('es_core_news_lg')
        
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 3),
            stop_words=['spanish']
        )
        
    def extraer_temas_principales(self, texto: str) -> List[Dict[str, Any]]:
        """Extrae temas principales usando LDA"""
        # Preprocesar texto
        doc = self.nlp(texto)
        oraciones = [sent.text for sent in doc.sents]
        
        # Vectorizar texto
        vectores = self.vectorizer.fit_transform(oraciones)
        
        # Aplicar LDA
        lda = LatentDirichletAllocation(n_components=5, random_state=42)
        temas = lda.fit_transform(vectores)
        
        # Extraer palabras clave por tema
        feature_names = self.vectorizer.get_feature_names_out()
        temas_principales = []
        
        for tema_idx, tema in enumerate(lda.components_):
            top_palabras = [
                feature_names[i]
                for i in tema.argsort()[:-10:-1]
            ]
            temas_principales.append({
                'id': tema_idx,
                'palabras_clave': top_palabras,
                'peso': np.mean(temas[:, tema_idx])
            })
            
        return temas_principales
        
    def analizar_estructura_semantica(self, texto: str) -> Dict[str, Any]:
        """Analiza la estructura semántica del texto"""
        doc = self.nlp(texto)
        
        # Identificar conceptos clave
        conceptos = defaultdict(list)
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN']:
                # Encontrar modificadores
                modificadores = []
                for hijo in token.children:
                    if hijo.dep_ in ['amod', 'nmod']:
                        modificadores.append(hijo.text)
                        
                concepto = {
                    'texto': token.text,
                    'pos': token.pos_,
                    'modificadores': modificadores,
                    'importancia': token.dep_ in ['nsubj', 'dobj']
                }
                conceptos[token.lemma_].append(concepto)
                
        # Identificar relaciones
        relaciones = []
        for token in doc:
            if token.dep_ == 'ROOT' and token.pos_ == 'VERB':
                for hijo in token.children:
                    if hijo.dep_ in ['nsubj', 'dobj']:
                        relacion = {
                            'sujeto': hijo.text,
                            'verbo': token.text,
                            'objeto': next((c.text for c in token.children 
                                         if c.dep_ == 'dobj'), None)
                        }
                        relaciones.append(relacion)
                        
        # Identificar estructuras causales
        causales = []
        for token in doc:
            if token.dep_ == 'prep' and token.text in ['porque', 'ya que', 'debido a']:
                causa = []
                for r in token.rights:
                    causa.extend([t.text for t in r.subtree])
                causales.append(' '.join(causa))
                
        return {
            'conceptos': conceptos,
            'relaciones': relaciones, 
            'causales': causales,
            'oraciones': [sent.text for sent in doc.sents]  # Agregar esta línea
        }
    
    def identificar_jerarquia_conceptos(self, texto: str) -> Dict[str, List[str]]:
        """Identifica jerarquías entre conceptos"""
        doc = self.nlp(texto)
        jerarquias = defaultdict(list)
        
        for token in doc:
            if token.dep_ == 'nmod':
                padre = token.head.text
                hijo = token.text
                jerarquias[padre].append(hijo)
                
        return dict(jerarquias)
    
    def calcular_similitud_semantica(self, texto1: str, texto2: str) -> float:
        """Calcula la similitud semántica entre dos textos"""
        doc1 = self.nlp(texto1)
        doc2 = self.nlp(texto2)
        
        # Comparar embeddings
        similitud = doc1.similarity(doc2)
        
        # Comparar estructura sintáctica
        similitud_pos = self._comparar_estructura_pos(doc1, doc2)
        
        # Comparar entidades nombradas
        similitud_ents = self._comparar_entidades(doc1, doc2)
        
        # Combinar métricas
        return np.mean([similitud, similitud_pos, similitud_ents])
    
    def _comparar_estructura_pos(self, doc1, doc2) -> float:
        """Compara estructura POS entre documentos"""
        pos1 = [token.pos_ for token in doc1]
        pos2 = [token.pos_ for token in doc2]
        
        # Calcular secuencias comunes más largas
        matriz = np.zeros((len(pos1) + 1, len(pos2) + 1))
        for i, x in enumerate(pos1, 1):
            for j, y in enumerate(pos2, 1):
                if x == y:
                    matriz[i,j] = matriz[i-1,j-1] + 1
                    
        max_len = np.max(matriz)
        return max_len / max(len(pos1), len(pos2))
    
    def _comparar_entidades(self, doc1, doc2) -> float:
        """Compara entidades nombradas entre documentos"""
        ents1 = set([(ent.text, ent.label_) for ent in doc1.ents])
        ents2 = set([(ent.text, ent.label_) for ent in doc2.ents])
        
        if not ents1 or not ents2:
            return 0.0
            
        interseccion = len(ents1.intersection(ents2))
        union = len(ents1.union(ents2))
        
        return interseccion / union
    
    def obtener_contexto_semantico(self, sujeto: str, verbo: str) -> Dict[str, Any]:
        """Analiza el contexto semántico de una relación sujeto-verbo"""
        doc_sujeto = self.nlp(sujeto)
        doc_verbo = self.nlp(verbo)
        
        # Obtener clusters semánticos relacionados
        cluster_sujeto = self._obtener_cluster_semantico(doc_sujeto)
        cluster_verbo = self._obtener_cluster_semantico(doc_verbo)
        
        return {
            'descripcion': self._generar_descripcion_contexto(cluster_sujeto, cluster_verbo),
            'conceptos_relacionados': cluster_sujeto,
            'acciones_relacionadas': cluster_verbo
        }
        
    def _obtener_cluster_semantico(self, doc) -> List[str]:
        """Obtiene palabras semánticamente relacionadas"""
        cluster = []
        for token in doc:
            # Encontrar sinónimos y palabras relacionadas
            similares = [
                t.text for t in token.vocab 
                if t.has_vector and token.similarity(t) > 0.7
            ][:5]
            cluster.extend(similares)
            
        return list(set(cluster))
        
    def _generar_descripcion_contexto(self, cluster_sujeto: List[str], 
                                    cluster_verbo: List[str]) -> str:
        """Genera una descripción contextual"""
        return f"Contexto relacionado con {', '.join(cluster_sujeto)} " \
               f"en términos de {', '.join(cluster_verbo)}"
    
class GeneradorPreguntasDinamico:
    def __init__(self, analizador: AnalizadorSemantico):
        self.analizador = analizador
        self.nlp = spacy.load('es_core_news_lg')

    def generar_preguntas(self, texto: str) -> List[Dict[str, Any]]:
        """Método principal que usa el analizador para generar preguntas"""
        # Obtener análisis completo del texto
        temas = self.analizador.extraer_temas_principales(texto)
        estructura = self.analizador.analizar_estructura_semantica(texto)
        jerarquia = self.analizador.identificar_jerarquia_conceptos(texto)
        
        preguntas = []
        
        # Generar preguntas basadas en temas principales
        for tema in temas:
            pregunta = self._generar_pregunta_tema(tema, estructura)
            if pregunta:
                preguntas.append(pregunta)

        # Generar preguntas basadas en relaciones semánticas
        for concepto, relaciones in estructura['relaciones'].items():
            pregunta = self._generar_pregunta_relacion(concepto, relaciones, estructura)
            if pregunta:
                preguntas.append(pregunta)

        # Generar preguntas basadas en jerarquía de conceptos
        for concepto_padre, conceptos_hijos in jerarquia.items():
            pregunta = self._generar_pregunta_jerarquia(concepto_padre, conceptos_hijos, estructura)
            if pregunta:
                preguntas.append(pregunta)

        return preguntas

    def _calcular_densidad_informativa(self, oracion) -> float:
        """Calcula qué tan informativa es una oración"""
        palabras_contenido = 0
        total_palabras = len([token for token in oracion])
        
        for token in oracion:
            # Considerar palabras con significado
            if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN']:
                # Pesar según importancia
                peso = self._calcular_peso_token(token)
                palabras_contenido += peso

        return palabras_contenido / total_palabras if total_palabras > 0 else 0

    def _calcular_peso_token(self, token) -> float:
        """Calcula el peso semántico de un token"""
        pesos = {
            'NOUN': 1.0,
            'PROPN': 1.0,
            'VERB': 0.8,
            'ADJ': 0.6
        }
        peso_base = pesos.get(token.pos_, 0.3)
        
        # Ajustar según dependencia sintáctica
        if token.dep_ in ['ROOT', 'nsubj', 'dobj']:
            peso_base *= 1.5
            
        return peso_base

    def _analizar_estructura(self, oracion) -> Dict[str, Any]:
        """Analiza la estructura semántica de una oración"""
        elementos = {
            'sujetos': [],
            'verbos': [],
            'objetos': [],
            'circunstancias': []
        }
        
        for token in oracion:
            if token.dep_ == 'nsubj':
                elementos['sujetos'].append({
                    'texto': token.text,
                    'subtree': [t.text for t in token.subtree]
                })
            elif token.dep_ == 'ROOT' and token.pos_ == 'VERB':
                elementos['verbos'].append({
                    'texto': token.text,
                    'tiempo': token.morph.get('Tense', [''])[0],
                    'modo': token.morph.get('Mood', [''])[0]
                })
            elif token.dep_ in ['dobj', 'iobj']:
                elementos['objetos'].append({
                    'texto': token.text,
                    'tipo': token.dep_
                })
            elif token.dep_ in ['advmod', 'prep']:
                elementos['circunstancias'].append({
                    'texto': token.text,
                    'tipo': token.pos_
                })
                
        return elementos

    def _construir_pregunta(self, oracion_base: Dict, analisis: Dict) -> Dict[str, Any]:
        estructura = oracion_base['estructura']
        
        # Determinar tipo de pregunta basado en la estructura
        if len(estructura['sujetos']) > 0 and len(estructura['verbos']) > 0:
            return self._generar_pregunta_proceso(estructura, analisis)
        elif len(estructura['objetos']) > 1:
            return self._generar_pregunta_relacion(estructura, analisis)
        else:
            return self._generar_pregunta_concepto(estructura, analisis)

    def _generar_pregunta_proceso(self, estructura: Dict, analisis: Dict) -> Dict[str, Any]:
        """Genera una pregunta sobre un proceso o acción"""
        verbo_principal = estructura['verbos'][0]
        sujeto = estructura['sujetos'][0]
        
        # Analizar el contexto semántico
        contexto = self.analizador.obtener_contexto_semantico(
            sujeto['texto'], 
            verbo_principal['texto']
        )
        
        # Construir pregunta basada en el contexto
        elementos_pregunta = self._extraer_elementos_pregunta(contexto)
        pregunta = self._componer_pregunta(elementos_pregunta)
        
        return {
            'tipo': 'alternativas',
            'pregunta': pregunta,
            'contexto': contexto
        }

    def _generar_pregunta_tema(self, tema: Dict[str, Any], estructura: Dict[str, Any]) -> Dict[str, Any]:
        """Genera pregunta basada en tema principal"""
        # Usar palabras clave del tema
        palabras_clave = tema['palabras_clave']
        
        # Encontrar oraciones relevantes usando el analizador
        oraciones_relevantes = []
        for oracion in estructura['oraciones']:
            similitud = self.analizador.calcular_similitud_semantica(
                ' '.join(palabras_clave),
                oracion
            )
            if similitud > 0.5:
                oraciones_relevantes.append(oracion)

        if not oraciones_relevantes:
            return None

        # Analizar estructura de la oración más relevante
        oracion = self.nlp(random.choice(oraciones_relevantes))
        oracion_analizada = self._analizar_estructura(oracion)
        elementos = {'estructura': oracion_analizada}
        
        # Construir pregunta basada en elementos identificados
        pregunta = self._construir_pregunta(elementos)
        
        return pregunta if pregunta else None

    def _generar_pregunta_relacion(self, concepto: str, relaciones: List[Dict], 
                                estructura: Dict[str, Any]) -> Dict[str, Any]:
        """Genera pregunta basada en relaciones semánticas"""
        # Encontrar relación más significativa
        relacion_principal = max(relaciones, key=lambda x: x.get('importancia', 0))
        
        # Analizar contexto de la relación
        contexto = self._analizar_contexto_relacion(concepto, relacion_principal, estructura)
        
        # Construir pregunta basada en el análisis
        elementos_pregunta = self._extraer_elementos_pregunta(contexto)
        pregunta = self._componer_pregunta(elementos_pregunta)
        
        return {
            'tipo': 'alternativas',
            'pregunta': pregunta,
            'contexto': contexto
        } if pregunta else None

    def _generar_pregunta_jerarquia(self, concepto_padre: str, conceptos_hijos: List[str],
                                 estructura: Dict[str, Any]) -> Dict[str, Any]:
        """Genera pregunta basada en jerarquía de conceptos"""
        # Analizar relaciones jerárquicas
        relaciones = self._analizar_relaciones_jerarquicas(
            concepto_padre,
            conceptos_hijos,
            estructura
        )
        
        if not relaciones:
            return None
            
        # Construir pregunta basada en la jerarquía
        elementos = {
            'concepto_principal': concepto_padre,
            'conceptos_relacionados': conceptos_hijos,
            'tipo_relacion': relaciones['tipo'],
            'contexto': relaciones['contexto']
        }
        
        pregunta = self._construir_pregunta_jerarquica(elementos)
        
        return pregunta if pregunta else None

    def _extraer_elementos_pregunta(self, contexto: Dict) -> Dict[str, str]:
        """Extrae elementos para construir la pregunta"""
        doc = self.nlp(contexto['descripcion'])
        
        elementos = {
            'accion': None,
            'entidad': None,
            'aspecto': None
        }
        
        for token in doc:
            if token.pos_ == 'VERB' and not elementos['accion']:
                elementos['accion'] = token.text
            elif token.pos_ in ['NOUN', 'PROPN'] and not elementos['entidad']:
                elementos['entidad'] = token.text
            elif token.pos_ == 'ADJ' and not elementos['aspecto']:
                elementos['aspecto'] = token.text
                
        return elementos

    def _componer_pregunta(self, elementos: Dict[str, str]) -> str:
        """Compone dinámicamente una pregunta basada en elementos semánticos"""
        doc_elementos = self.nlp(' '.join(filter(None, elementos.values())))
        
        # Analizar estructura y generar pregunta
        estructura_pregunta = {}
        for token in doc_elementos:
            if token.pos_ == 'VERB':
                estructura_pregunta['verbo'] = token.lemma_
            elif token.pos_ in ['NOUN', 'PROPN']:
                if 'sujeto' not in estructura_pregunta:
                    estructura_pregunta['sujeto'] = token.text
                else:
                    estructura_pregunta['objeto'] = token.text
                    
        # Componer pregunta según estructura identificada
        pregunta = self._estructurar_pregunta(estructura_pregunta)
        
        return pregunta

    def _estructurar_pregunta(self, estructura: Dict[str, str]) -> str:
        """Estructura la pregunta final basada en análisis lingüístico"""
        doc = self.nlp(' '.join(estructura.values()))
        tokens_pregunta = []
        
        # Reorganizar elementos según reglas gramaticales
        for token in doc:
            if token.pos_ == 'VERB':
                tokens_pregunta.insert(0, token.text)
            else:
                tokens_pregunta.append(token.text)
                
        pregunta = ' '.join(tokens_pregunta) + '?'
        return pregunta.capitalize()
    
    def _analizar_contexto_relacion(self, concepto: str, relacion: Dict, 
                                  estructura: Dict) -> Dict[str, Any]:
        """Analiza el contexto completo de una relación"""
        # Obtener todas las oraciones que mencionan el concepto
        doc = self.nlp(concepto)
        menciones = []
        
        for sent in estructura.get('oraciones', []):
            doc_sent = self.nlp(sent)
            if doc.similarity(doc_sent) > 0.6:
                menciones.append({
                    'texto': sent,
                    'similitud': doc.similarity(doc_sent)
                })
                
        # Ordenar por relevancia
        menciones_ordenadas = sorted(menciones, 
                                   key=lambda x: x['similitud'], 
                                   reverse=True)
        
        return {
            'descripcion': menciones_ordenadas[0]['texto'] if menciones_ordenadas else '',
            'menciones_relacionadas': menciones_ordenadas,
            'tipo_relacion': relacion.get('tipo', 'indefinida')
        }

    def _analizar_relaciones_jerarquicas(self, concepto_padre: str, 
                                       conceptos_hijos: List[str],
                                       estructura: Dict) -> Dict[str, Any]:
        """Analiza las relaciones jerárquicas entre conceptos"""
        # Determinar tipo de jerarquía
        tipo_relacion = self._determinar_tipo_jerarquia(
            concepto_padre, 
            conceptos_hijos,
            estructura
        )
        
        # Encontrar contexto que explica la relación
        contexto = self._encontrar_contexto_jerarquia(
            concepto_padre,
            conceptos_hijos,
            estructura
        )
        
        return {
            'tipo': tipo_relacion,
            'contexto': contexto,
            'nivel': self._determinar_nivel_jerarquico(concepto_padre, estructura)
        }
        
    def _determinar_tipo_jerarquia(self, padre: str, hijos: List[str], 
                                 estructura: Dict) -> str:
        """Determina el tipo de relación jerárquica"""
        doc_padre = self.nlp(padre)
        
        # Analizar patrones de relación
        patrones = {
            'categoria': ['tipo', 'clase', 'grupo', 'categoría'],
            'parte': ['parte', 'componente', 'elemento'],
            'proceso': ['paso', 'etapa', 'fase']
        }
        
        for tipo, indicadores in patrones.items():
            if any(ind in padre.lower() for ind in indicadores):
                return tipo
                
        return 'general'
        
    def _encontrar_contexto_jerarquia(self, padre: str, hijos: List[str],
                                    estructura: Dict) -> str:
        """Encuentra el contexto que explica la relación jerárquica"""
        oraciones = estructura.get('oraciones', [])
        
        # Buscar oraciones que mencionan tanto al padre como a algún hijo
        contextos = []
        for oracion in oraciones:
            if padre.lower() in oracion.lower() and \
               any(hijo.lower() in oracion.lower() for hijo in hijos):
                contextos.append(oracion)
                
        return max(contextos, key=len) if contextos else ''
        
    def _determinar_nivel_jerarquico(self, concepto: str, estructura: Dict) -> int:
        """Determina el nivel jerárquico de un concepto"""
        jerarquia = estructura.get('jerarquia', {})
        nivel = 0
        
        # Buscar nivel en la jerarquía
        for padre, hijos in jerarquia.items():
            if concepto in hijos:
                nivel = self._determinar_nivel_jerarquico(padre, estructura) + 1
                break
                
        return nivel
        
    def _construir_pregunta_jerarquica(self, elementos: Dict) -> Dict[str, Any]:
        """Construye una pregunta basada en relación jerárquica"""
        concepto = elementos['concepto_principal']
        relacionados = elementos['conceptos_relacionados']
        tipo = elementos['tipo_relacion']
        
        doc = self.nlp(elementos['contexto'])
        
        # Identificar verbos de relación
        verbos = [token.text for token in doc if token.pos_ == 'VERB']
        
        if not verbos:
            return None
            
        estructura_pregunta = {
            'sujeto': concepto,
            'verbo': random.choice(verbos),
            'objeto': ', '.join(relacionados[:2])  # Limitar a 2 conceptos relacionados
        }
        
        return {
            'tipo': 'alternativas',
            'pregunta': self._estructurar_pregunta(estructura_pregunta)
        }
    
class GeneradorAlternativasExplicaciones:
    def __init__(self, analizador: AnalizadorSemantico):
        self.analizador = analizador
        self.nlp = spacy.load('es_core_news_lg')
        self.contexto_actual = "" 

    def _extraer_informacion_relevante(self, doc: spacy.tokens.Doc) -> Dict[str, Any]:
        """Extrae información relevante del contexto"""
        info = {
            'conceptos_principales': [],
            'relaciones': [],
            'detalles_importantes': []
        }
        
        # Identificar conceptos principales
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN'] and token.dep_ in ['nsubj', 'dobj']:
                concepto = {
                    'texto': token.text,
                    'importancia': self._calcular_importancia_concepto(token),
                    'contexto': self._obtener_contexto_concepto(token)
                }
                info['conceptos_principales'].append(concepto)
                
        # Identificar relaciones
        for token in doc:
            if token.dep_ == 'ROOT' and token.pos_ == 'VERB':
                relacion = self._extraer_relacion(token)
                if relacion:
                    info['relaciones'].append(relacion)
                    
        # Identificar detalles importantes
        for ent in doc.ents:
            if ent.label_ in ['DATE', 'ORG', 'GPE', 'CARDINAL']:
                info['detalles_importantes'].append({
                    'texto': ent.text,
                    'tipo': ent.label_
                })
                
        return info

    def _componer_explicacion_desde_info(self, info: Dict[str, Any], es_correcta: bool) -> str:
        """Compone explicación usando información extraída"""
        elementos_explicacion = []
        
        # Usar conceptos principales
        if info['conceptos_principales']:
            concepto = max(info['conceptos_principales'], 
                        key=lambda x: x['importancia'])
            contexto = concepto['contexto']
            elementos_explicacion.append(
                self._generar_oracion_concepto(concepto, contexto)
            )
            
        # Usar relaciones relevantes
        if info['relaciones']:
            relacion = info['relaciones'][0]  # Tomar la más relevante
            elementos_explicacion.append(
                self._generar_oracion_relacion(relacion)
            )
            
        # Agregar detalles
        if info['detalles_importantes']:
            elementos_explicacion.append(
                self._generar_oracion_detalles(info['detalles_importantes'])
            )
            
        # Componer explicación final
        if elementos_explicacion:
            return ' '.join(elementos_explicacion)
        else:
            return "No hay suficiente información para generar una explicación detallada."

    def _calcular_importancia_relacion(self, token: spacy.tokens.Token) -> float:
        """Calcula la importancia de una relación"""
        importancia = 1.0
        
        # Factores que aumentan importancia
        if token.dep_ == 'ROOT':
            importancia *= 1.5
        
        # Considerar número de dependientes
        num_deps = len(list(token.children))
        importancia *= (1 + (num_deps * 0.1))
        
        # Considerar posición en la oración
        pos_weight = 1 - (token.i / len(token.doc))
        importancia *= (1 + pos_weight)
        
        return importancia

    def _extraer_relacion(self, token: spacy.tokens.Token) -> Dict[str, str]:
        """Extrae relación completa de un token verbal"""
        relacion = {
            'verbo': token.text,
            'sujeto': None,
            'objeto': None,
            'modificadores': []
        }
        
        for hijo in token.children:
            if hijo.dep_ == 'nsubj':
                relacion['sujeto'] = hijo.text
            elif hijo.dep_ == 'dobj':
                relacion['objeto'] = hijo.text
            elif hijo.dep_ in ['advmod', 'amod']:
                relacion['modificadores'].append(hijo.text)
                
        if relacion['sujeto'] and relacion['objeto']:
            return relacion
        return None

    def _extraer_caracteristicas(self, concepto: Dict[str, Any], 
                            contexto: Dict[str, Any]) -> List[str]:
        """Extrae características distintivas de un concepto"""
        doc = self.nlp(concepto['texto'])
        caracteristicas = []
        
        # Buscar modificadores directos
        for token in doc[0].children:
            if token.dep_ in ['amod', 'nmod']:
                caracteristicas.append(token.text)
                
        # Buscar en el contexto más amplio
        doc_contexto = self.nlp(contexto.get('descripcion', ''))
        for sent in doc_contexto.sents:
            if concepto['texto'] in sent.text:
                # Extraer adjetivos y modificadores
                for token in sent:
                    if token.pos_ == 'ADJ' and token.head.text == concepto['texto']:
                        caracteristicas.append(token.text)
                        
        return list(set(caracteristicas))

    def _son_similares(self, caract1: str, caract2: str) -> bool:
        """Determina si dos características son semánticamente similares"""
        doc1 = self.nlp(caract1)
        doc2 = self.nlp(caract2)
        
        # Calcular similitud
        similitud = doc1.similarity(doc2)
        return similitud > 0.7

    def _buscar_conexiones(self, concepto1: str, concepto2: str, 
                        contexto: Dict[str, Any]) -> List[str]:
        """Busca conexiones entre dos conceptos en el contexto"""
        conexiones = []
        doc_contexto = self.nlp(contexto.get('descripcion', ''))
        
        # Buscar oraciones que mencionan ambos conceptos
        for sent in doc_contexto.sents:
            if concepto1 in sent.text and concepto2 in sent.text:
                # Analizar la naturaleza de la conexión
                conexion = self._analizar_conexion(sent, concepto1, concepto2)
                if conexion:
                    conexiones.append(conexion)
                    
        return conexiones

    def _determinar_tipo_relacion(self, conexiones: List[str]) -> str:
        """Determina el tipo de relación basado en las conexiones"""
        if not conexiones:
            return "indefinida"
            
        # Analizar palabras clave en las conexiones
        palabras_clave = {
            'causal': ['causa', 'produce', 'genera', 'resulta'],
            'temporal': ['antes', 'después', 'durante', 'mientras'],
            'espacial': ['dentro', 'fuera', 'cerca', 'lejos'],
            'comparativa': ['más', 'menos', 'mejor', 'peor'],
            'funcional': ['sirve', 'funciona', 'permite', 'facilita']
        }
        
        conteo_tipos = defaultdict(int)
        for conexion in conexiones:
            doc = self.nlp(conexion.lower())
            for tipo, palabras in palabras_clave.items():
                if any(palabra in doc.text for palabra in palabras):
                    conteo_tipos[tipo] += 1
                    
        if conteo_tipos:
            return max(conteo_tipos.items(), key=lambda x: x[1])[0]
        return "asociativa"

    def _generar_oracion_concepto(self, concepto: Dict[str, Any], 
                            contexto: Dict[str, Any]) -> str:
        """Genera una oración describiendo un concepto"""
        if not contexto:
            return ""
            
        doc = self.nlp(concepto['texto'])
        caracteristicas = []
        
        # Extraer información relevante
        for token in doc[0].children:
            if token.dep_ in ['amod', 'compound']:
                caracteristicas.append(token.text)
                
        # Construir oración dinámicamente
        elementos = []
        elementos.append(concepto['texto'])
        
        if caracteristicas:
            elementos.append("se caracteriza por")
            elementos.append(" y ".join(caracteristicas))
            
        if contexto.get('verbos_asociados'):
            elementos.append("y")
            elementos.append(random.choice(contexto['verbos_asociados']))
            
        return " ".join(elementos)

    def _generar_oracion_relacion(self, relacion: Dict[str, str]) -> str:
        """Genera una oración describiendo una relación"""
        elementos = [relacion['sujeto']]
        
        if relacion['modificadores']:
            elementos.append(random.choice(relacion['modificadores']))
            
        elementos.append(relacion['verbo'])
        
        if relacion['objeto']:
            elementos.append(relacion['objeto'])
            
        return " ".join(elementos)

    def _generar_oracion_detalles(self, detalles: List[Dict[str, Any]]) -> str:
        """Genera una oración con detalles importantes"""
        elementos = []
        
        for detalle in detalles:
            if detalle['tipo'] == 'DATE':
                elementos.append(f"en {detalle['texto']}")
            elif detalle['tipo'] == 'ORG':
                elementos.append(f"por parte de {detalle['texto']}")
            elif detalle['tipo'] == 'CARDINAL':
                elementos.append(f"con {detalle['texto']}")
                
        if elementos:
            return " ".join(elementos)
        return ""

    def _obtener_contexto_concepto(self, token: spacy.tokens.Token) -> Dict[str, Any]:
        """Obtiene contexto detallado de un concepto"""
        return {
            'modificadores': [t.text for t in token.children if t.dep_ in ['amod', 'nmod']],
            'verbos_asociados': [t.head.text for t in token.ancestors if t.pos_ == 'VERB'],
            'entidades_relacionadas': [e.text for e in token.doc.ents if token in e]
        }

    def _analizar_conexion(self, sent: spacy.tokens.Span, 
                        concepto1: str, concepto2: str) -> str:
        """Analiza la naturaleza de la conexión entre conceptos en una oración"""
        doc = self.nlp(sent.text)
        
        # Encontrar los tokens de los conceptos
        token1 = None
        token2 = None
        for token in doc:
            if token.text == concepto1:
                token1 = token
            elif token.text == concepto2:
                token2 = token
                
        if not (token1 and token2):
            return None
            
        # Analizar la ruta sintáctica entre los conceptos
        ruta = []
        for token in doc:
            if token in token1.subtree and token in token2.subtree:
                ruta.append(token.text)
                
        return " ".join(ruta) if ruta else None

    def generar_alternativas(self, pregunta: Dict[str, Any], texto: str) -> List[str]:
        """Genera alternativas para una pregunta basada en análisis semántico"""
        # Obtener contexto relevante
        contexto = self._extraer_contexto_relevante(pregunta, texto)
        
        # Analizar la respuesta correcta
        respuesta_correcta = pregunta['contexto']['descripcion']
        doc_respuesta = self.nlp(respuesta_correcta)
        
        # Generar alternativas según el tipo de pregunta
        if pregunta['tipo'] == 'alternativas':
            return self._generar_alternativas_multiple(doc_respuesta, contexto)
        else:  # verdadero_falso
            return self._generar_alternativas_vf(doc_respuesta, contexto)
            
    def _extraer_contexto_relevante(self, pregunta: Dict[str, Any], texto: str) -> List[str]:
        """Extrae oraciones relevantes al contexto de la pregunta"""
        doc_pregunta = self.nlp(pregunta['pregunta'])
        doc_texto = self.nlp(texto)
        
        # Encontrar oraciones relacionadas
        oraciones_relevantes = []
        for sent in doc_texto.sents:
            similitud = doc_pregunta.similarity(sent)
            if similitud > 0.5:  # Umbral ajustable
                oraciones_relevantes.append({
                    'texto': sent.text,
                    'similitud': similitud
                })
                
        # Ordenar por relevancia
        return sorted(oraciones_relevantes, key=lambda x: x['similitud'], reverse=True)

    def _generar_alternativas_multiple(self, doc_respuesta: spacy.tokens.Doc, 
                                 contexto: List[Dict[str, Any]]) -> List[str]:
        alternativas = []
        
        # 1. Alternativas por sustitución de conceptos clave
        conceptos_clave = self._identificar_conceptos_clave(doc_respuesta)
        for concepto in conceptos_clave:
            alt = self._generar_por_sustitucion(concepto, contexto)
            if alt:
                alternativas.append(alt)

        # 2. Alternativas por modificación de relaciones
        relaciones = self._identificar_relaciones(doc_respuesta)
        for relacion in relaciones:
            alt = self._generar_por_modificacion_relacion(relacion, contexto)
            if alt:
                alternativas.append(alt)

        # 3. Alternativas por variación semántica
        alt_semanticas = self._generar_por_variacion_semantica(doc_respuesta, contexto)
        alternativas.extend(alt_semanticas)

        # Filtrar y seleccionar mejores alternativas
        alternativas = self._filtrar_alternativas(alternativas, doc_respuesta)
        
        return alternativas[:3]  # Retornar las 3 mejores alternativas

    def _identificar_conceptos_clave(self, doc: spacy.tokens.Doc) -> List[Dict[str, Any]]:
        """Identifica conceptos clave en el texto"""
        conceptos = []
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN'] and len(token.text) > 3:
                # Analizar contexto del concepto
                contexto = self._analizar_contexto_concepto(token)
                conceptos.append({
                    'texto': token.text,
                    'tipo': token.pos_,
                    'contexto': contexto,
                    'importancia': self._calcular_importancia_concepto(token)
                })
        return sorted(conceptos, key=lambda x: x['importancia'], reverse=True)

    def _analizar_contexto_concepto(self, token: spacy.tokens.Token) -> Dict[str, Any]:
        """Analiza el contexto lingüístico de un concepto"""
        return {
            'modificadores': [t.text for t in token.children if t.dep_ in ['amod', 'nmod']],
            'verbos_asociados': [t.head.text for t in token.ancestors if t.pos_ == 'VERB'],
            'entidades_relacionadas': [e.text for e in token.doc.ents if token in e]
        }

    def _calcular_importancia_concepto(self, token: spacy.tokens.Token) -> float:
        """Calcula la importancia de un concepto"""
        importancia = 1.0
        
        # Factores que aumentan importancia
        if token.dep_ in ['nsubj', 'dobj']:
            importancia *= 1.5
        if any(t.dep_ == 'ROOT' for t in token.ancestors):
            importancia *= 1.3
        if token.ent_type_:
            importancia *= 1.2
        if len(list(token.children)) > 2:
            importancia *= 1.1
            
        return importancia

    def _generar_por_sustitucion(self, concepto: Dict[str, Any], 
                                contexto: List[Dict[str, Any]]) -> str:
        """Genera alternativa sustituyendo conceptos clave"""
        # Encontrar conceptos similares en el contexto
        similares = self._encontrar_conceptos_similares(
            concepto['texto'],
            [c['texto'] for c in contexto]
        )
        
        if not similares:
            return None
            
        # Seleccionar concepto similar
        concepto_similar = random.choice(similares)
        
        # Construir alternativa manteniendo estructura
        return self._construir_alternativa_con_sustitucion(
            concepto['texto'],
            concepto_similar,
            concepto['contexto']
        )

    def _generar_por_modificacion_relacion(self, relacion: Dict[str, Any],
                                         contexto: List[Dict[str, Any]]) -> str:
        """Genera alternativa modificando relaciones entre conceptos"""
        # Analizar la relación original
        sujeto = relacion['sujeto']
        verbo = relacion['verbo']
        objeto = relacion['objeto']
        
        # Buscar relaciones alternativas en el contexto
        relaciones_alt = self._encontrar_relaciones_alternativas(
            sujeto, verbo, objeto, contexto
        )
        
        if not relaciones_alt:
            return None
            
        # Seleccionar y construir alternativa
        relacion_alt = random.choice(relaciones_alt)
        return self._construir_alternativa_con_relacion(relacion_alt)

    def _generar_por_variacion_semantica(self, doc: spacy.tokens.Doc,
                                       contexto: List[Dict[str, Any]]) -> List[str]:
        """Genera alternativas por variación semántica"""
        variaciones = []
        
        # Obtener significado principal
        significado = self._extraer_significado_principal(doc)
        
        # Generar variaciones manteniendo estructura pero cambiando significado
        for _ in range(3):
            variacion = self._generar_variacion_significado(
                significado,
                contexto
            )
            if variacion:
                variaciones.append(variacion)
                
        return variaciones

    def _filtrar_alternativas(self, alternativas: List[str], 
                            doc_respuesta: spacy.tokens.Doc) -> List[str]:
        """Filtra y selecciona las mejores alternativas"""
        alternativas_filtradas = []
        
        for alt in alternativas:
            # Verificar que sea suficientemente diferente
            if self._es_alternativa_valida(alt, doc_respuesta):
                alternativas_filtradas.append(alt)
                
        return alternativas_filtradas

    def _es_alternativa_valida(self, alternativa: str, 
                             doc_respuesta: spacy.tokens.Doc) -> bool:
        """Verifica si una alternativa es válida"""
        doc_alt = self.nlp(alternativa)
        
        # Calcular similitud
        similitud = doc_alt.similarity(doc_respuesta)
        
        # No debe ser muy similar ni muy diferente
        if 0.3 <= similitud <= 0.7:
            return True
            
        return False

    def generar_explicacion(self, pregunta: Dict[str, Any], respuesta: str, 
                          es_correcta: bool) -> str:
        """Genera una explicación detallada de la respuesta"""
        # Analizar pregunta y respuesta
        doc_pregunta = self.nlp(pregunta['pregunta'])
        doc_respuesta = self.nlp(respuesta)
        
        # Construir explicación basada en análisis
        return self._construir_explicacion(
            doc_pregunta,
            doc_respuesta,
            es_correcta,
            pregunta['contexto']
        )

    def _construir_explicacion(self, doc_pregunta: spacy.tokens.Doc,
                             doc_respuesta: spacy.tokens.Doc,
                             es_correcta: bool,
                             contexto: Dict[str, Any]) -> str:
        """Construye una explicación detallada"""
        # Analizar elementos clave
        conceptos_pregunta = self._identificar_conceptos_clave(doc_pregunta)
        conceptos_respuesta = self._identificar_conceptos_clave(doc_respuesta)
        
        # Generar explicación
        if es_correcta:
            return self._generar_explicacion_correcta(
                conceptos_pregunta,
                conceptos_respuesta,
                contexto
            )
        else:
            return self._generar_explicacion_incorrecta(
                conceptos_pregunta,
                conceptos_respuesta,
                contexto
            )

    def _generar_explicacion_correcta(self, conceptos_pregunta: List[Dict[str, Any]],
                                    conceptos_respuesta: List[Dict[str, Any]],
                                    contexto: Dict[str, Any]) -> str:
        """Genera explicación para respuesta correcta"""
        # Identificar relaciones clave
        relaciones = self._identificar_relaciones_explicativas(
            conceptos_pregunta,
            conceptos_respuesta,
            contexto
        )
        
        # Construir explicación
        return self._componer_explicacion(relaciones, True)

    def _generar_explicacion_incorrecta(self, conceptos_pregunta: List[Dict[str, Any]],
                                      conceptos_respuesta: List[Dict[str, Any]],
                                      contexto: Dict[str, Any]) -> str:
        """Genera explicación para respuesta incorrecta"""
        # Identificar diferencias clave
        diferencias = self._identificar_diferencias(
            conceptos_pregunta,
            conceptos_respuesta,
            contexto
        )
        
        # Construir explicación
        return self._componer_explicacion(diferencias, False)

    def _identificar_relaciones_explicativas(self, conceptos_pregunta: List[Dict[str, Any]],
                                          conceptos_respuesta: List[Dict[str, Any]],
                                          contexto: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identifica relaciones para explicación"""
        relaciones = []
        
        # Analizar conexiones entre conceptos
        for cp in conceptos_pregunta:
            for cr in conceptos_respuesta:
                relacion = self._analizar_relacion_conceptos(cp, cr, contexto)
                if relacion:
                    relaciones.append(relacion)
                    
        return relaciones

    def _identificar_diferencias(self, conceptos_pregunta: List[Dict[str, Any]],
                               conceptos_respuesta: List[Dict[str, Any]],
                               contexto: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identifica diferencias para explicación"""
        diferencias = []
        
        # Comparar conceptos
        for cp in conceptos_pregunta:
            for cr in conceptos_respuesta:
                diferencia = self._analizar_diferencia_conceptos(cp, cr, contexto)
                if diferencia:
                    diferencias.append(diferencia)
                    
        return diferencias

    def _componer_explicacion(self, elementos: List[Dict[str, Any]], 
                            es_correcta: bool) -> str:
        """Compone la explicación final"""
        if es_correcta:
            return self._componer_explicacion_correcta(elementos)
        else:
            return self._componer_explicacion_incorrecta(elementos)

    def _componer_explicacion_correcta(self, elementos: List[Dict[str, Any]]) -> str:
        """Compone explicación para respuesta correcta"""
        if not elementos:
            return "La respuesta es correcta según el contexto."
            
        partes = []
        for elem in elementos:
            parte = self._generar_parte_explicacion(elem)
            if parte:
                partes.append(parte)
                
        if not partes:
            return "La respuesta es correcta según el contexto."
            
        return " ".join(partes)

    def _componer_explicacion_incorrecta(self, elementos: List[Dict[str, Any]]) -> str:
        """Compone explicación para respuesta incorrecta"""
        if not elementos:
            return "La respuesta es incorrecta según el contexto."
            
        partes = []
        for elem in elementos:
            parte = self._generar_parte_explicacion(elem, es_correcta=False)
            if parte:
                partes.append(parte)
                
        if not partes:
            return "La respuesta es incorrecta según el contexto."
            
        return " ".join(partes)
    
    def _encontrar_conceptos_similares(self, concepto: str, contexto: List[str]) -> List[str]:
        """Encuentra conceptos semánticamente similares"""
        doc_concepto = self.nlp(concepto)
        conceptos_similares = []
        
        for texto in contexto:
            doc = self.nlp(texto)
            # Buscar sustantivos y entidades nombradas
            for token in doc:
                if token.pos_ in ['NOUN', 'PROPN']:
                    similitud = doc_concepto.similarity(token)
                    if 0.4 < similitud < 0.8:  # Similar pero no idéntico
                        conceptos_similares.append({
                            'texto': token.text,
                            'similitud': similitud
                        })
                        
        # Ordenar por similitud y eliminar duplicados
        conceptos_unicos = {}
        for c in sorted(conceptos_similares, key=lambda x: x['similitud'], reverse=True):
            if c['texto'].lower() not in conceptos_unicos:
                conceptos_unicos[c['texto'].lower()] = c['texto']
                
        return list(conceptos_unicos.values())

    def _construir_alternativa_con_sustitucion(self, concepto_original: str, 
                                            concepto_nuevo: str, 
                                            contexto: Dict[str, Any]) -> str:
        """Construye alternativa sustituyendo conceptos"""
        # Usar el contexto para mantener coherencia
        verbos = contexto.get('verbos_asociados', [])
        modificadores = contexto.get('modificadores', [])
        
        # Construir frase con elementos del contexto
        elementos = []
        if modificadores:
            elementos.append(random.choice(modificadores))
        elementos.append(concepto_nuevo)
        if verbos:
            elementos.append(random.choice(verbos))
            
        return ' '.join(elementos)

    def _identificar_relaciones(self, doc: spacy.tokens.Doc) -> List[Dict[str, Any]]:
        """Identifica relaciones sintácticas en el texto"""
        relaciones = []
        
        for token in doc:
            if token.dep_ == 'ROOT' and token.pos_ == 'VERB':
                # Buscar sujeto y objeto
                sujeto = None
                objeto = None
                for hijo in token.children:
                    if hijo.dep_ == 'nsubj':
                        sujeto = hijo.text
                    elif hijo.dep_ == 'dobj':
                        objeto = hijo.text
                        
                if sujeto and objeto:
                    relaciones.append({
                        'sujeto': sujeto,
                        'verbo': token.text,
                        'objeto': objeto,
                        'importancia': self._calcular_importancia_relacion(token)
                    })
                    
        return relaciones

    def _encontrar_relaciones_alternativas(self, sujeto: str, verbo: str, 
                                        objeto: str, contexto: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Encuentra relaciones alternativas en el contexto"""
        relaciones_alt = []
        
        for oracion in contexto:
            doc = self.nlp(oracion['texto'])
            
            for token in doc:
                if token.pos_ == 'VERB' and token.text != verbo:
                    # Buscar nueva relación
                    nueva_rel = self._extraer_relacion(token)
                    if nueva_rel and nueva_rel['sujeto'] != sujeto:
                        relaciones_alt.append(nueva_rel)
                        
        return relaciones_alt

    def _construir_alternativa_con_relacion(self, relacion: Dict[str, str]) -> str:
        """Construye alternativa usando una relación"""
        return f"{relacion['sujeto']} {relacion['verbo']} {relacion['objeto']}"

    def _extraer_significado_principal(self, doc: spacy.tokens.Doc) -> Dict[str, Any]:
        """Extrae el significado principal del texto"""
        # Encontrar verbo principal y sus argumentos
        raiz = None
        argumentos = []
        
        for token in doc:
            if token.dep_ == 'ROOT':
                raiz = token
                # Recolectar argumentos
                for hijo in token.children:
                    if hijo.dep_ in ['nsubj', 'dobj', 'iobj']:
                        argumentos.append({
                            'texto': hijo.text,
                            'tipo': hijo.dep_,
                            'pos': hijo.pos_
                        })
                        
        return {
            'raiz': raiz.text if raiz else None,
            'argumentos': argumentos,
            'tipo': raiz.pos_ if raiz else None
        }

    def _generar_variacion_significado(self, significado: Dict[str, Any], 
                                    contexto: List[Dict[str, Any]]) -> str:
        """Genera variación manteniendo estructura pero cambiando significado"""
        if not significado['raiz'] or not significado['argumentos']:
            return None
            
        # Buscar elementos similares en el contexto
        elementos_similares = self._buscar_elementos_similares(
            significado, 
            contexto
        )
        
        if not elementos_similares:
            return None
            
        # Construir nueva variación
        return self._construir_variacion(
            significado['raiz'],
            elementos_similares
        )

    def _buscar_elementos_similares(self, significado: Dict[str, Any], 
                                contexto: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Busca elementos similares en el contexto"""
        elementos = {
            'raices': [],
            'argumentos': []
        }
        
        for oracion in contexto:
            doc = self.nlp(oracion['texto'])
            for token in doc:
                if token.pos_ == significado['tipo']:
                    similitud = self.nlp(significado['raiz']).similarity(token)
                    if 0.3 < similitud < 0.7:
                        elementos['raices'].append(token.text)
                        
                for arg in significado['argumentos']:
                    if token.pos_ == arg['pos']:
                        similitud = self.nlp(arg['texto']).similarity(token)
                        if 0.3 < similitud < 0.7:
                            elementos['argumentos'].append(token.text)
                            
        return elementos

    def _construir_variacion(self, raiz: str, elementos: Dict[str, List[str]]) -> str:
        """Construye una variación usando elementos similares"""
        if not elementos['raices'] or not elementos['argumentos']:
            return None
            
        nueva_raiz = random.choice(elementos['raices'])
        nuevos_args = random.sample(elementos['argumentos'], 
                                min(2, len(elementos['argumentos'])))
                                
        return f"{nuevos_args[0]} {nueva_raiz} {nuevos_args[1] if len(nuevos_args) > 1 else ''}"

    def _analizar_relacion_conceptos(self, concepto1: Dict[str, Any], 
                                concepto2: Dict[str, Any], 
                                contexto: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza la relación entre dos conceptos"""
        # Calcular similitud
        doc1 = self.nlp(concepto1['texto'])
        doc2 = self.nlp(concepto2['texto'])
        similitud = doc1.similarity(doc2)
        
        # Buscar conexiones en el contexto
        conexiones = self._buscar_conexiones(
            concepto1['texto'],
            concepto2['texto'],
            contexto
        )
        
        return {
            'concepto1': concepto1['texto'],
            'concepto2': concepto2['texto'],
            'similitud': similitud,
            'conexiones': conexiones,
            'tipo': self._determinar_tipo_relacion(conexiones)
        }

    def _analizar_diferencia_conceptos(self, concepto1: Dict[str, Any], 
                                    concepto2: Dict[str, Any], 
                                    contexto: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza las diferencias entre dos conceptos"""
        # Analizar características distintivas
        caract1 = self._extraer_caracteristicas(concepto1, contexto)
        caract2 = self._extraer_caracteristicas(concepto2, contexto)
        
        # Encontrar diferencias
        diferencias = []
        for c1 in caract1:
            if not any(self._son_similares(c1, c2) for c2 in caract2):
                diferencias.append(c1)
                
        return {
            'concepto1': concepto1['texto'],
            'concepto2': concepto2['texto'],
            'diferencias': diferencias,
            'importancia': len(diferencias)
        }

    def _generar_parte_explicacion(self, elemento: Dict[str, Any], 
                            es_correcta: bool = True) -> str:
        """Genera una parte de la explicación dinámicamente"""
        doc = None
        
        if es_correcta:
            if 'conexiones' in elemento:
                # Analizar el contexto de la conexión
                doc = self.nlp(elemento['conexiones'][0] if elemento['conexiones'] else '')
            elif 'valor' in elemento:
                # Analizar la característica
                doc = self.nlp(elemento['valor'])
        else:
            if 'diferencias' in elemento:
                # Analizar las diferencias
                doc = self.nlp(elemento['diferencias'][0] if elemento['diferencias'] else '')
            elif 'razon' in elemento:
                # Analizar la razón del error
                doc = self.nlp(elemento['razon'])
                
        if not doc:
            return None
            
        # Construir explicación basada en análisis
        elementos_explicacion = self._analizar_elementos_explicacion(doc)
        return self._construir_explicacion_dinamica(elementos_explicacion, es_correcta)

    def _analizar_elementos_explicacion(self, doc: spacy.tokens.Doc) -> Dict[str, Any]:
        """Analiza elementos para construir explicación"""
        elementos = {
            'hechos': [],
            'causas': [],
            'consecuencias': []
        }
        
        for sent in doc.sents:
            # Identificar tipo de información
            if any(token.dep_ == 'ROOT' and token.pos_ == 'VERB' for token in sent):
                elementos['hechos'].append(sent.text)
            elif any(token.dep_ == 'prep' and token.text in ['porque', 'ya que'] for token in sent):
                elementos['causas'].append(sent.text)
            elif any(token.dep_ == 'prep' and token.text in ['por lo tanto', 'entonces'] for token in sent):
                elementos['consecuencias'].append(sent.text)
                
        return elementos

    def _construir_explicacion_dinamica(self, elementos: Dict[str, List[str]], 
                                    es_correcta: bool) -> str:
        """Construye explicación dinámicamente"""
        partes = []
        
        # Construir basado en elementos disponibles
        if elementos['hechos']:
            partes.extend(elementos['hechos'])
        if elementos['causas']:
            partes.extend(elementos['causas'])
        if elementos['consecuencias']:
            partes.extend(elementos['consecuencias'])
            
        # Si no hay elementos suficientes, analizar el contexto más amplio
        if not partes:
            return self._generar_explicacion_alternativa(es_correcta)
            
        return ' '.join(partes)

    def _generar_explicacion_alternativa(self, es_correcta: bool) -> str:
        """Genera explicación alternativa basada en análisis del contexto"""
        # Analizar contexto general
        doc = self.nlp(self.contexto_actual)  # Necesitaríamos mantener un contexto actual
        
        # Extraer información relevante
        info_relevante = self._extraer_informacion_relevante(doc)
        
        # Construir explicación
        return self._componer_explicacion_desde_info(info_relevante, es_correcta)
    

class SistemaGeneradorCuestionarios:
    def __init__(self):
        self.analizador = AnalizadorSemantico()
        self.generador_preguntas = GeneradorPreguntasDinamico(self.analizador)
        self.generador_alternativas = GeneradorAlternativasExplicaciones(self.analizador)
        
    def generar_cuestionarios(self, ruta_entrada: str, ruta_salida: str):
        """Genera cuestionarios completos desde JSON"""
        try:
            # Cargar datos de entrada
            with open(ruta_entrada, 'r', encoding='utf-8') as f:
                datos = json.load(f)
                
            if 'quiz' not in datos:
                raise ValueError("JSON inválido: falta clave 'quiz'")
                
            # Procesar cada texto
            cuestionarios = []
            for item in datos['quiz']:
                cuestionario = self._procesar_texto(
                    texto=item['texto'],
                    materia=item['materia'],
                    fuente=item['fuente']
                )
                if cuestionario:
                    cuestionarios.append(cuestionario)
                    
            # Guardar resultados
            self._guardar_cuestionarios(cuestionarios, ruta_salida)
            
        except Exception as e:
            print(f"Error procesando cuestionarios: {str(e)}")
            
    def _procesar_texto(self, texto: str, materia: str, fuente: str) -> Dict[str, Any]:
        """Procesa un texto y genera su cuestionario"""
        try:
            # 1. Generar preguntas base
            preguntas_base = self.generador_preguntas.generar_preguntas(texto)
            
            # 2. Generar alternativas y explicaciones
            preguntas_completas = []
            for pregunta in preguntas_base:
                # Generar alternativas
                alternativas = self.generador_alternativas.generar_alternativas(
                    pregunta, texto
                )
                
                if not alternativas:
                    continue
                    
                # Asignar respuesta correcta
                respuesta_correcta = alternativas[0]  # La primera es la correcta
                
                # Generar explicación
                explicacion = self.generador_alternativas.generar_explicacion(
                    pregunta,
                    respuesta_correcta,
                    True  # Es correcta
                )
                
                preguntas_completas.append({
                    'tipo': pregunta['tipo'],
                    'pregunta': pregunta['pregunta'],
                    'opciones': alternativas if pregunta['tipo'] == 'alternativas' else None,
                    'respuesta_correcta': respuesta_correcta,
                    'explicacion': explicacion
                })
                
            return {
                'materia': materia,
                'fuente': fuente,
                'preguntas': preguntas_completas
            }
            
        except Exception as e:
            print(f"Error procesando texto: {str(e)}")
            return None
            
    def _guardar_cuestionarios(self, cuestionarios: List[Dict], ruta: str):
        """Guarda los cuestionarios generados"""
        try:
            # Guardar JSON
            with open(ruta, 'w', encoding='utf-8') as f:
                json.dump({
                    'cuestionarios': cuestionarios
                }, f, ensure_ascii=False, indent=2)
                
            print(f"Cuestionarios guardados en {ruta}")
            
        except Exception as e:
            print(f"Error guardando cuestionarios: {str(e)}")