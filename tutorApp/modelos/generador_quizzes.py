# Generador de Quizzes
# por Franco Benassi
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqGeneration,
    AutoModelForQuestionAnswering,
    T5ForConditionalGeneration,
    pipeline
)
import spacy
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
import random
import re
from tqdm import tqdm

class ModeloPreguntasInteligente(nn.Module):
    """Modelo neuronal para generación de preguntas"""
    def __init__(self, vocab_size: int, embedding_dim: int = 768, hidden_dim: int = 512):
        super().__init__()
        # Capas de procesamiento
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=8)
        self.fc_pregunta = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_alternativas = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Embedding y LSTM
        embedded = self.dropout(self.embedding(x))
        lstm_out, _ = self.lstm(embedded)
        
        # Atención
        if mask is not None:
            lstm_out = lstm_out * mask.unsqueeze(-1)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Generación de pregunta y alternativas
        pregunta_features = self.fc_pregunta(attn_out)
        alternativas_logits = self.fc_alternativas(pregunta_features)
        
        return pregunta_features, alternativas_logits

class GeneradorCuestionarios:
    """Clase principal para generación de cuestionarios inteligentes"""
    def __init__(self):
        print("Inicializando modelos...")
        
        # Modelo base T5 para español
        self.tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
        self.t5_model = T5ForConditionalGeneration.from_pretrained("google/mt5-small")
        
        # Modelo BERT para español
        self.qa_pipeline = pipeline(
            "question-answering",
            model="mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"
        )
        
        # Modelo para embeddings y similitud
        self.semantic_model = SentenceTransformer('hiiamsid/sentence_similarity_spanish_es')
        
        # SpaCy para análisis lingüístico
        self.nlp = spacy.load('es_core_news_lg')
        
        # Modelo personalizado
        self.modelo = ModeloPreguntasInteligente(
            vocab_size=self.tokenizer.vocab_size,
            embedding_dim=768,
            hidden_dim=512
        )
        
        # Dispositivo (GPU/CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.modelo.to(self.device)
        self.t5_model.to(self.device)
        
        # Conjuntos para control
        self.preguntas_generadas = set()
        self.conceptos_procesados = {}
        
        print("Modelos cargados correctamente.")
        
        # Inicializar patrones de preguntas por materia
        self._init_patrones_preguntas()

    def _init_patrones_preguntas(self):
        """Inicializa patrones base para diferentes tipos de preguntas"""
        self.patrones_medicina = {
            'conceptual': [
                "¿Qué es {}?",
                "¿Cuál es la función principal de {}?",
                "¿Para qué sirve {}?"
            ],
            'proceso': [
                "¿Cómo funciona {}?",
                "¿Qué proceso se sigue para {}?",
                "¿Cuál es el mecanismo de {}?"
            ]
        }
        
        self.patrones_programacion = {
            'conceptual': [
                "¿Qué es {}?",
                "¿Cuál es el propósito de {}?",
                "¿Para qué se utiliza {}?"
            ],
            'practica': [
                "¿Cómo se implementa {}?",
                "¿Qué hace {}?",
                "¿Cuál es la diferencia entre {} y {}?"
            ]
        }
        
        self.patrones_pedagogia = {
            'conceptual': [
                "¿Qué es {}?",
                "¿Cuál es el objetivo de {}?",
                "¿Para qué se aplica {}?"
            ],
            'metodologia': [
                "¿Cómo se desarrolla {}?",
                "¿Qué estrategias se usan para {}?",
                "¿Cuál es la metodología de {}?"
            ]
        }

    def _verificar_pregunta(self, pregunta: str) -> bool:
        """Verifica la calidad lingüística de una pregunta"""
        # Verificar estructura básica
        if not pregunta.startswith('¿') or not pregunta.endswith('?'):
            return False
            
        # Analizar con SpaCy
        doc = self.nlp(pregunta)
        
        # Verificar componentes esenciales
        tiene_verbo = False
        tiene_sustantivo = False
        
        for token in doc:
            if token.pos_ == 'VERB':
                tiene_verbo = True
            elif token.pos_ in ['NOUN', 'PROPN']:
                tiene_sustantivo = True
            
        if not (tiene_verbo and tiene_sustantivo):
            return False
            
        # Verificar coherencia
        if len(pregunta.split()) < 4:  # Muy corta
            return False
            
        # Verificar que no tenga referencias al texto
        referencias = ['este texto', 'el texto', 'el párrafo', 'este contexto', 'el contexto']
        if any(ref in pregunta.lower() for ref in referencias):
            return False
        
        return True

    def _limpiar_texto(self, texto: str) -> str:
        """Limpia y normaliza el texto"""
        # Eliminar caracteres especiales
        texto = re.sub(r'[^\w\s\?\¿\!\¡\.]', '', texto)
        
        # Normalizar espacios
        texto = ' '.join(texto.split())
        
        # Asegurar mayúscula inicial
        texto = texto[0].upper() + texto[1:] if texto else texto
        
        return texto.strip()

    def _extraer_conceptos(self, texto: str) -> List[Dict]:
        """Extrae conceptos clave del texto usando NLP"""
        doc = self.nlp(texto)
        conceptos = []
        
        for sent in doc.sents:
            # Analizar solo oraciones sustanciales
            if len(sent.text.split()) >= 8:
                # Buscar definiciones y conceptos
                definiciones = []
                for token in sent:
                    if token.dep_ == 'ROOT' and token.lemma_ in ['ser', 'estar', 'significar', 'representar']:
                        sujeto = None
                        predicado = None
                        
                        # Buscar sujeto y predicado
                        for child in token.children:
                            if child.dep_ == 'nsubj':
                                sujeto = ' '.join([t.text for t in child.subtree])
                            elif child.dep_ in ['attr', 'dobj']:
                                predicado = ' '.join([t.text for t in child.subtree])
                        
                        if sujeto and predicado:
                            definiciones.append({
                                'concepto': sujeto,
                                'definicion': predicado
                            })
                
                # Extraer frases nominales importantes
                for chunk in sent.noun_chunks:
                    if len(chunk.text.split()) >= 2:
                        importancia = sum(1 for token in chunk 
                                        if not token.is_stop and token.has_vector)
                        if importancia >= 2:
                            conceptos.append({
                                'texto': chunk.text,
                                'tipo': 'concepto',
                                'oracion': sent.text,
                                'importancia': importancia
                            })
                
                # Agregar definiciones encontradas
                for def_item in definiciones:
                    conceptos.append({
                        'texto': def_item['concepto'],
                        'tipo': 'definicion',
                        'oracion': sent.text,
                        'definicion': def_item['definicion'],
                        'importancia': 3  # Priorizar definiciones
                    })
        
        return sorted(conceptos, key=lambda x: x['importancia'], reverse=True)

    def _generar_pregunta_concepto(self, concepto: Dict) -> Dict:
        """Genera una pregunta sobre un concepto específico"""
        try:
            pregunta = None
            # Elegir tipo de pregunta basado en el concepto
            if concepto['tipo'] == 'definicion':
                pregunta = f"¿Cuál es la definición correcta de {concepto['texto']}?"
                respuesta_correcta = concepto['definicion']
            else:
                # Generar pregunta usando T5
                prompt = f"generar pregunta sobre: {concepto['texto']}"
                inputs = self.tokenizer(prompt, return_tensors="pt", max_length=128).to(self.device)
                
                outputs = self.t5_model.generate(
                    **inputs,
                    max_length=64,
                    num_beams=5,
                    temperature=0.7,
                    top_k=50,
                    do_sample=True
                )
                
                pregunta_base = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                pregunta = self._formatear_pregunta(pregunta_base, concepto['texto'])
                
                # Obtener respuesta usando QA
                respuesta = self.qa_pipeline(
                    question=pregunta,
                    context=concepto['oracion']
                )
                respuesta_correcta = respuesta['answer']
            
            # Verificar calidad de la pregunta
            if not self._verificar_pregunta(pregunta):
                return None
                
            # Generar alternativas
            alternativas = self._generar_alternativas(
                respuesta_correcta,
                concepto['texto'],
                concepto['oracion']
            )
            
            if not alternativas or len(alternativas) < 4:
                return None
                
            return {
                'pregunta': pregunta,
                'opciones': alternativas,
                'respuesta_correcta': respuesta_correcta,
                'tipo': 'alternativas'
            }
            
        except Exception as e:
            print(f"Error generando pregunta: {str(e)}")
            return None

    def _generar_alternativas(self, respuesta: str, concepto: str, contexto: str) -> List[str]:
        """Genera alternativas plausibles y coherentes"""
        try:
            # Obtener embedding de la respuesta correcta
            emb_respuesta = self.semantic_model.encode([respuesta])[0]
            alternativas = [respuesta]
            
            # Generar alternativas basadas en el contexto
            doc = self.nlp(contexto)
            candidatos = []
            
            # Extraer frases candidatas
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) >= 2:
                    texto = self._limpiar_texto(chunk.text)
                    if texto != respuesta and concepto.lower() not in texto.lower():
                        emb_candidato = self.semantic_model.encode([texto])[0]
                        similitud = cosine_similarity([emb_respuesta], [emb_candidato])[0][0]
                        
                        if 0.3 <= similitud <= 0.7:  # Similar pero no idéntico
                            candidatos.append(texto)
            
            # Agregar mejores candidatos
            candidatos = list(set(candidatos))  # Eliminar duplicados
            random.shuffle(candidatos)
            
            for candidato in candidatos:
                if len(alternativas) >= 4:
                    break
                if candidato not in alternativas:
                    alternativas.append(candidato)
            
            # Si faltan alternativas, generarlas sintéticamente
            while len(alternativas) < 4:
                alt = self._generar_alternativa_sintetica(respuesta)
                if alt and alt not in alternativas:
                    alternativas.append(alt)
            
            random.shuffle(alternativas)
            return alternativas
            
        except Exception as e:
            print(f"Error generando alternativas: {str(e)}")
            return None

    def _generar_alternativa_sintetica(self, texto: str) -> str:
        """Genera una alternativa sintética modificando el texto base"""
        doc = self.nlp(texto)
        tokens = [token.text for token in doc]
        
        for i, token in enumerate(doc):
            if token.pos_ in ['NOUN', 'ADJ', 'VERB']:
                similares = []
                for word in token.vocab:
                    if (word.is_lower and 
                        word.has_vector and 
                        token.similarity(word) > 0.5):
                        similares.append(word.text)
                
                if similares:
                    tokens[i] = random.choice(similares)
                    return ' '.join(tokens)
        
        return None

    def _generar_verdadero_falso(self, concepto: Dict) -> Dict:
        """Genera una pregunta de verdadero/falso"""
        try:
            texto = concepto['oracion']
            es_verdadero = random.choice([True, False])
            
            if es_verdadero:
                pregunta = texto
            else:
                doc = self.nlp(texto)
                tokens = []
                modificado = False
                
                for token in doc:
                    if token.pos_ == 'VERB' and not modificado:
                        tokens.extend(['no', token.text])
                        modificado = True
                    else:
                        tokens.append(token.text)
                
                pregunta = ' '.join(tokens)
            
            pregunta = self._limpiar_texto(pregunta)
            
            return {
                'pregunta': pregunta,
                'opciones': ['Verdadero', 'Falso'],
                'respuesta_correcta': 'Verdadero' if es_verdadero else 'Falso',
                'tipo': 'verdadero_falso'
            }
            
        except Exception as e:
            print(f"Error generando pregunta V/F: {str(e)}")
            return None

    def _formatear_pregunta(self, pregunta: str, concepto: str) -> str:
        """Formatea la pregunta para hacerla autónoma y coherente"""
        pregunta = pregunta.strip()
        
        # Asegurar signos de interrogación
        if not pregunta.startswith('¿'):
            pregunta = '¿' + pregunta
        if not pregunta.endswith('?'):
            pregunta = pregunta + '?'
        
        # Reemplazar referencias ambiguas
        pregunta = pregunta.replace('esto', concepto)
        pregunta = pregunta.replace('esta', concepto)
        pregunta = pregunta.replace('eso', concepto)
        
        # Asegurar mayúscula inicial
        pregunta = pregunta[0] + pregunta[1].upper() + pregunta[2:]
        
        return pregunta

    def _validar_alternativas(self, alternativas: List[str], respuesta_correcta: str) -> bool:
        """Valida la calidad y coherencia de las alternativas"""
        if len(alternativas) != 4:
            return False
            
        if len(set(alternativas)) != 4:  # Verificar duplicados
            return False
            
        if respuesta_correcta not in alternativas:
            return False
            
        # Verificar coherencia semántica
        embeddings = self.semantic_model.encode(alternativas)
        similitudes = cosine_similarity(embeddings)
        
        # Verificar que las alternativas no sean muy similares entre sí
        for i in range(len(alternativas)):
            for j in range(i + 1, len(alternativas)):
                if i != j and similitudes[i][j] > 0.9:  # Muy similares
                    return False
        
        # Verificar longitud y estructura
        for alt in alternativas:
            if len(alt.split()) < 2:  # Muy corta
                return False
            
            doc = self.nlp(alt)
            if not any(token.pos_ in ['NOUN', 'VERB'] for token in doc):
                return False
        
        return True

    def generar_cuestionario(self, ruta_archivo: str) -> Dict:
        """Genera cuestionarios completos a partir del archivo JSON"""
        try:
            print("Cargando datos...")
            with open(ruta_archivo, 'r', encoding='utf-8') as archivo:
                datos = json.load(archivo)

            print("Procesando textos y generando cuestionarios...")
            cuestionarios = {}
            
            for item in datos['quiz']:
                materia = item['materia']
                texto = item['texto']
                
                if materia not in cuestionarios:
                    cuestionarios[materia] = []
                
                print(f"\nProcesando materia: {materia}")
                
                # Extraer conceptos
                conceptos = self._extraer_conceptos(texto)
                if not conceptos:
                    print(f"No se encontraron conceptos relevantes para {materia}")
                    continue
                
                # Generar preguntas
                preguntas = []
                intentos = 0
                max_intentos = 40  # Aumentar intentos para mejor calidad
                
                # Generar preguntas de alternativas (7)
                print("Generando preguntas de alternativas...")
                while len([p for p in preguntas if p['tipo'] == 'alternativas']) < 7 and intentos < max_intentos:
                    concepto = random.choice(conceptos)
                    
                    # Evitar conceptos ya usados
                    if concepto['texto'] not in self.conceptos_procesados:
                        pregunta = self._generar_pregunta_concepto(concepto)
                        
                        if pregunta and self._validar_pregunta_completa(pregunta, preguntas):
                            preguntas.append(pregunta)
                            self.preguntas_generadas.add(pregunta['pregunta'])
                            self.conceptos_procesados[concepto['texto']] = True
                    
                    intentos += 1
                
                # Generar preguntas de verdadero/falso (3)
                print("Generando preguntas de verdadero/falso...")
                intentos = 0
                while len([p for p in preguntas if p['tipo'] == 'verdadero_falso']) < 3 and intentos < max_intentos:
                    concepto = random.choice(conceptos)
                    pregunta = self._generar_verdadero_falso(concepto)
                    
                    if pregunta and pregunta['pregunta'] not in self.preguntas_generadas:
                        preguntas.append(pregunta)
                        self.preguntas_generadas.add(pregunta['pregunta'])
                    
                    intentos += 1
                
                # Verificar calidad del cuestionario
                if len(preguntas) >= 8:
                    print(f"Generadas {len(preguntas)} preguntas de calidad para {materia}")
                    cuestionarios[materia].append({
                        'texto': texto,
                        'fuente': item['fuente'],
                        'preguntas': preguntas
                    })
                else:
                    print(f"No se generaron suficientes preguntas de calidad para {materia}")
                    continue
                
                # Limpiar estado para siguiente iteración
                self.preguntas_generadas.clear()
                self.conceptos_procesados.clear()
            
            return cuestionarios
            
        except Exception as e:
            print(f"Error en la generación de cuestionarios: {str(e)}")
            return None

    def _validar_pregunta_completa(self, pregunta: Dict, preguntas_existentes: List[Dict]) -> bool:
        """Validación completa de una pregunta y sus alternativas"""
        # Verificar duplicados
        if pregunta['pregunta'] in self.preguntas_generadas:
            return False
        
        # Verificar similitud con preguntas existentes
        for p_existente in preguntas_existentes:
            similitud = self.semantic_model.encode([pregunta['pregunta'], p_existente['pregunta']])
            if cosine_similarity([similitud[0]], [similitud[1]])[0][0] > 0.8:
                return False
        
        # Verificar estructura de la pregunta
        if not self._verificar_pregunta(pregunta['pregunta']):
            return False
        
        # Verificar alternativas
        if pregunta['tipo'] == 'alternativas':
            if not self._validar_alternativas(pregunta['opciones'], pregunta['respuesta_correcta']):
                return False
        
        # Verificar coherencia pregunta-respuesta
        doc_pregunta = self.nlp(pregunta['pregunta'])
        doc_respuesta = self.nlp(pregunta['respuesta_correcta'])
        
        # Verificar que la respuesta sea relevante para la pregunta
        tiene_relacion = False
        for token_p in doc_pregunta:
            for token_r in doc_respuesta:
                if token_p.similarity(token_r) > 0.5:
                    tiene_relacion = True
                    break
            if tiene_relacion:
                break
        
        if not tiene_relacion:
            return False
        
        return True