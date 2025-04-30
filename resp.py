# Generador de Quizzes de alternativas
# por Franco Benassi
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
    MarianMTModel,
    pipeline
)
import torch
import torch.nn as nn
import spacy
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
import random
import re
from tqdm import tqdm

class GeneradorPreguntas(nn.Module):
    """Modelo neuronal para generar preguntas y alternativas"""
    def __init__(self, hidden_size=768, num_layers=3):
        super().__init__()
        
        # Encoder
        self.encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=0.2,
            batch_first=True
        )
        
        # Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=8,
            dropout=0.1
        )
        
        # Decoder para preguntas
        self.pregunta_decoder = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.2,
            batch_first=True
        )
        
        # Decoder para alternativas
        self.alternativa_decoder = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.2,
            batch_first=True
        )
        
        # Capas de salida
        self.fc_pregunta = nn.Linear(hidden_size, hidden_size)
        self.fc_alternativa = nn.Linear(hidden_size, hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, mask=None):
        # Codificación
        encoded, _ = self.encoder(x)
        
        # Atención
        if mask is not None:
            encoded = encoded * mask.unsqueeze(-1)
        context, _ = self.attention(encoded, encoded, encoded)
        
        # Decodificación de pregunta
        pregunta_hidden, _ = self.pregunta_decoder(context)
        pregunta_out = self.fc_pregunta(self.dropout(pregunta_hidden))
        
        # Decodificación de alternativas  
        alt_hidden, _ = self.alternativa_decoder(context)
        alt_out = self.fc_alternativa(self.dropout(alt_hidden))
        
        return pregunta_out, alt_out

class GeneradorCuestionarios:
    def __init__(self):
        print("Inicializando modelos...")
        
        try:
            # Modelo T5 multilingüe para generación
            self.gen_tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")
            self.gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-base")
            
            # Modelo M2M100 para traducción/parafraseo
            self.trans_tokenizer = AutoTokenizer.from_pretrained("facebook/m2m100_418M")
            self.trans_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/m2m100_418M")
            
            # Modelo para embeddings
            self.semantic_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
            
            # SpaCy para análisis lingüístico
            try:
                self.nlp = spacy.load('es_core_news_lg')
            except OSError:
                print("Descargando modelo de SpaCy...")
                spacy.cli.download('es_core_news_lg')
                self.nlp = spacy.load('es_core_news_lg')
            
            # Inicializar cache
            self.cache_conceptos = {}
            self.cache_embeddings = {}
            
            # Configurar dispositivo
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Usando dispositivo: {self.device}")
            
            # Mover modelos al dispositivo
            self.gen_model.to(self.device)
            self.trans_model.to(self.device)
            
            print("Modelos cargados correctamente")
            
        except Exception as e:
            print(f"Error inicializando modelos: {str(e)}")
            raise
        
    def _extraer_conceptos_clave(self, texto: str) -> List[Dict]:
        """Extrae conceptos clave usando análisis lingüístico profundo"""
        if texto in self.cache_conceptos:
            return self.cache_conceptos[texto]
            
        doc = self.nlp(texto)
        conceptos = []
        
        # Extraer entidades nombradas
        for ent in doc.ents:
            if ent.label_ in ['PER', 'ORG', 'LOC', 'MISC']:
                conceptos.append({
                    'texto': ent.text,
                    'tipo': ent.label_,
                    'contexto': self._obtener_contexto(doc, ent.start, ent.end)
                })
        
        # Extraer conceptos por análisis sintáctico
        for sent in doc.sents:
            # Buscar definiciones explícitas
            for token in sent:
                if token.dep_ == 'ROOT' and token.lemma_ in ['ser', 'estar', 'significar']:
                    sujeto = None
                    predicado = None
                    
                    for child in token.children:
                        if child.dep_ == 'nsubj':
                            sujeto = child
                        elif child.dep_ in ['attr', 'acomp']:
                            predicado = child
                            
                    if sujeto and predicado:
                        conceptos.append({
                            'texto': sujeto.text,
                            'definicion': predicado.text,
                            'contexto': sent.text,
                            'tipo': 'definicion'
                        })
            
            # Extraer frases nominales importantes
            for chunk in sent.noun_chunks:
                if len(chunk.text.split()) >= 2:  # Frases sustantivas compuestas
                    relevancia = sum(1 for token in chunk if not token.is_stop)
                    if relevancia >= 2:
                        conceptos.append({
                            'texto': chunk.text,
                            'tipo': 'concepto',
                            'contexto': sent.text
                        })
        
        # Filtrar y ordenar por relevancia
        conceptos = self._filtrar_conceptos(conceptos)
        self.cache_conceptos[texto] = conceptos
        return conceptos
        
    def _obtener_contexto(self, doc, start, end, window=2):
        """Obtiene el contexto alrededor de una entidad"""
        sent = doc[start].sent
        return sent.text
        
    def _filtrar_conceptos(self, conceptos: List[Dict]) -> List[Dict]:
        """Filtra y ordena conceptos por relevancia"""
        # Eliminar duplicados
        conceptos_unicos = []
        textos_vistos = set()
        
        for concepto in conceptos:
            texto = concepto['texto'].lower()
            if texto not in textos_vistos:
                conceptos_unicos.append(concepto)
                textos_vistos.add(texto)
        
        # Calcular relevancia usando embeddings
        for concepto in conceptos_unicos:
            if concepto['texto'] not in self.cache_embeddings:
                self.cache_embeddings[concepto['texto']] = self.semantic_model.encode([concepto['texto']])[0]
            
            # Calcular similitud con el contexto
            contexto_emb = self.semantic_model.encode([concepto['contexto']])[0]
            concepto['relevancia'] = cosine_similarity(
                [self.cache_embeddings[concepto['texto']]],
                [contexto_emb]
            )[0][0]
        
        # Ordenar por relevancia
        return sorted(conceptos_unicos, key=lambda x: x['relevancia'], reverse=True)
        
    def _generar_pregunta_desde_texto(self, texto: str, contexto: str) -> Dict:
        """Genera una pregunta analítica usando el modelo de lenguaje"""
        try:
            doc = self.nlp(texto)
            oraciones_importantes = [sent for sent in doc.sents 
                                if len(sent.text.split()) >= 8]
            
            if not oraciones_importantes:
                return None
                
            # Seleccionar una oración importante
            oracion = random.choice(oraciones_importantes)
            
            # Generar pregunta usando el modelo T5
            prompt = f"""
            Contexto: {oracion.text}
            Instrucción: Genera una pregunta de análisis en español que evalúe la comprensión del concepto principal.
            La pregunta debe ser clara, específica y requerir una respuesta concreta.
            """
            
            inputs = self.gen_tokenizer(prompt, return_tensors="pt", max_length=512).to(self.device)
            outputs = self.gen_model.generate(
                **inputs,
                max_length=128,
                num_beams=5,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                no_repeat_ngram_size=2
            )
            
            pregunta_base = self.gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
            pregunta = self._formatear_pregunta(pregunta_base)
            
            # Extraer respuesta usando análisis semántico
            doc_oracion = self.nlp(str(oracion))
            candidatos_respuesta = []
            
            # Extraer sintagmas nominales relevantes como posibles respuestas
            for chunk in doc_oracion.noun_chunks:
                if len(chunk.text.split()) >= 2:
                    # Calcular relevancia semántica con la pregunta
                    similitud = self._calcular_similitud(chunk.text, pregunta)
                    if similitud > 0.3:
                        candidatos_respuesta.append({
                            'texto': chunk.text,
                            'relevancia': similitud
                        })
            
            if not candidatos_respuesta:
                return None
                
            # Seleccionar la respuesta más relevante
            candidatos_respuesta.sort(key=lambda x: x['relevancia'], reverse=True)
            respuesta_correcta = candidatos_respuesta[0]['texto']
            
            # Generar alternativas coherentes
            alternativas = {respuesta_correcta}
            
            # Generar contraejemplos usando el modelo
            prompt_alt = f"""
            Contexto: {oracion.text}
            Pregunta: {pregunta}
            Respuesta correcta: {respuesta_correcta}
            Instrucción: Genera una respuesta alternativa que podría considerarse pero que no es correcta.
            """
            
            for _ in range(5):  # Intentar generar más alternativas de las necesarias
                if len(alternativas) >= 4:
                    break
                    
                inputs_alt = self.gen_tokenizer(prompt_alt, return_tensors="pt", max_length=512).to(self.device)
                outputs_alt = self.gen_model.generate(
                    **inputs_alt,
                    max_length=64,
                    num_beams=3,
                    temperature=0.8,
                    do_sample=True
                )
                
                alternativa = self.gen_tokenizer.decode(outputs_alt[0], skip_special_tokens=True)
                
                # Validar la alternativa
                if (len(alternativa.split()) >= 2 and
                    alternativa not in alternativas and
                    self._validar_alternativa(alternativa, respuesta_correcta, pregunta)):
                    alternativas.add(alternativa)
            
            # Si aún faltan alternativas, buscar en el contexto
            if len(alternativas) < 4:
                doc_contexto = self.nlp(contexto)
                for chunk in doc_contexto.noun_chunks:
                    if len(alternativas) >= 4:
                        break
                        
                    if (chunk.text not in alternativas and 
                        len(chunk.text.split()) >= 2 and
                        self._validar_alternativa(chunk.text, respuesta_correcta, pregunta)):
                        alternativas.add(chunk.text)
            
            if len(alternativas) < 4:
                return None
                
            alternativas = list(alternativas)[:4]
            random.shuffle(alternativas)
            
            return {
                'pregunta': pregunta,
                'opciones': alternativas,
                'respuesta_correcta': respuesta_correcta,
                'tipo': 'alternativas'
            }
            
        except Exception as e:
            print(f"Error generando pregunta: {str(e)}")
            return None
        
    def _validar_alternativa(self, alternativa: str, respuesta: str, pregunta: str) -> bool:
        """Valida que una alternativa sea coherente y desafiante"""
        try:
            # Verificar similitud semántica
            similitud_respuesta = self._calcular_similitud(alternativa, respuesta)
            if similitud_respuesta > 0.8 or similitud_respuesta < 0.2:
                return False
                
            # Verificar relevancia con la pregunta
            similitud_pregunta = self._calcular_similitud(alternativa, pregunta)
            if similitud_pregunta < 0.3:
                return False
                
            # Verificar estructura gramatical
            doc_alt = self.nlp(alternativa)
            if not any(token.pos_ in ['NOUN', 'PROPN', 'VERB'] for token in doc_alt):
                return False
                
            return True
            
        except Exception as e:
            print(f"Error validando alternativa: {str(e)}")
            return False

    def _generar_alternativas_semanticas(self, respuesta: str, tipo: str, contexto) -> List[str]:
        """Genera alternativas semánticamente relacionadas"""
        try:
            alternativas = {respuesta}
            doc_respuesta = self.nlp(respuesta)
            
            # Encontrar términos del mismo tipo semántico
            candidatos = []
            for token in contexto:
                if token.pos_ == doc_respuesta[0].pos_:  # Mismo tipo gramatical
                    similitud = token.similarity(doc_respuesta[0])
                    if 0.3 <= similitud <= 0.8:  # Similar pero no idéntico
                        candidatos.append({
                            'texto': token.text,
                            'similitud': similitud
                        })
            
            # Si no hay suficientes candidatos, buscar en el vocabulario
            if len(candidatos) < 5:
                for word in self.nlp.vocab:
                    if (word.has_vector and 
                        word.is_lower and 
                        word.prob >= -15 and  # Palabras comunes
                        word.pos_ == doc_respuesta[0].pos_):  # Mismo tipo
                        similitud = word.similarity(doc_respuesta[0])
                        if 0.3 <= similitud <= 0.8:
                            candidatos.append({
                                'texto': word.text,
                                'similitud': similitud
                            })
            
            # Filtrar y ordenar candidatos
            candidatos = sorted(
                [c for c in candidatos if self._es_alternativa_valida(c['texto'], respuesta)],
                key=lambda x: x['similitud'],
                reverse=True
            )
            
            # Agregar mejores alternativas
            for candidato in candidatos:
                if len(alternativas) >= 4:
                    break
                alternativas.add(candidato['texto'])
            
            if len(alternativas) >= 4:
                return list(alternativas)
            return None
            
        except Exception as e:
            print(f"Error generando alternativas: {str(e)}")
            return None

    def _es_alternativa_valida(self, alternativa: str, respuesta: str) -> bool:
        """Verifica que una alternativa es válida y coherente"""
        try:
            # Verificar longitud mínima
            if len(alternativa) < 2:
                return False
                
            # Verificar que no es igual a la respuesta
            if alternativa.lower() == respuesta.lower():
                return False
                
            # Verificar coherencia semántica
            doc_alt = self.nlp(alternativa)
            doc_resp = self.nlp(respuesta)
            
            # Mismo tipo gramatical
            if doc_alt[0].pos_ != doc_resp[0].pos_:
                return False
                
            # Verificar que es una palabra con sentido
            if not doc_alt[0].has_vector or doc_alt[0].is_punct:
                return False
                
            return True
            
        except Exception as e:
            print(f"Error validando alternativa: {str(e)}")
            return False
            
    def _extraer_respuesta(self, pregunta: str, contexto: str) -> str:
        """Extrae la respuesta más probable del contexto"""
        doc = self.nlp(contexto)
        max_score = 0
        mejor_respuesta = None
        
        # Analizamos cada frase del contexto
        for sent in doc.sents:
            # Calculamos similitud semántica
            similitud = sent.similarity(self.nlp(pregunta))
            if similitud > max_score:
                max_score = similitud
                mejor_respuesta = sent.text
        
        if max_score > 0.5 and mejor_respuesta:
            # Extraer la parte más relevante de la respuesta
            doc_resp = self.nlp(mejor_respuesta)
            chunks = list(doc_resp.noun_chunks) + [token for token in doc_resp if token.pos_ in ['VERB', 'ADJ']]
            
            if chunks:
                return max(chunks, key=lambda x: len(str(x).split())).text
                
        return None

    def _generar_alternativa_sintetica(self, respuesta: str) -> str:
        """Genera una alternativa sintética modificando la respuesta correcta"""
        try:
            doc = self.nlp(respuesta)
            tokens = []
            modificado = False
            
            for token in doc:
                if not modificado and token.pos_ in ['NOUN', 'VERB', 'ADJ']:
                    # Buscar palabras relacionadas
                    similares = []
                    if token.has_vector:
                        # Buscar en el vocabulario
                        for word in self.nlp.vocab:
                            if (word.has_vector and 
                                word.is_lower and 
                                word.prob >= -15 and
                                word.text != token.text and
                                len(word.text) > 3):
                                sim = token.similarity(word)
                                if 0.4 <= sim <= 0.8:  # Similar pero no demasiado
                                    similares.append(word)
                    
                    if similares:
                        palabra_similar = random.choice(similares).text
                        tokens.append(palabra_similar)
                        modificado = True
                        continue
                
                tokens.append(token.text)
            
            if not modificado:
                return None
            
            return ' '.join(tokens)
            
        except Exception as e:
            print(f"Error generando alternativa sintética: {str(e)}")
            return None

    # Corregir el método de parafraseo
    def _generar_parafraseo(self, texto: str) -> List[str]:
        """Genera paráfrasis usando M2M100"""
        try:
            # Configurar tokens de idioma
            self.trans_tokenizer.src_lang = "es"
            self.trans_tokenizer.tgt_lang = "es"
            
            # Codificar input
            inputs = self.trans_tokenizer(texto, return_tensors="pt", padding=True).to(self.device)
            
            # Generar múltiples paráfrasis
            outputs = self.trans_model.generate(
                **inputs,
                max_length=128,
                num_beams=5,
                num_return_sequences=3,
                temperature=0.8,
                do_sample=False,  # Cambiado a False por el diversity_penalty
                diversity_penalty=0.5,
                num_beam_groups=3
            )
            
            # Decodificar y filtrar resultados
            paraphrases = [
                self.trans_tokenizer.decode(out, skip_special_tokens=True)
                for out in outputs
            ]
            
            # Filtrar paráfrasis muy similares o idénticas
            filtered_paraphrases = []
            for p in paraphrases:
                if p != texto and self._calcular_similitud(p, texto) < 0.9:
                    filtered_paraphrases.append(p)
                    
            return filtered_paraphrases
            
        except Exception as e:
            print(f"Error en parafraseo: {str(e)}")
            return []

    # Mejorar el método de generación de alternativas
    def _generar_alternativas_inteligentes(self, respuesta: str, texto: str, contexto: str) -> List[str]:
        """Genera alternativas usando una combinación de técnicas"""
        try:
            alternativas = {respuesta}  # Usamos set para evitar duplicados
            
            # 1. Generar paráfrasis
            paraphrases = self._generar_parafraseo(respuesta)
            if paraphrases:
                for p in paraphrases[:2]:
                    if len(alternativas) < 4 and p not in alternativas:
                        alternativas.add(p)
            
            # 2. Extraer alternativas del contexto
            if len(alternativas) < 4:
                contexto_alts = self._extraer_alternativas_contexto(respuesta, contexto)
                if contexto_alts:
                    for alt in contexto_alts:
                        if len(alternativas) < 4 and alt not in alternativas:
                            alternativas.add(alt)
            
            # 3. Generar alternativas sintéticas
            max_intentos = 10
            intentos = 0
            while len(alternativas) < 4 and intentos < max_intentos:
                alt_sintetica = self._generar_alternativa_sintetica(respuesta)
                if alt_sintetica and alt_sintetica not in alternativas:
                    alternativas.add(alt_sintetica)
                intentos += 1
            
            # Si no tenemos suficientes alternativas, intentar generar más del contexto
            if len(alternativas) < 4:
                doc = self.nlp(contexto)
                for chunk in doc.noun_chunks:
                    if len(alternativas) >= 4:
                        break
                    if len(chunk.text.split()) >= 2 and chunk.text not in alternativas:
                        alternativas.add(chunk.text)
            
            alternativas_lista = list(alternativas)
            if len(alternativas_lista) >= 4:
                return alternativas_lista[:4]
            return None
            
        except Exception as e:
            print(f"Error generando alternativas: {str(e)}")
            return None

    def _validar_pregunta_completa(self, pregunta: str, alternativas: List[str], respuesta: str) -> bool:
        """Validación completa de pregunta y alternativas"""
        try:
            # Validar estructura de la pregunta
            if not self._validar_pregunta(pregunta, alternativas, respuesta):
                return False
                
            # Verificar que la pregunta es gramaticalmente correcta
            doc = self.nlp(pregunta)
            if not any(token.pos_ == 'VERB' for token in doc):
                return False
                
            # Verificar coherencia de alternativas
            alt_embeddings = self.semantic_model.encode(alternativas)
            similitudes = cosine_similarity(alt_embeddings)
            
            # Verificar que las alternativas son distintas pero relacionadas
            for i in range(len(alternativas)):
                for j in range(i + 1, len(alternativas)):
                    sim = similitudes[i][j]
                    if sim > 0.95:  # Muy similares
                        return False
                    if sim < 0.1:   # Muy diferentes
                        return False
            
            # Verificar que la respuesta correcta tiene sentido
            resp_doc = self.nlp(respuesta)
            preg_doc = self.nlp(pregunta)
            if resp_doc.similarity(preg_doc) < 0.3:
                return False
            
            return True
            
        except Exception as e:
            print(f"Error en validación: {str(e)}")
            return False

    def _calcular_similitud(self, texto1: str, texto2: str) -> float:
        """Calcula la similitud semántica entre dos textos"""
        try:
            emb1 = self.semantic_model.encode([texto1])[0]
            emb2 = self.semantic_model.encode([texto2])[0]
            return float(util.pytorch_cos_sim(emb1, emb2))
        except:
            return 0.0

    def _extraer_alternativas_contexto(self, respuesta: str, contexto: str) -> List[str]:
        """Extrae alternativas plausibles del contexto"""
        doc = self.nlp(contexto)
        candidatos = []
        
        for sent in doc.sents:
            for chunk in sent.noun_chunks:
                if (len(chunk.text.split()) >= len(respuesta.split()) - 1 and
                    chunk.text != respuesta):
                    similitud = self._calcular_similitud(chunk.text, respuesta)
                    if 0.3 <= similitud <= 0.8:
                        candidatos.append(chunk.text)
                        
        return candidatos
        
    def _generar_alternativas_v2(self, respuesta: str, texto: str, contexto: str) -> List[str]:
        """Nuevo método para generar alternativas más coherentes"""
        try:
            alternativas = {respuesta}
            doc = self.nlp(contexto)
            
            # 1. Extraer candidatos del contexto
            candidatos = []
            for sent in doc.sents:
                for chunk in sent.noun_chunks:
                    if len(chunk.text.split()) >= 2 and chunk.text != respuesta:
                        emb_chunk = self.semantic_model.encode([chunk.text])[0]
                        emb_respuesta = self.semantic_model.encode([respuesta])[0]
                        similitud = float(util.pytorch_cos_sim(emb_chunk, emb_respuesta))
                        
                        if 0.3 <= similitud <= 0.8:
                            candidatos.append({
                                'texto': chunk.text,
                                'similitud': similitud
                            })
            
            # 2. Generar alternativas por sustitución
            doc_resp = self.nlp(respuesta)
            for token in doc_resp:
                if token.pos_ in ['NOUN', 'VERB', 'ADJ'] and len(token.text) > 3:
                    # Buscar palabras similares
                    similares = []
                    if token.has_vector:
                        for word in self.nlp.vocab:
                            if (word.has_vector and 
                                word.is_lower and 
                                word.prob >= -15):
                                sim = token.similarity(word)
                                if 0.4 <= sim <= 0.8:
                                    similares.append(word.text)
                                    
                    if similares:
                        nueva_alt = respuesta.replace(token.text, random.choice(similares))
                        if nueva_alt != respuesta:
                            emb_alt = self.semantic_model.encode([nueva_alt])[0]
                            emb_resp = self.semantic_model.encode([respuesta])[0]
                            similitud = float(util.pytorch_cos_sim(emb_alt, emb_resp))
                            
                            if 0.3 <= similitud <= 0.8:
                                candidatos.append({
                                    'texto': nueva_alt,
                                    'similitud': similitud
                                })
            
            # 3. Seleccionar mejores alternativas
            candidatos = sorted(candidatos, key=lambda x: x['similitud'], reverse=True)
            for candidato in candidatos:
                if len(alternativas) >= 4:
                    break
                if candidato['texto'] not in alternativas:
                    alternativas.add(candidato['texto'])
            
            # 4. Si faltan alternativas, generar más del contexto
            if len(alternativas) < 4:
                for ent in doc.ents:
                    if len(alternativas) >= 4:
                        break
                    if (ent.label_ in ['ORG', 'PERSON', 'LOC', 'MISC'] and 
                        ent.text not in alternativas):
                        alternativas.add(ent.text)
            
            return list(alternativas) if len(alternativas) >= 4 else None
            
        except Exception as e:
            print(f"Error generando alternativas: {str(e)}")
            return None

    def _generar_verdadero_falso(self, texto: str) -> Dict:
        """Versión mejorada del generador de preguntas V/F"""
        try:
            doc = self.nlp(texto)
            # Encontrar oraciones con hechos o afirmaciones
            oraciones_candidatas = []
            
            for sent in doc.sents:
                # Verificar que la oración es una afirmación
                if len(sent.text.split()) >= 8:
                    tiene_verbo = any(token.pos_ == 'VERB' for token in sent)
                    tiene_sujeto = any(token.dep_ == 'nsubj' for token in sent)
                    
                    if tiene_verbo and tiene_sujeto:
                        oraciones_candidatas.append(sent.text)
            
            if not oraciones_candidatas:
                return None
                
            oracion = random.choice(oraciones_candidatas)
            es_verdadero = random.choice([True, False])
            
            if es_verdadero:
                pregunta = self._limpiar_texto(oracion)
            else:
                doc_oracion = self.nlp(oracion)
                modificaciones = []
                
                # Identificar elementos clave para modificar
                for token in doc_oracion:
                    if token.pos_ in ['VERB', 'ADJ', 'NUM', 'PROPN'] and token.has_vector:
                        # Buscar antónimos o términos opuestos
                        opuestos = []
                        for word in self.nlp.vocab:
                            if (word.has_vector and 
                                word.is_lower and 
                                word.prob >= -15 and
                                word.pos_ == token.pos_):
                                sim = token.similarity(word)
                                if sim < 0.3:  # Palabras semánticamente opuestas
                                    opuestos.append(word.text)
                        
                        if opuestos:
                            modificaciones.append((token, random.choice(opuestos)))
                
                if not modificaciones:
                    return None
                    
                # Aplicar una modificación aleatoria
                mod = random.choice(modificaciones)
                pregunta = oracion.replace(mod[0].text, mod[1])
                
            if not self._validar_verdadero_falso(pregunta, oracion):
                return None
                
            return {
                'pregunta': pregunta,
                'opciones': ['Verdadero', 'Falso'],
                'respuesta_correcta': 'Verdadero' if es_verdadero else 'Falso',
                'tipo': 'verdadero_falso'
            }
            
        except Exception as e:
            print(f"Error generando pregunta V/F: {str(e)}")
            return None

    def _validar_verdadero_falso(self, pregunta: str, contexto_original: str) -> bool:
        """Valida la calidad de una pregunta verdadero/falso"""
        try:
            # Verificar longitud mínima
            if len(pregunta.split()) < 8:
                return False
                
            # Verificar que no sea idéntica al contexto
            if pregunta.lower() == contexto_original.lower():
                return False
                
            # Verificar coherencia semántica
            doc_pregunta = self.nlp(pregunta)
            doc_contexto = self.nlp(contexto_original)
            
            # Verificar que contiene un verbo
            tiene_verbo = False
            for token in doc_pregunta:
                if hasattr(token, 'pos_') and token.pos_ == 'VERB':
                    tiene_verbo = True
                    break
                    
            if not tiene_verbo:
                return False
                
            # Calcular similitud con contexto original
            similitud = doc_pregunta.similarity(doc_contexto)
            if similitud < 0.3 or similitud > 0.9:  # Muy diferente o muy similar
                return False
                
            return True
            
        except Exception as e:
            print(f"Error en validación V/F: {str(e)}")
            return False
        
    def _validar_pregunta(self, pregunta: str, alternativas: List[str], respuesta: str) -> bool:
        """Valida la calidad de una pregunta y sus alternativas"""
        # Verificar estructura básica
        if not pregunta.startswith('¿') or not pregunta.endswith('?'):
            return False
            
        # Verificar longitud mínima
        if len(pregunta.split()) < 5:
            return False
            
        # Validar alternativas
        if len(alternativas) != 4 or len(set(alternativas)) != 4:
            return False
            
        if respuesta not in alternativas:
            return False
            
        # Verificar coherencia semántica entre pregunta y alternativas
        doc_pregunta = self.nlp(pregunta)
        embeddings_alt = self.semantic_model.encode(alternativas)
        
        # Verificar que las alternativas son semánticamente relacionadas
        similitudes = cosine_similarity(embeddings_alt)
        for i in range(len(alternativas)):
            for j in range(i + 1, len(alternativas)):
                if similitudes[i][j] < 0.2 or similitudes[i][j] > 0.9:
                    return False
        
        return True
        
    def _formatear_pregunta(self, pregunta: str) -> str:
        """Formatea y limpia una pregunta"""
        # Limpiar espacios
        pregunta = pregunta.strip()
        
        # Asegurar signos de interrogación
        if not pregunta.startswith('¿'):
            pregunta = '¿' + pregunta
        if not pregunta.endswith('?'):
            pregunta = pregunta + '?'
        
        # Corregir mayúsculas
        pregunta = pregunta[0] + pregunta[1].upper() + pregunta[2:]
        
        # Eliminar referencias al texto
        pregunta = re.sub(r'(?i)(en el texto|según el texto|el texto dice|el párrafo|este contexto)', '', pregunta)
        
        return pregunta.strip()
        
    def _limpiar_texto(self, texto: str) -> str:
        """Limpia y normaliza un texto"""
        # Eliminar caracteres especiales
        texto = re.sub(r'[^\w\s\?\¿\!\¡\.]', '', texto)
        
        # Normalizar espacios
        texto = ' '.join(texto.split())
        
        # Corregir puntuación
        texto = texto.replace(' .', '.').replace(' ,', ',')
        
        return texto.strip()
        
    def generar_cuestionario(self, ruta_archivo: str) -> Dict:
        """Genera cuestionarios completos a partir del archivo JSON"""
        try:
            print("Cargando datos...")
            with open(ruta_archivo, 'r', encoding='utf-8') as archivo:
                datos = json.load(archivo)
            
            print("Procesando textos y generando cuestionarios...")
            cuestionarios = {}
            
            for item in tqdm(datos['quiz'], desc="Generando cuestionarios"):
                materia = item['materia']
                texto = item['texto']
                
                if materia not in cuestionarios:
                    cuestionarios[materia] = []
                
                preguntas = []
                intentos = 0
                max_intentos = 20
                
                print(f"\nGenerando preguntas para {materia}...")
                
                while len(preguntas) < 10 and intentos < max_intentos:
                    try:
                        # Procesar el texto
                        doc = self.nlp(texto)
                        
                        # Extraer oraciones significativas
                        oraciones = [sent.text for sent in doc.sents 
                                if len(sent.text.split()) >= 8]
                        
                        if not oraciones:
                            continue
                            
                        # Seleccionar oración para generar pregunta
                        oracion = random.choice(oraciones)
                        
                        # Generar pregunta usando T5
                        prompt = f"Genera una pregunta en español sobre: {oracion}"
                        inputs = self.gen_tokenizer(prompt, return_tensors="pt", max_length=512).to(self.device)
                        
                        outputs = self.gen_model.generate(
                            **inputs,
                            max_length=128,
                            num_beams=4,
                            temperature=0.7,
                            do_sample=True
                        )
                        
                        pregunta_base = self.gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
                        
                        # Obtener respuesta usando análisis semántico
                        doc_oracion = self.nlp(oracion)
                        candidatos = []
                        
                        for chunk in doc_oracion.noun_chunks:
                            if len(chunk.text.split()) >= 2:
                                candidatos.append(chunk.text)
                        
                        if not candidatos:
                            for token in doc_oracion:
                                if token.pos_ in ['NOUN', 'PROPN', 'ADJ']:
                                    candidatos.append(token.text)
                        
                        if not candidatos:
                            continue
                            
                        respuesta = random.choice(candidatos)
                        
                        # Generar alternativas
                        alternativas = {respuesta}
                        
                        # Buscar alternativas en el texto
                        for sent in doc.sents:
                            for chunk in sent.noun_chunks:
                                if (chunk.text != respuesta and 
                                    len(chunk.text.split()) >= len(respuesta.split())):
                                    alternativas.add(chunk.text)
                        
                        # Agregar algunas alternativas similares
                        doc_resp = self.nlp(respuesta)
                        for token in doc_resp:
                            if token.has_vector:
                                similares = []
                                for word in self.nlp.vocab:
                                    if (word.has_vector and 
                                        word.is_lower and 
                                        word.prob >= -15):
                                        similitud = token.similarity(word)
                                        if 0.3 <= similitud <= 0.8:
                                            similares.append(word.text)
                                if similares:
                                    alternativas.add(random.choice(similares))
                        
                        alternativas = list(alternativas)[:4]
                        
                        if len(alternativas) == 4:
                            pregunta_final = self._formatear_pregunta(pregunta_base)
                            
                            preguntas.append({
                                'pregunta': pregunta_final,
                                'opciones': alternativas,
                                'respuesta_correcta': respuesta,
                                'tipo': 'alternativas'
                            })
                    
                    except Exception as e:
                        print(f"Error en iteración: {str(e)}")
                    
                    intentos += 1
                
                if len(preguntas) >= 8:
                    print(f"Generadas {len(preguntas)} preguntas para {materia}")
                    cuestionarios[materia].append({
                        'texto': texto,
                        'fuente': item['fuente'],
                        'preguntas': preguntas
                    })
                else:
                    print(f"No se generaron suficientes preguntas para {materia}")
                
            return cuestionarios
            
        except Exception as e:
            print(f"Error en la generación de cuestionarios: {str(e)}")
            return None

    def _formatear_pregunta(self, texto: str) -> str:
        """Formatea un texto como pregunta"""
        texto = texto.strip()
        
        # Eliminar referencias al texto
        texto = re.sub(r'(?i)(en el texto|según el texto|el texto dice|el párrafo)', '', texto)
        
        # Asegurar signos de interrogación
        if not texto.startswith('¿'):
            texto = '¿' + texto
        if not texto.endswith('?'):
            texto = texto + '?'
        
        # Corregir mayúsculas
        texto = texto[0] + texto[1].upper() + texto[2:]
        
        return texto.strip()