# Generador de Quizzes
# por Franco Benassi
import torch
from transformers import (
    AutoTokenizer, 
    T5ForConditionalGeneration,
    pipeline,
    AutoModelForQuestionAnswering
)
import spacy
import json
import random
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class DatasetPreguntas(Dataset):
    def __init__(self, textos, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.textos = textos
        self.max_length = max_length

    def __len__(self):
        return len(self.textos)

    def __getitem__(self, idx):
        texto = self.textos[idx]
        encoding = self.tokenizer(
            texto,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }

class ModeloGenerador(nn.Module):
    def __init__(self, model_name='google/mt5-small'):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)
        self.question_head = nn.Linear(self.t5.config.hidden_size, self.t5.config.vocab_size)
        self.answer_head = nn.Linear(self.t5.config.hidden_size, self.t5.config.vocab_size)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.t5.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        hidden_states = outputs.last_hidden_state
        question_logits = self.question_head(hidden_states)
        answer_logits = self.answer_head(hidden_states)
        
        return question_logits, answer_logits

class GeneradorCuestionarios:
    def __init__(self):
        print("Inicializando modelos...")
        
        # Modelo base T5 multilingüe
        self.model_name = 'google/mt5-small'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.modelo = ModeloGenerador(self.model_name)
        
        # Pipeline de Question-Answering en español
        self.qa_pipeline = pipeline(
            "question-answering",
            model="mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es",
            tokenizer="mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"
        )
        
        # Modelo para embeddings semánticos
        self.semantic_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        
        # SpaCy para análisis lingüístico
        self.nlp = spacy.load("es_core_news_lg")
        
        # Vectorizador TF-IDF
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))

    def entrenar_modelo(self, textos: List[str], epochs: int = 3):
        """Entrena el modelo con los textos proporcionados"""
        dataset = DatasetPreguntas(textos, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        optimizer = optim.AdamW(self.modelo.parameters(), lr=1e-4)
        
        self.modelo.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                
                question_logits, answer_logits = self.modelo(input_ids, attention_mask)
                
                # Pérdida para generación de preguntas y respuestas
                loss = self.compute_loss(question_logits, answer_logits, input_ids)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

    def compute_loss(self, question_logits, answer_logits, targets):
        """Calcula la pérdida combinada para preguntas y respuestas"""
        loss_fct = nn.CrossEntropyLoss()
        
        # Simplificamos usando solo el primer token como objetivo
        question_loss = loss_fct(question_logits[:, 0, :], targets[:, 0])
        answer_loss = loss_fct(answer_logits[:, 0, :], targets[:, 0])
        
        return question_loss + answer_loss

    def _extraer_conceptos_clave(self, texto: str) -> List[Dict]:
        """Extrae conceptos clave usando análisis lingüístico y TF-IDF"""
        doc = self.nlp(texto)
        
        # Análisis TF-IDF
        matriz_tfidf = self.vectorizer.fit_transform([texto])
        terminos = self.vectorizer.get_feature_names_out()
        scores = matriz_tfidf.toarray()[0]
        
        importantes = sorted(zip(terminos, scores), key=lambda x: x[1], reverse=True)[:10]
        
        conceptos = []
        for sent in doc.sents:
            if len(sent.text.split()) > 8:
                entidades = []
                for ent in sent.ents:
                    if ent.label_ in ['ORG', 'PERSON', 'LOC', 'EVENT', 'MISC']:
                        relevancia = next((score for term, score in importantes if term in ent.text.lower()), 0)
                        entidades.append({
                            'texto': ent.text,
                            'tipo': ent.label_,
                            'relevancia': relevancia
                        })
                
                if entidades:
                    conceptos.append({
                        'oracion': sent.text,
                        'entidades': entidades,
                        'relevancia': sum(e['relevancia'] for e in entidades)
                    })
        
        return conceptos

    def generar_pregunta(self, concepto: Dict, texto: str) -> Dict:
        """Genera una pregunta usando el modelo entrenado y QA"""
        try:
            # Generar pregunta
            input_text = f"generar pregunta: {concepto['oracion']}"
            input_ids = self.tokenizer(
                input_text, 
                max_length=512, 
                truncation=True, 
                return_tensors="pt"
            )['input_ids']
            
            outputs = self.modelo.t5.generate(
                input_ids,
                max_length=64,
                num_beams=4,
                length_penalty=1.0,
                early_stopping=True
            )
            
            pregunta = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Obtener respuesta usando QA
            qa_result = self.qa_pipeline(
                question=pregunta,
                context=texto
            )
            
            if qa_result['score'] < 0.1:
                return None
            
            respuesta = qa_result['answer']
            
            # Generar alternativas
            alternativas = self._generar_alternativas(respuesta, texto, concepto)
            
            if not alternativas or len(alternativas) < 4:
                return None
            
            return {
                'pregunta': pregunta,
                'opciones': alternativas,
                'respuesta_correcta': respuesta,
                'tipo': 'alternativas'
            }
            
        except Exception as e:
            print(f"Error generando pregunta: {str(e)}")
            return None

    def _generar_alternativas(self, respuesta: str, texto: str, concepto: Dict) -> List[str]:
        """Genera alternativas plausibles usando análisis semántico"""
        try:
            # Obtener embedding de la respuesta correcta
            embedding_respuesta = self.semantic_model.encode([respuesta])[0]
            
            # Extraer frases candidatas del texto
            doc = self.nlp(texto)
            candidatos = []
            
            for sent in doc.sents:
                for chunk in sent.noun_chunks:
                    if len(chunk.text.split()) >= 2:
                        embedding_chunk = self.semantic_model.encode([chunk.text])[0]
                        similitud = cosine_similarity([embedding_respuesta], [embedding_chunk])[0][0]
                        
                        if 0.3 <= similitud <= 0.7:
                            candidatos.append({
                                'texto': chunk.text,
                                'similitud': similitud
                            })
            
            # Seleccionar mejores candidatos
            candidatos.sort(key=lambda x: x['similitud'], reverse=True)
            alternativas = [respuesta]
            
            for candidato in candidatos:
                if len(alternativas) >= 4:
                    break
                if candidato['texto'] not in alternativas:
                    alternativas.append(candidato['texto'])
            
            # Asegurar que tenemos 4 alternativas
            while len(alternativas) < 4:
                nueva_alt = self._generar_alternativa_sintetica(respuesta)
                if nueva_alt and nueva_alt not in alternativas:
                    alternativas.append(nueva_alt)
            
            random.shuffle(alternativas)
            return alternativas
            
        except Exception as e:
            print(f"Error generando alternativas: {str(e)}")
            return None

    def _generar_alternativa_sintetica(self, texto: str) -> str:
        """Genera una alternativa sintética modificando el texto original"""
        doc = self.nlp(texto)
        tokens = [token.text for token in doc]
        
        # Intentar diferentes modificaciones
        estrategias = [
            self._sustituir_sustantivo,
            self._modificar_verbo,
            self._agregar_modificador
        ]
        
        for _ in range(3):  # Intentar hasta 3 veces
            estrategia = random.choice(estrategias)
            resultado = estrategia(tokens, doc)
            if resultado:
                return resultado
        
        return None

    def _sustituir_sustantivo(self, tokens: List[str], doc) -> str:
        """Sustituye un sustantivo por otro relacionado"""
        for token in doc:
            if token.pos_ == 'NOUN':
                similares = [t.text for t in token.vocab if t.is_lower and t.prob >= -15]
                if similares:
                    tokens[token.i] = random.choice(similares)
                    return ' '.join(tokens)
        return None

    def _modificar_verbo(self, tokens: List[str], doc) -> str:
        """Modifica un verbo por un sinónimo o forma relacionada"""
        for token in doc:
            if token.pos_ == 'VERB':
                similares = [t.text for t in token.vocab if t.is_lower and t.prob >= -15]
                if similares:
                    tokens[token.i] = random.choice(similares)
                    return ' '.join(tokens)
        return None

    def _agregar_modificador(self, tokens: List[str], doc) -> str:
        """Agrega un modificador al texto"""
        modificadores = ['principalmente', 'generalmente', 'ocasionalmente', 'raramente']
        posicion = random.randint(0, len(tokens))
        tokens.insert(posicion, random.choice(modificadores))
        return ' '.join(tokens)

    def generar_cuestionario(self, ruta_archivo: str) -> Dict:
        """Genera cuestionarios a partir de un archivo JSON"""
        try:
            with open(ruta_archivo, 'r', encoding='utf-8') as archivo:
                datos = json.load(archivo)
        except Exception as e:
            print(f"Error al cargar archivo: {str(e)}")
            return None

        # Extraer todos los textos para entrenamiento
        textos_entrenamiento = []
        for item in datos['quiz']:
            textos_entrenamiento.append(item['texto'])

        # Entrenar el modelo
        print("Entrenando modelo...")
        self.entrenar_modelo(textos_entrenamiento)

        cuestionarios = {}
        
        for item in datos['quiz']:
            materia = item['materia']
            texto = item['texto']
            
            if materia not in cuestionarios:
                cuestionarios[materia] = []
            
            conceptos = self._extraer_conceptos_clave(texto)
            if not conceptos:
                continue
                
            preguntas = []
            intentos = 0
            max_intentos = 30
            
            while len(preguntas) < 10 and intentos < max_intentos:
                concepto = random.choice(conceptos)
                pregunta = self.generar_pregunta(concepto, texto)
                
                if pregunta and self._validar_pregunta(pregunta, preguntas):
                    preguntas.append(pregunta)
                
                intentos += 1
            
            if len(preguntas) >= 8:
                cuestionarios[materia].append({
                    "texto": texto,
                    "fuente": item['fuente'],
                    "preguntas": preguntas
                })
            else:
                print(f"No se generaron suficientes preguntas de calidad para {materia}")
        
        return cuestionarios

    def _validar_pregunta(self, pregunta: Dict, existentes: List[Dict]) -> bool:
        """Valida la calidad y unicidad de una pregunta"""
        if not pregunta or not pregunta.get('pregunta') or not pregunta.get('opciones'):
            return False
        
        # Verificar longitud y formato
        texto_pregunta = pregunta['pregunta']
        if len(texto_pregunta.split()) < 5 or len(texto_pregunta.split()) > 30:
            return False
            
        # Verificar duplicados
        for existente in existentes:
            if existente['pregunta'] == texto_pregunta:
                return False