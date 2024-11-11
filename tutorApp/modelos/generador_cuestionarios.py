# Generador de Cuestionarios
# Por Franco Benassi
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import spacy
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import random
import json
import re

class GeneradorCuestionarios:
    def __init__(self):
        # Modelo específico para español
        self.model_name = "PlanTL-GOB-ES/T5-base-spanish"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.nlp = spacy.load('es_core_news_lg')
        
        # Pipeline optimizado para español
        self.generator = pipeline(
            "text2text-generation",
            model=self.model_name,
            tokenizer=self.model_name,
            max_length=512
        )
        
        nltk.download('punkt', quiet=True)
        self.sentence_model = SentenceTransformer('hiiamsid/sentence_similarity_spanish_es')

    def _extraer_oraciones_clave(self, texto):
        """Extrae oraciones relevantes del texto."""
        doc = self.nlp(texto)
        oraciones = [sent.text.strip() for sent in doc.sents 
                    if len(sent.text.split()) > 10]  # Filtrar oraciones muy cortas
        
        if len(oraciones) < 10:
            return oraciones
            
        # Vectorizar y agrupar oraciones significativas
        embeddings = self.sentence_model.encode(oraciones)
        kmeans = KMeans(n_clusters=min(10, len(oraciones)), random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        oraciones_seleccionadas = []
        for i in range(kmeans.n_clusters):
            cluster_sentences = [s for j, s in enumerate(oraciones) if clusters[j] == i]
            if cluster_sentences:
                # Seleccionar la oración más informativa del cluster
                mejor_oracion = max(cluster_sentences, 
                                  key=lambda x: len(set([t.text for t in self.nlp(x) 
                                                       if t.pos_ in ['NOUN', 'VERB', 'ADJ']])))
                oraciones_seleccionadas.append(mejor_oracion)
        
        return oraciones_seleccionadas

    def _generar_pregunta_alternativas(self, oracion):
        """Genera una pregunta de opción múltiple coherente."""
        doc = self.nlp(oracion)
        
        # Extraer conceptos clave
        conceptos = [ent.text for ent in doc.ents]
        if not conceptos:
            conceptos = [token.text for token in doc 
                        if token.pos_ in ['NOUN', 'PROPN'] 
                        and len(token.text) > 3]
        
        if not conceptos:
            return None
            
        concepto_principal = max(conceptos, key=lambda x: len(x))
        
        # Generar pregunta contextualizada
        prompt = f"Genera una pregunta completa sobre '{concepto_principal}' basada en: {oracion}"
        pregunta = self.generator(prompt, max_length=128)[0]['generated_text']
        
        # Limpiar y formatear la pregunta
        pregunta = re.sub(r'\s+', ' ', pregunta).strip()
        if not pregunta.endswith('?'):
            pregunta += '?'
            
        # Generar alternativas coherentes
        opciones = self._generar_alternativas_relevantes(doc, concepto_principal)
        random.shuffle(opciones)
        
        # Generar explicación contextualizada
        explicacion = self._generar_explicacion_contextualizada(oracion, concepto_principal, True)
        
        return {
            "tipo": "alternativas",
            "pregunta": pregunta,
            "opciones": opciones,
            "respuesta_correcta": concepto_principal,
            "explicacion": explicacion
        }

    def _generar_alternativas_relevantes(self, doc, respuesta_correcta):
        """Genera alternativas semánticamente relacionadas."""
        alternativas = set([respuesta_correcta])
        tokens_relevantes = [token for token in doc 
                           if token.pos_ in ['NOUN', 'PROPN', 'VERB'] 
                           and len(token.text) > 3]
        
        # Generar alternativas del mismo tipo semántico
        for token in tokens_relevantes:
            if token.has_vector and len(alternativas) < 4:
                similares = token.vocab.vectors.most_similar(
                    token.vector.reshape(1, -1),
                    n=10
                )
                for similar in similares:
                    palabra = token.vocab.strings[similar[0]]
                    if (palabra != respuesta_correcta and 
                        len(palabra) > 3 and 
                        palabra not in alternativas):
                        alternativas.add(palabra)
                        if len(alternativas) == 4:
                            break
                            
        # Completar si faltan alternativas
        while len(alternativas) < 4:
            if tokens_relevantes:
                alternativas.add(random.choice(tokens_relevantes).text)
            else:
                alternativas.add(f"Opción {len(alternativas) + 1}")
                
        return list(alternativas)

    def _generar_pregunta_verdadero_falso(self, oracion):
        """Genera una pregunta de verdadero/falso con modificación significativa."""
        doc = self.nlp(oracion)
        es_verdadero = random.choice([True, False])
        
        if not es_verdadero:
            # Identificar componentes clave para modificación
            sujetos = [token for token in doc if token.dep_ == "nsubj"]
            verbos = [token for token in doc if token.pos_ == "VERB"]
            objetos = [token for token in doc if token.dep_ in ["dobj", "pobj"]]
            
            if sujetos and verbos:
                # Modificar estructura significativamente
                modificacion = random.choice([
                    lambda x: x.replace(sujetos[0].text, random.choice([t.text for t in doc if t.pos_ == "NOUN" and t != sujetos[0]])),
                    lambda x: x.replace(verbos[0].text, random.choice([t.text for t in doc if t.pos_ == "VERB" and t != verbos[0]])),
                    lambda x: x.replace(objetos[0].text, random.choice([t.text for t in doc if t.pos_ == "NOUN" and t != objetos[0]])) if objetos else x
                ])
                oracion_modificada = modificacion(oracion)
            else:
                oracion_modificada = oracion
        else:
            oracion_modificada = oracion
            
        explicacion = self._generar_explicacion_contextualizada(oracion, oracion_modificada, False)
        
        return {
            "tipo": "verdadero_falso",
            "pregunta": oracion_modificada,
            "respuesta_correcta": "Verdadero" if es_verdadero else "Falso",
            "explicacion": explicacion
        }

    def _generar_explicacion_contextualizada(self, contexto, respuesta, es_alternativas):
        """Genera una explicación coherente y contextualizada."""
        if es_alternativas:
            prompt = (
                f"Explica detalladamente por qué '{respuesta}' es la respuesta correcta "
                f"en el siguiente contexto: {contexto}"
            )
        else:
            prompt = (
                f"Explica detalladamente por qué esta afirmación es "
                f"{'correcta' if respuesta == contexto else 'incorrecta'}: {contexto}"
            )
        
        explicacion = self.generator(prompt, max_length=256)[0]['generated_text']
        return explicacion.strip()

    def generar_cuestionario(self, json_path):
        """Genera cuestionarios completos a partir del JSON."""
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
        cuestionarios = []
        
        for entrada in data['quiz']:
            texto = entrada['texto']
            oraciones = self._extraer_oraciones_clave(texto)
            preguntas = []
            
            for i, oracion in enumerate(oraciones):
                if len(preguntas) >= 10:
                    break
                    
                pregunta = (self._generar_pregunta_alternativas(oracion) 
                          if i % 2 == 0 
                          else self._generar_pregunta_verdadero_falso(oracion))
                
                if pregunta:
                    preguntas.append(pregunta)
            
            if preguntas:
                cuestionario = {
                    "materia": entrada['materia'],
                    "fuente": entrada['fuente'],
                    "preguntas": preguntas
                }
                cuestionarios.append(cuestionario)
        
        # Guardar cuestionarios
        output_path = 'tutorApp/static/json/cuestionarios.json'
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(cuestionarios, file, ensure_ascii=False, indent=4)
            
        return cuestionarios