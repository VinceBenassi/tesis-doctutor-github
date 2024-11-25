# Generador de Cuestionarios
# Por Franco Benassi
import torch
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    pipeline
)
from sentence_transformers import SentenceTransformer
import spacy
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

class GeneradorVF:
    def __init__(self):
        """Inicializa modelos de question answering"""
        try:
            # Modelo QA específico para español
            self.model_name = "mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"
            
            # Inicializar pipeline de QA
            self.qa_pipeline = pipeline(
                'question-answering',
                model=self.model_name,
                tokenizer=self.model_name,
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Modelo para similitud semántica
            self.sentence_model = SentenceTransformer('hiiamsid/sentence_similarity_spanish_es')
            
            # Modelo spaCy para análisis de texto
            self.nlp = spacy.load("es_core_news_lg")
            
            print(f"Modelos cargados en {'GPU' if torch.cuda.is_available() else 'CPU'}")
            
        except Exception as e:
            print(f"Error inicializando modelos: {str(e)}")
            raise

    def generar_cuestionario(self, texto: str, materia: str, 
                           fuente: str, num_preguntas: int = 10) -> Dict[str, Any]:
        """Genera cuestionario de verdadero/falso basado en el texto"""
        try:
            # Procesar texto
            doc = self.nlp(texto)
            oraciones = [sent for sent in doc.sents if len(sent.text.split()) > 5]
            
            if not oraciones:
                return None
                
            preguntas = []
            intentos = 0
            max_intentos = num_preguntas * 3
            
            while len(preguntas) < num_preguntas and intentos < max_intentos:
                pregunta = self._generar_pregunta_vf(texto, oraciones)
                
                if pregunta and self._es_pregunta_valida(pregunta, preguntas):
                    preguntas.append(pregunta)
                    
                intentos += 1
                
            return {
                "materia": materia,
                "fuente": fuente,
                "preguntas": preguntas[:num_preguntas]
            }
            
        except Exception as e:
            print(f"Error generando cuestionario: {str(e)}")
            return None

    def _generar_pregunta_vf(self, texto_completo: str, 
                            oraciones: List[spacy.tokens.Span]) -> Optional[Dict[str, Any]]:
        """Genera una pregunta individual de verdadero/falso"""
        try:
            # Seleccionar oración base
            oracion = oraciones[torch.randint(0, len(oraciones), (1,)).item()]
            oracion_texto = oracion.text
            
            # Obtener información relevante usando QA
            info_clave = self._extraer_info_relevante(oracion_texto, texto_completo)
            if not info_clave:
                return None
                
            # Decidir si será verdadera o falsa
            es_verdadero = bool(torch.randint(0, 2, (1,)).item())
            
            if es_verdadero:
                afirmacion = oracion_texto
            else:
                afirmacion = self._generar_variacion_falsa(oracion_texto, texto_completo)
                
            if not afirmacion:
                return None
                
            # Generar explicación
            explicacion = self._generar_explicacion(
                afirmacion,
                oracion_texto,
                texto_completo,
                es_verdadero
            )
            
            return {
                "tipo": "verdadero_falso",
                "pregunta": afirmacion,
                "respuesta_correcta": "Verdadero" if es_verdadero else "Falso",
                "explicacion": explicacion,
                "opciones": None
            }
            
        except Exception as e:
            print(f"Error generando pregunta V/F: {str(e)}")
            return None

    def _extraer_info_relevante(self, oracion: str, contexto: str) -> Optional[Dict[str, Any]]:
        """Extrae información relevante usando question answering"""
        try:
            # Hacer preguntas de análisis usando QA
            respuestas = []
            
            # Usar QA para extraer información clave
            resultado = self.qa_pipeline(
                question="¿Cuál es el tema principal?",
                context=oracion,
                handle_impossible_answer=True
            )
            
            if resultado['score'] > 0.1:
                respuestas.append(resultado['answer'])
            
            # Verificar contexto más amplio
            resultado_contexto = self.qa_pipeline(
                question=f"¿Qué se menciona sobre {' '.join(respuestas)}?",
                context=contexto,
                handle_impossible_answer=True
            )
            
            if resultado_contexto['score'] > 0.1:
                respuestas.append(resultado_contexto['answer'])
                
            return {
                'respuestas': respuestas,
                'score': resultado['score']
            } if respuestas else None
            
        except Exception as e:
            print(f"Error extrayendo información: {str(e)}")
            return None

    def _generar_variacion_falsa(self, oracion: str, contexto: str) -> Optional[str]:
        """Genera una variación falsa de la oración"""
        try:
            # Obtener información contradictoria del contexto
            doc = self.nlp(oracion)
            
            # Identificar el verbo principal
            verbo_principal = None
            for token in doc:
                if token.pos_ == "VERB" and token.dep_ == "ROOT":
                    verbo_principal = token
                    break
            
            if verbo_principal:
                # Buscar información contradictoria
                resultado = self.qa_pipeline(
                    question=f"¿Qué otros hechos diferentes se mencionan sobre esto?",
                    context=contexto,
                    handle_impossible_answer=True
                )
                
                if resultado['score'] > 0.3:
                    return resultado['answer']
                    
            # Si no se encuentra contradicción, negar la oración
            return self._negar_oracion(oracion)
            
        except Exception as e:
            print(f"Error generando variación falsa: {str(e)}")
            return None

    def _negar_oracion(self, oracion: str) -> str:
        """Niega una oración de forma coherente"""
        doc = self.nlp(oracion)
        tokens = []
        verbo_negado = False
        
        for token in doc:
            if token.pos_ == "VERB" and not verbo_negado:
                tokens.extend(["no", token.text])
                verbo_negado = True
            else:
                tokens.append(token.text)
                
        return " ".join(tokens)

    def _generar_explicacion(self, afirmacion: str, oracion_original: str, 
                           contexto: str, es_verdadero: bool) -> str:
        """Genera una explicación detallada"""
        try:
            # Obtener evidencia usando QA
            evidencia = self.qa_pipeline(
                question=f"¿Qué evidencia respalda o contradice esto?",
                context=contexto,
                handle_impossible_answer=True
            )
            
            # Construir explicación
            explicacion = f"La afirmación es {'VERDADERA' if es_verdadero else 'FALSA'}. "
            
            if evidencia['score'] > 0.3:
                explicacion += evidencia['answer'] + " "
            
            # Agregar referencia al texto
            if es_verdadero:
                explicacion += f"\nEsto se evidencia en el texto cuando menciona: '{oracion_original}'"
            else:
                explicacion += f"\nLa versión correcta según el texto es: '{oracion_original}'"
            
            return explicacion
            
        except Exception as e:
            print(f"Error generando explicación: {str(e)}")
            return f"La afirmación es {'VERDADERA' if es_verdadero else 'FALSA'}."

    def _es_pregunta_valida(self, pregunta: Dict[str, Any], 
                           preguntas_previas: List[Dict[str, Any]]) -> bool:
        """Valida la calidad de la pregunta"""
        try:
            if not pregunta['pregunta'] or not pregunta['explicacion']:
                return False
            
            # Verificar longitud mínima
            if len(pregunta['pregunta'].split()) < 8:
                return False
                
            if len(pregunta['explicacion'].split()) < 30:
                return False
            
            # Verificar duplicados
            pregunta_emb = self.sentence_model.encode(pregunta['pregunta'])
            
            for prev in preguntas_previas:
                prev_emb = self.sentence_model.encode(prev['pregunta'])
                similitud = float(torch.cosine_similarity(
                    torch.tensor(pregunta_emb),
                    torch.tensor(prev_emb),
                    dim=0
                ))
                
                if similitud > 0.8:
                    return False
            
            return True
            
        except Exception as e:
            print(f"Error validando pregunta: {str(e)}")
            return False

    def procesar_json(self, ruta_entrada: str, ruta_salida: str) -> None:
        """Procesa archivo JSON de entrada y genera cuestionarios"""
        try:
            # Cargar datos
            with open(ruta_entrada, 'r', encoding='utf-8') as f:
                datos = json.load(f)
            
            if 'quiz' not in datos:
                raise ValueError("Formato JSON inválido")
            
            # Generar cuestionarios
            cuestionarios = []
            for item in datos['quiz']:
                cuestionario = self.generar_cuestionario(
                    texto=item['texto'],
                    materia=item['materia'],
                    fuente=item['fuente']
                )
                
                if cuestionario:
                    cuestionarios.append(cuestionario)
            
            # Guardar resultados
            salida = {
                "fecha_generacion": datetime.now().strftime("%Y-%m-%d"),
                "total_cuestionarios": len(cuestionarios),
                "cuestionarios": cuestionarios
            }
            
            with open(ruta_salida, 'w', encoding='utf-8') as f:
                json.dump(salida, f, ensure_ascii=False, indent=4)
                
            print(f"Se generaron {len(cuestionarios)} cuestionarios")
            
        except Exception as e:
            print(f"Error procesando JSON: {str(e)}")