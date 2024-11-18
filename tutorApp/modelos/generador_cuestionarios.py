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
        # Modelo de respuesta a preguntas (QA)
        self.qa_model = pipeline("question-answering", model="dccuchile/bert-base-spanish-wwm-cased")
        # Modelo de procesamiento del lenguaje natural
        self.nlp = spacy.load("es_core_news_sm")

    def analizar_texto(self, texto: str) -> Dict[str, Any]:
        """Analiza el texto para identificar conceptos clave y estructuras importantes."""
        doc = self.nlp(texto)
        oraciones = [sent.text for sent in doc.sents if len(sent.text) > 10]
        entidades = [ent.text for ent in doc.ents if ent.label_ in ["PER", "ORG", "CONCEPT", "EVENT"]]
        conceptos = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 1]

        return {
            "oraciones": oraciones,
            "entidades": list(set(entidades)),
            "conceptos": list(set(conceptos))
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
        """Genera múltiples preguntas dinámicamente."""
        analisis = self.analizar_texto(texto)
        preguntas = []

        while len(preguntas) < num_preguntas:
            tipo_pregunta = random.choice(["alternativas", "verdadero_falso"])
            
            if tipo_pregunta == "alternativas":
                pregunta = self.generar_pregunta_alternativas(analisis, texto)
            else:
                pregunta = self.generar_pregunta_verdadero_falso(analisis, texto)

            if pregunta:
                preguntas.append(pregunta)

        return preguntas

    def generar_pregunta_alternativas(self, analisis: Dict[str, Any], texto: str) -> Dict[str, Any]:
        """Genera una pregunta de opción múltiple con distractores relevantes."""
        if not analisis["conceptos"]:
            return None

        concepto = random.choice(analisis["conceptos"])
        respuesta_correcta = self.qa_model(
            question=f"¿Qué implica el concepto {concepto}?", context=texto
        )["answer"]

        distractores = self.generar_distractores(respuesta_correcta, texto, concepto)

        if len(distractores) < 3:
            return None

        opciones = [respuesta_correcta] + distractores[:3]
        random.shuffle(opciones)

        return {
            "tipo": "alternativas",
            "pregunta": f"¿Cuál es la mejor descripción de {concepto}?",
            "opciones": opciones,
            "respuesta_correcta": respuesta_correcta,
            "explicacion": f"{concepto} se refiere a '{respuesta_correcta}', como se menciona en el texto."
        }

    def generar_pregunta_verdadero_falso(self, analisis: Dict[str, Any], texto: str) -> Dict[str, Any]:
        """Genera preguntas de verdadero/falso más coherentes."""
        if not analisis["oraciones"]:
            return None

        oracion = random.choice(analisis["oraciones"])
        es_verdadero = random.choice([True, False])

        if not es_verdadero:
            oracion = self.alterar_oracion(oracion)

        return {
            "tipo": "verdadero_falso",
            "pregunta": oracion,
            "respuesta_correcta": "Verdadero" if es_verdadero else "Falso",
            "explicacion": f"La afirmación es {'correcta' if es_verdadero else 'incorrecta'} porque refleja el contexto del texto."
        }

    def generar_distractores(self, respuesta: str, texto: str, concepto: str) -> List[str]:
        """Genera distractores semánticamente relevantes."""
        doc = self.nlp(texto)
        distractores = []

        for sent in doc.sents:
            if concepto.lower() in sent.text.lower() and respuesta.lower() not in sent.text.lower():
                distractores.append(sent.text.strip())

        return list(set(distractores))

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