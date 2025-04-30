# Explicación Detallada de Fórmulas Matemáticas en DocTutor

Este documento proporciona una explicación detallada de las fórmulas matemáticas y algoritmos utilizados en los componentes principales del proyecto DocTutor.

## 1. Chatbot (chatbot.py)

El chatbot utiliza un modelo de procesamiento de lenguaje natural basado en redes neuronales para clasificar las intenciones del usuario.

### Vectorización de palabras (Bag of Words)

```python
mochila = [0] * len(palabras)
for p in palabras_a_corregir:
    for i, palabra in enumerate(palabras):
        if palabra == p:
            mochila[i] = 1
```

**Explicación**: Esta técnica convierte el texto en un vector binario donde cada posición representa la presencia (1) o ausencia (0) de una palabra específica del vocabulario. Es una representación numérica del texto que permite al modelo procesar lenguaje natural.

1. Se crea un vector de ceros con longitud igual al vocabulario total
2. Para cada palabra en el texto de entrada, se busca su posición en el vocabulario
3. Si la palabra existe, se marca con un 1 en la posición correspondiente

Esta representación vectorial permite transformar texto en datos numéricos que pueden ser procesados por el modelo de redes neuronales.

### Predicción de clase mediante redes neuronales

```python
resultado = modelo.predict(np.array([bolsa_palabras]))[0]
indice_max = np.where(resultado == np.max(resultado))[0][0]
```

**Explicación**: El modelo de redes neuronales procesa el vector de palabras y genera probabilidades para cada clase posible. Se selecciona la clase con mayor probabilidad.

1. `modelo.predict()` aplica la red neuronal al vector de entrada y devuelve un vector de probabilidades
2. `np.max(resultado)` encuentra el valor máximo (mayor probabilidad)
3. `np.where()` obtiene el índice de ese valor máximo, que corresponde a la clase predicha

La arquitectura de la red neuronal utilizada es secuencial con capas densas:

```python
modelo = Sequential()
modelo.add(Dense(128, input_shape=(len(entrenamiento_x[0]),), activation='relu'))
modelo.add(Dropout(0.5))
modelo.add(Dense(64, activation='relu'))
modelo.add(Dropout(0.5))
modelo.add(Dense(len(entrenamiento_y[0]), activation='softmax'))
```

Esta red neuronal tiene:
- Una capa de entrada con 128 neuronas y activación ReLU
- Una capa oculta con 64 neuronas y activación ReLU
- Una capa de salida con activación softmax que genera probabilidades para cada clase
- Capas de Dropout (0.5) para prevenir el sobreajuste

## 2. Clasificador PDF (clasificadorPDF.py)

Este módulo clasifica documentos PDF en categorías temáticas utilizando técnicas de clustering y análisis de texto.

### TF-IDF (Term Frequency-Inverse Document Frequency)

```python
vectorizador = TfidfVectorizer(ngram_range=(1, 2))
X = vectorizador.fit_transform(nombres_preprocesados)
```

**Explicación**: TF-IDF es una técnica de vectorización que asigna pesos a las palabras basándose en su frecuencia en el documento y su rareza en el corpus completo.

La fórmula matemática de TF-IDF es:

$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$

Donde:
- $\text{TF}(t, d)$ es la frecuencia del término $t$ en el documento $d$
- $\text{IDF}(t, D) = \log\frac{|D|}{|\{d \in D : t \in d\}|}$ donde $|D|$ es el número total de documentos y $|\{d \in D : t \in d\}|$ es el número de documentos que contienen el término $t$

Esta técnica:
1. Da mayor peso a palabras que aparecen frecuentemente en un documento específico
2. Reduce el peso de palabras comunes que aparecen en muchos documentos
3. Permite representar documentos como vectores numéricos para análisis

El parámetro `ngram_range=(1, 2)` indica que se consideran tanto palabras individuales como pares de palabras consecutivas.

### DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

```python
agrupamiento = DBSCAN(eps=0.5, min_samples=1)
etiquetas_cluster = agrupamiento.fit_predict(X.toarray())
```

**Explicación**: DBSCAN es un algoritmo de clustering que agrupa puntos basándose en su densidad en el espacio vectorial.

El algoritmo funciona así:
1. Para cada punto, encuentra todos los puntos vecinos dentro de una distancia `eps`
2. Si un punto tiene al menos `min_samples` vecinos, se considera un punto central
3. Los puntos centrales y sus vecinos forman un cluster
4. Los puntos que no son centrales ni vecinos de ningún punto central se consideran ruido

Los parámetros utilizados son:
- `eps=0.5`: Define la distancia máxima entre dos puntos para considerarlos vecinos
- `min_samples=1`: Número mínimo de puntos para formar un cluster

Este algoritmo es útil porque:
- No requiere especificar el número de clusters de antemano
- Puede encontrar clusters de formas arbitrarias
- Identifica puntos de ruido que no pertenecen a ningún cluster

## 3. Generador de Cuestionarios (generador_cuestionarios.py)

Este módulo genera preguntas de tipo verdadero/falso utilizando modelos de question answering y similitud semántica.

### Similitud coseno para validación de preguntas

```python
similitud = float(torch.cosine_similarity(
    torch.tensor(pregunta_emb),
    torch.tensor(prev_emb),
    dim=0
))
```

**Explicación**: La similitud coseno mide el ángulo entre dos vectores, calculando cuán similares son dos textos en el espacio vectorial semántico.

La fórmula matemática es:

$\cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|}$

Donde:
- $A \cdot B$ es el producto escalar de los vectores
- $\|A\|$ y $\|B\|$ son las normas (magnitudes) de los vectores

Esta medida:
1. Produce un valor entre -1 y 1, donde 1 indica vectores idénticos
2. Es independiente de la magnitud de los vectores, solo considera su dirección
3. Permite detectar preguntas semánticamente similares para evitar duplicados

En el código, se utiliza un umbral de 0.8 para determinar si dos preguntas son demasiado similares:

```python
if similitud > 0.8:
    return False  # La pregunta es demasiado similar a una existente
```

### Puntuación de confianza en question answering

```python
resultado = self.qa_pipeline(
    question="¿Cuál es el tema principal?",
    context=oracion,
    handle_impossible_answer=True
)
if resultado['score'] > 0.1:
    respuestas.append(resultado['answer'])
```

**Explicación**: El modelo de question answering asigna una puntuación de confianza a cada respuesta generada, utilizando un umbral para filtrar respuestas poco confiables.

Esta puntuación:
1. Representa la probabilidad estimada de que la respuesta sea correcta
2. Se calcula mediante un modelo de lenguaje entrenado específicamente para responder preguntas
3. Permite filtrar respuestas con baja confianza usando umbrales (0.1, 0.3)

El modelo utilizado es `mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es`, una variante de BERT especializada en español y ajustada para tareas de question answering.

## 4. Traductor (traductor.py)

Este módulo detecta el idioma y traduce texto utilizando técnicas de procesamiento de lenguaje natural.

### Cálculo de puntuación para detección de idioma

```python
df['nota'] = (df['palabras']+df['stopwords'])*df['valor']*df['match']
```

**Explicación**: Esta fórmula pondera múltiples factores para determinar la confianza en la detección del idioma.

Los componentes son:
- `palabras`: Número de palabras reconocidas en el texto
- `stopwords`: Número de palabras vacías (como "el", "la", "en") detectadas en el idioma
- `valor`: Porcentaje de confianza en la detección del idioma por el modelo spaCy
- `match`: Número de veces que se detectó el mismo idioma en diferentes pruebas

La fórmula multiplica estos factores para obtener una puntuación compuesta que:
1. Aumenta con el número de palabras reconocidas
2. Aumenta con el número de stopwords detectadas (indicador fuerte del idioma)
3. Se escala por la confianza del modelo y la consistencia de detección

### Intersección de conjuntos para validación de idioma

```python
len(relacion_texto_palabras(traductor_voz.split(), lista_fr))
```

Donde `relacion_texto_palabras` implementa:

```python
return set(lst1).intersection(lst2)
```

**Explicación**: Esta operación encuentra las palabras comunes entre el texto traducido y las stopwords de cada idioma, utilizando la teoría de conjuntos.

El proceso:
1. Convierte las listas de palabras en conjuntos (estructuras de datos que eliminan duplicados)
2. Calcula la intersección matemática entre ambos conjuntos
3. Cuenta el número de elementos en la intersección

Esta técnica es eficiente para determinar cuántas palabras vacías específicas de un idioma están presentes en el texto, lo que proporciona una fuerte señal sobre el idioma del texto.

## 5. Arquitectura de la Red Neuronal del Chatbot

El archivo `entrenamientoChatbot.py` implementa la arquitectura de la red neuronal utilizada por el chatbot para clasificar intenciones.

```python
modelo = Sequential()
modelo.add(Dense(128, input_shape=(len(entrenamiento_x[0]),), activation='relu'))
modelo.add(Dropout(0.5))
modelo.add(Dense(64, activation='relu'))
modelo.add(Dropout(0.5))
modelo.add(Dense(len(entrenamiento_y[0]), activation='softmax'))
```

**Explicación**: Esta arquitectura implementa una red neuronal feed-forward con las siguientes características:

1. **Capa de entrada**: 128 neuronas con función de activación ReLU
   - La dimensión de entrada corresponde al tamaño del vector de palabras (bag of words)
   - La función ReLU (Rectified Linear Unit) se define como $f(x) = max(0, x)$, permitiendo activaciones no lineales

2. **Dropout (0.5)**: Técnica de regularización que desactiva aleatoriamente el 50% de las neuronas durante el entrenamiento
   - Matemáticamente, multiplica las activaciones por una máscara binaria donde cada elemento tiene probabilidad 0.5 de ser 0
   - Previene el sobreajuste al forzar a la red a aprender representaciones más robustas

3. **Capa oculta**: 64 neuronas con activación ReLU
   - Reduce la dimensionalidad y extrae características de alto nivel

4. **Capa de salida**: Número de neuronas igual al número de clases (intenciones)
   - Utiliza la función de activación softmax: $\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$
   - Convierte las salidas en probabilidades que suman 1, donde cada valor representa la probabilidad de pertenecer a cada clase

El entrenamiento utiliza:
- **Optimizador Adam**: Algoritmo de optimización que adapta la tasa de aprendizaje para cada parámetro
- **Función de pérdida categorical_crossentropy**: Mide la diferencia entre la distribución de probabilidad predicha y la real
- **Métrica de accuracy**: Porcentaje de predicciones correctas

## 6. Detalles Adicionales del Algoritmo DBSCAN

El algoritmo DBSCAN utilizado en el clasificador PDF implementa las siguientes operaciones matemáticas:

1. **Cálculo de distancias**: Utiliza la distancia euclidiana entre vectores TF-IDF
   - $d(p,q) = \sqrt{\sum_{i=1}^{n} (q_i - p_i)^2}$
   - Donde $p$ y $q$ son vectores TF-IDF de dos documentos

2. **Expansión de clusters**:
   - Para cada punto $p$ en el conjunto de datos:
     - Si $p$ no ha sido visitado, marcarlo como visitado
     - Si $p$ tiene al menos `min_samples` vecinos dentro de la distancia `eps`, crear un nuevo cluster
     - Agregar todos los vecinos de $p$ a una cola
     - Para cada punto en la cola, repetir el proceso de expansión

3. **Asignación de etiquetas**:
   - Puntos centrales: Puntos con al menos `min_samples` vecinos
   - Puntos fronterizos: Puntos que están dentro de la distancia `eps` de un punto central
   - Puntos de ruido: Puntos que no son ni centrales ni fronterizos (etiquetados como -1)

En el contexto del clasificador PDF, este algoritmo permite agrupar documentos con nombres similares sin necesidad de especificar el número de categorías de antemano, adaptándose automáticamente a la estructura natural de los datos.

## 7. Algoritmo de Similitud Semántica en el Generador de Cuestionarios

El generador de cuestionarios utiliza modelos de embeddings para representar textos en espacios vectoriales de alta dimensión:

```python
self.sentence_model = SentenceTransformer('hiiamsid/sentence_similarity_spanish_es')
```

**Explicación**: El modelo SentenceTransformer convierte oraciones en vectores densos que capturan su significado semántico.

1. **Proceso de embedding**:
   - Cada texto se tokeniza y se procesa a través de una arquitectura transformer
   - Se obtiene un vector de embedding de dimensión fija (generalmente 768 o 1024)
   - Estos vectores posicionan textos semánticamente similares cerca en el espacio vectorial

2. **Aplicaciones en el generador**:
   - **Detección de duplicados**: Evita preguntas redundantes calculando la similitud coseno
   - **Validación de calidad**: Asegura que las preguntas generadas sean semánticamente coherentes
   - **Selección de contenido relevante**: Identifica las partes más importantes del texto para generar preguntas

3. **Ventajas del enfoque vectorial**:
   - Captura relaciones semánticas más allá de la coincidencia exacta de palabras
   - Permite comparar textos de diferentes longitudes
   - Funciona bien con el español al utilizar un modelo específicamente entrenado para este idioma

Este enfoque basado en embeddings representa el estado del arte en procesamiento de lenguaje natural, permitiendo generar cuestionarios de alta calidad que capturan los conceptos clave del material educativo.