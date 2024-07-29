from tensorflow import keras
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from nltk.stem import WordNetLemmatizer as wnl
from keras.models import Sequential
import pandas as pd
import numpy as np
import random
import pickle
import nltk
import json
from sklearn.model_selection import train_test_split

# Descargar recursos de NLTK
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Inicializar el lematizador
lematizador = wnl()

# Cargar el archivo JSON con las preguntas y respuestas
intentos = json.loads(open('tutorApp/static/json/intentos.json', encoding='utf-8').read())

palabras = []
clases = []
documentos = []
caracteres_ignorados = ['?', '!', '¡', '¿', '.', ',', '´']  #caracteres que no interesan en preguntas y respuestas

# Declarar los patrones del json
for intento in intentos['intentos']:
    for patron in intento['patrones']:
        lista_palabras = nltk.word_tokenize(patron.lower(), language='spanish')
        
        # Eliminar los caracteres especiales
        lista_palabras = [palabra for palabra in lista_palabras if palabra not in caracteres_ignorados]
        
        palabras.extend(lista_palabras)
        documentos.append((lista_palabras, intento["tag"]))

        if intento["tag"] not in clases:
            clases.append(intento["tag"])

# Las palabras separadas de los patrones volverán a su forma raíz, sean palabras derivadas o verbos conjugados 
# que vuelvan a su forma raíz. Ej: caminaba, forma raíz: caminar.
palabras = [lematizador.lemmatize(palabra) for palabra in palabras]
palabras = sorted(set(palabras)) # Ordenar las palabras y convertirlas a un set para que tenga un buen entrenamiento

# Guardar las palabras en un archivo pickle tanto las palabras anteriores que son los patrones y las clases que serían los tag del json
pickle.dump(palabras, open('tutorApp/static/entrenamiento/palabras.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(clases, open('tutorApp/static/entrenamiento/clases.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

entrenamiento = []
salida_vacia = [0] * len(clases)

for documento in documentos:
    mochila = []
    patrones_palabra = documento[0]
    patrones_palabra = [lematizador.lemmatize(palabra.lower()) for palabra in patrones_palabra] # poner en minúsculas

    for palabra in palabras:
        mochila.append(1) if palabra in patrones_palabra else mochila.append(0)  # agregar un 1 si la palabra está en patrones_palabra
    
    salida_fila = list(salida_vacia)
    salida_fila[clases.index(documento[1])] = 1
    entrenamiento.append([mochila, salida_fila])

# Encuentra la longitud máxima de las mochilas
longitud_maxima = max(len(mochila) for mochila, _ in entrenamiento)

# Rellena las mochilas con ceros para que todas tengan la misma longitud
entrenamiento_padding = []
for mochila, salida_fila in entrenamiento:
    # Rellena con ceros hasta alcanzar la longitud máxima
    mochila.extend([0] * (longitud_maxima - len(mochila)))
    entrenamiento_padding.append((mochila, salida_fila))

random.shuffle(entrenamiento)

# Convierte a un array NumPy
entrenamiento_x = np.array([np.array(mochila) for mochila, _ in entrenamiento_padding])
entrenamiento_y = np.array([salida_fila for _, salida_fila in entrenamiento_padding])


# Convertir palabras y clases en arreglos NumPy
palabras = np.array(palabras)
clases = np.array(clases)

entrenamiento_x = []
entrenamiento_y = []

for mochila, salida_fila in entrenamiento:
    entrenamiento_x.append(mochila)
    entrenamiento_y.append(salida_fila)

# Convertir en arreglos NumPy
entrenamiento_x = np.array(entrenamiento_x)
entrenamiento_y = np.array(entrenamiento_y)

# División en datos de entrenamiento y de prueba (80% entrenamiento, 20% prueba)
x_entrenamiento, x_prueba, y_entrenamiento, y_prueba = train_test_split(entrenamiento_x, entrenamiento_y, test_size=0.2, random_state=42)

# Modelo de aprendizaje automático, red neuronal secuencial
modelo = Sequential()
# Agregado de capas
modelo.add(Dense(128, input_shape=(len(entrenamiento_x[0]),), activation='relu'))
modelo.add(Dropout(0.5))
modelo.add(Dense(64, activation='relu'))
modelo.add(Dropout(0.5))
modelo.add(Dense(len(entrenamiento_y[0]), activation='softmax'))

# Definición del optimizador
optimizador = Adam(learning_rate=0.001)
modelo.compile(loss='categorical_crossentropy', optimizer=optimizador, metrics=['accuracy'])

# Entrenamiento del modelo
proceso_entrenamiento = modelo.fit(x_entrenamiento, y_entrenamiento, validation_data=(x_prueba, y_prueba), epochs=5000, batch_size=5, verbose=1)

# Guardar el modelo entrenado
modelo.save("tutorApp/static/entrenamiento/chatbot_modelo.h5", proceso_entrenamiento)