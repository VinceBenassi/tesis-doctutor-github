# Chatbot con IA - Parte 2
from nltk.stem import WordNetLemmatizer as wnl
from keras.models import load_model
import numpy as np
import random
import pickle
import json
import nltk
import string   # para eliminar signos de puntuación

lematizador = wnl()
intentos = json.loads(open('tutorApp/static/json/intentos.json', encoding='utf-8').read())

palabras = pickle.load(open('tutorApp/static/entrenamiento/palabras.pkl', 'rb'), encoding='utf-8')
clases = pickle.load(open('tutorApp/static/entrenamiento/clases.pkl', 'rb'), encoding='utf-8')
modelo = load_model('tutorApp/static/entrenamiento/chatbot_modelo.h5')

def corregir_palabras(frase):
    palabras_a_corregir = nltk.word_tokenize(frase.lower(), language='spanish')
    
    # Lematización y eliminación de signos de puntuación
    palabras_a_corregir = [lematizador.lemmatize(palabra) for palabra in palabras_a_corregir if palabra not in string.punctuation]
    return palabras_a_corregir

def mochila_palabras(frase):
    # devolver todas las palabras de la función llamada
    palabras_a_corregir = corregir_palabras(frase)
    mochila = [0] * len(palabras)
    
    for p in palabras_a_corregir:
        for i, palabra in enumerate(palabras):
            # de todas las palabras que aparecen en palabras_a_corregir
            # si una de las palabras que ingresa el usuario está dentro de las palabras
            # definidas anteriormente pues esa palabra cambiará a 1
            if palabra == p:
                mochila[i] = 1
    return np.array(mochila)  # devolver arreglo numpy con estas palabras

def predecir_clase_frase(frase):
    bolsa_palabras = mochila_palabras(frase)
    # resultado de la predicción que se haga devolverá una probabilidad de que pertenezca a cierta categoría
    resultado = modelo.predict(np.array([bolsa_palabras]))[0]
    # devolver el indice del valor que ha salido con más probabilidades en la predicción realizada
    indice_max = np.where(resultado == np.max(resultado))[0][0]
    categoria = clases[indice_max]
    return categoria  # saber a que categoría pertenece cada palabra

def obtener_respuesta(tag, intentos_json):
    lista_intentos = intentos_json['intentos']
    resultado = ""

    for i in lista_intentos:
        if i["tag"] == tag:
            resultado = random.choice(i['respuestas'])
            break
        else:
            resultado = "No entiendo, ¿puedes explicarte mejor?"  # respuesta por defecto
    return resultado