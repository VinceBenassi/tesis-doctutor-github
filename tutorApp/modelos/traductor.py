# Traductor de idiomas hecho
# con procesamiento de lenguaje
# natural para DocTutor
# por Franco Benassi
from spacy_langdetect import LanguageDetector
from spacy.language import Language
from googletrans import Translator # Librería para traducir la voz o texto de un idioma a otro
from nltk.corpus import stopwords
import speech_recognition as sr
import pandas as pd
from gtts import gTTS
import pyttsx3    # Librería para convertir texto a voz
import spacy
import nltk
import time

# Descargar stopwords si no están disponibles
try:
    nltk.data.find('corpora/stopwords') 
except LookupError:
    nltk.download('stopwords')


#Funciones para inicializar Spacy
def obtener_detector(nlp, name):
    return LanguageDetector()

# Cargar modelo de lenguaje 
nlp = spacy.load("en_core_web_sm")

try:
    Language.factory("language_detector", func=obtener_detector)
except Exception as e: 
    next
nlp.add_pipe('language_detector', last=True)

# Función para relacionar el texto traducido
# con las palabras vacías definidas
def relacion_texto_palabras(lst1, lst2):
    return set(lst1).intersection(lst2)

#Inicializando variables
grabador = sr.Recognizer()  # Grabar la voz
grabador.energy_threshold = 300
traductor_google = Translator() # iniciar traductor de google

# Convertir el texto en una voz llamada juan
convertidor_voz = pyttsx3.init()
convertidor_voz.setProperty('rate', 200)
convertidor_voz.setProperty('volume', 1)
convertidor_voz.setProperty('voice','com.apple.speech.synthesis.voice.juan')

# Generar listas con algunas palabras vacías (stopwords)
# que no son reconocidas por los robots de google y carecen de
# sentido al usarlas solas. Ej: aún, ante, antes, en, y, etc. en
# algunos idiomas diferentes.
lista_fr = stopwords.words('french')
lista_pt = stopwords.words('portuguese')
lista_es = stopwords.words('spanish')
lista_en = stopwords.words('english')

# Función para reconocer el sonido
def reconocer(sonido):
    idiomas = ['fr-FR', 'pt-BR', 'es-ES', 'en-US']  # idiomas que se usaran para traducir un audio sin saber el idioma original
    fr, pt, es, en = 0, 0, 0, 0

    # Dataframe para 
    df = pd.DataFrame(columns = ['traductor_voz', 'idioma', 'valor', 'palabras', 'stopwords', 'idioma_final', 'match', 'nota'])
    idioma_final = 'Ninguno'
    
    for elemento in idiomas:
        try:
            # traducir el audio recibido a los idiomas en cada elemento del arreglo de los idiomas
            traductor_voz = grabador.recognize_google(sonido, language=elemento)
            
            # Descifrar con spacy que idioma es el del audio con la 
            # probabilidad de que el audio realmente es el idioma 
            # especificado por el descifrado
            valor     = nlp(traductor_voz)._.language.get('score') 
            idioma    = nlp(traductor_voz)._.language.get('language') # devolver el nombre del idioma descifrado
            palabras  = len(traductor_voz.split()) # contar la cantidad de palabras reconocidas del audio por la función
            stopwords = 0

            # Si se reconoce alguno de los idiomas, que se guarden las
            # stopwords, el contador de cada idioma y en la función interseccion
            # relacionar el texto al cual fue traducido la voz del usuario con cada
            # arreglo con palabras de los siguientes idiomas para saber cuantas palabras 
            # en francés fueron reconocidas por la traducción y compararlas con cada arreglo
            # de los idiomas
            if idioma == 'fr': stopwords, fr, idioma_final = len(relacion_texto_palabras(traductor_voz.split(), lista_fr)), (fr+1), 'Frances'
            if idioma == 'pt': stopwords, pt, idioma_final = len(relacion_texto_palabras(traductor_voz.split(), lista_pt)), (pt+1), 'Portugues'
            if idioma == 'es': stopwords, es, idioma_final = len(relacion_texto_palabras(traductor_voz.split(), lista_es)), (es+1), 'Español'
            if idioma == 'en': stopwords, en, idioma_final = len(relacion_texto_palabras(traductor_voz.split(), lista_en)), (en+1), 'Ingles'

            # Guardar la información anterior en el dataframe sabiendo si el idioma de la voz usuario traducida
            # y el traductor de spacy realmente detectó que el usuario estaba hablando en el idioma que detectó
            # pasando en una fila del dataframe el texto traducido, el idioma, las palabras y demás.
            if idioma == elemento[0:2]: df.loc[len(df)] = [traductor_voz, idioma, valor, palabras, stopwords, idioma_final, 0, 0.0]

        except Exception as e:
            print(e)
            next
    

    # Recorrer el dataframe para generar los valores de las ultimas 2 columnas
    # match y nota, donde match guardará cuantos idiomas de los establecidos
    # detectaron el idioma con el que estaba hablando el usuario.
    # el usuario buscó traducir su voz a 4 idiomas diferentes y match guardará
    # cuantos idiomas detectaron que el usuario hablaba en el idioma que estaba
    # hablando.
    for index, row in df.iterrows():
        # fr pregunta, cuantos idiomas detectaron que se habló en francés cuando el usuario bucaba traducir lo que estaba hablando?
        if row['idioma']=='fr': df.at[index, 'match'] = fr 
        if row['idioma']=='pt': df.at[index, 'match'] = pt
        if row['idioma']=='es': df.at[index, 'match'] = es
        if row['idioma']=='en': df.at[index, 'match'] = en
    
    # La columna nota guardará una nota final de acuerdo a
    # la cantidad de palabras que se reconocieron en el idioma
    # más cuantos stopwords se detectaron en ese idioma por el
    # porcentaje de acierto del idioma y por cuantas veces se
    # encontró el acierto del idioma en los demás idiomas
    df['nota'] = (df['palabras']+df['stopwords'])*df['valor']*df['match'] # valor es el % de que se acertó el idioma que hablaba el usuario
    resultado = df[df['nota'] == df['nota'].max()] # ordenar por la mayor nota y generar un dataframe con una fila que tendrá la mayor nota
    
    # si no está vacío el dataframe anterior
    # mostrar el idioma detectado por el traductor
    # que se encuentra en la columna idioma_final
    # en color azul y tamaño 40
    try:
        if len(resultado) > 0:
            idioma_detectado = resultado['idioma_final'].values[0]
            texto_original   = resultado['traductor_voz'].values[0]
            
            return idioma_detectado, texto_original
        else:
            return 'Nada', 'Nada'  # si el resultado está vacío
    except Exception as e:
        print(f"Error en reconocer: {e}")
        return 'Nada', 'Nada'  # en caso de error


# Función para escuchar el microfono
def escuchar_microfono(voz):
    try:
        print(f"Procesando archivo de audio: {voz}")
        start_time = time.time()
        with sr.AudioFile(voz) as source:
            print("Archivo de audio abierto correctamente")
            audio = grabador.record(source)
            print("Audio grabado correctamente")

        try:
            idioma_detectado, mensaje = reconocer(audio)
            print(f"Idioma detectado: {idioma_detectado}")
            print(f"La persona dijo: {mensaje}")

            if idioma_detectado != 'Nada' and mensaje != 'Nada' and mensaje is not None:
                texto_a_traducir = traductor_google.translate(mensaje, dest='es')
                texto = texto_a_traducir.text 

                print(f'Traducción: {texto}')
                print("\n")

                convertidor_voz = gTTS(text=texto, lang='es')
                convertidor_voz.save("respuesta.mp3")

                print(f"Tiempo de procesamiento: {time.time() - start_time} segundos")
                return idioma_detectado, mensaje, texto
            else:
                print("No se detectó ningún mensaje")
                return None, None, None
        except Exception as e:
            print(f"Error en el procesamiento del audio: {e}")
            return None, None, None
    except Exception as e:
        print(f"Error en escuchar_microfono: {e}")
        print(f"Tipo de archivo: {type(voz)}")
        print(f"Contenido del archivo: {voz[:100] if isinstance(voz, bytes) else 'No es bytes'}")
        return None, None, None