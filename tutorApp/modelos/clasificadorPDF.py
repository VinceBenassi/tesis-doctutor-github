# Clasificador de Materias PDF
# por Franco Benassi
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from collections import Counter
import re

# nltk.download('punkt', quiet=True)
# nltk.download('stopwords', quiet=True)

def preprocess_text(text):
    text = re.sub(r'\.pdf$', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚñÑ\s]', ' ', text)
    
    stop_words = set(stopwords.words('spanish') + stopwords.words('english'))
    keep_words = {'en', 'de', 'la', 'el', 'y', 'e', 'i', 'ii', 'iii', 'iv', 'v'}
    stop_words = stop_words - keep_words
    
    tokens = word_tokenize(text.lower())
    return ' '.join([word for word in tokens if word not in stop_words or word in keep_words])

def extract_keywords(text, min_words=1, max_words=5):
    words = word_tokenize(text.lower())
    word_freq = Counter(words)
    keywords = []
    for word, _ in word_freq.most_common():
        if len(keywords) >= max_words:
            break
        if len(word) > 1 or word in {'i', 'ii', 'iii', 'iv', 'v', 'en', 'i', 'y', 'e'}:
            keywords.append(word)
    
    while len(keywords) < min_words and words:
        word = words.pop(0)
        if word not in keywords:
            keywords.append(word)
    
    return keywords

def simplify_materia_name(name):
    words = name.split()
    if len(words) <= 2:
        return name
    elif 'en' in words[1:]:
        en_index = words.index('en')
        return ' '.join(words[:min(en_index + 3, len(words))])
    else:
        return words[0]

def is_common_word(word):
    common_words = {'especialidades', 'estandares', 'medicas', 'medio', 'chile'}
    return word.lower() in common_words

def classify_pdfs(pdf_dir):
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        print("No se encontraron archivos PDF.")
        return {}

    preprocessed_names = [preprocess_text(pdf) for pdf in pdf_files]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X = vectorizer.fit_transform(preprocessed_names)

    clustering = DBSCAN(eps=0.5, min_samples=1)
    cluster_labels = clustering.fit_predict(X.toarray())

    materias = {}
    for pdf, label in zip(pdf_files, cluster_labels):
        preprocessed_pdf = preprocess_text(pdf)
        keywords = extract_keywords(preprocessed_pdf)
        
        if label == -1:
            materia_name = ' '.join(keywords).capitalize()
        else:
            cluster_pdfs = [pdf_files[i] for i, l in enumerate(cluster_labels) if l == label]
            cluster_text = ' '.join([preprocess_text(cluster_pdf) for cluster_pdf in cluster_pdfs])
            cluster_keywords = extract_keywords(cluster_text)
            materia_name = ' '.join(cluster_keywords).capitalize()
        
        materia_name = simplify_materia_name(materia_name)
        materia_name = ' '.join([word for word in materia_name.split() if not is_common_word(word)])
        materia_name = re.sub(r'\s+', ' ', materia_name).strip()
        
        # Buscar si ya existe una materia similar
        existing_materia = None
        for existing in materias:
            if existing.split()[0].lower() == materia_name.split()[0].lower():
                existing_materia = existing
                break
        
        if existing_materia:
            materias[existing_materia].append(pdf)
        else:
            materias[materia_name] = [pdf]

    return materias

def get_materias_data(pdf_dir):
    return classify_pdfs(pdf_dir)