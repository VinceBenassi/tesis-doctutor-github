# Generador de Quizzes
# por Franco Benassi
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
import spacy
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')
nlp = spacy.load("es_core_news_sm")

# Cargar modelo BERT
bert_tokenizer = BertTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
bert_model = BertModel.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")

# Definir la red neuronal convolucional
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text).unsqueeze(1)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

# Función para entrenar el modelo
def train_model(model, iterator, optimizer, criterion, device):
    model.train()
    for text, label in iterator:
        text = text.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        predictions = model(text).squeeze(1)
        loss = criterion(predictions, label)
        loss.backward()
        optimizer.step()

# Función para evaluar el modelo
def evaluate_model(model, iterator, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for text, label in iterator:
            text = text.to(device)
            label = label.to(device)
            predictions = model(text).squeeze(1)
            loss = criterion(predictions, label)
            total_loss += loss.item()
    return total_loss / len(iterator)

# Función para cargar y preparar los datos
def cargar_datos_json(ruta_archivo):
    with open(ruta_archivo, 'r', encoding='utf-8') as archivo:
        return json.load(archivo)

# Función para preprocesar el texto
def preprocesar_texto(texto):
    doc = nlp(texto)
    return [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]

# Función para generar preguntas
def generar_pregunta(oracion, modelo_cnn, device):
    tokens = bert_tokenizer.encode(oracion, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = modelo_cnn(tokens)
    tipo_pregunta = torch.argmax(outputs).item()
    
    if tipo_pregunta == 0:  # Pregunta de alternativas
        return generar_pregunta_alternativas(oracion)
    else:  # Pregunta de verdadero/falso
        return generar_pregunta_verdadero_falso(oracion)

def generar_pregunta_alternativas(oracion):
    doc = nlp(oracion)
    sujeto = next((token for token in doc if token.dep_ == "nsubj"), None)
    verbo = next((token for token in doc if token.pos_ == "VERB"), None)
    
    if sujeto and verbo:
        pregunta = f"¿Qué {verbo.lemma_} {sujeto.text}?"
    else:
        pregunta = f"¿Qué menciona la oración sobre {doc[0].text}?"
    
    respuesta_correcta = oracion
    distractores = generar_distractores(oracion)
    
    return {
        'tipo': 'alternativas',
        'pregunta': pregunta,
        'opciones': [respuesta_correcta] + distractores,
        'respuesta_correcta': respuesta_correcta
    }

def generar_pregunta_verdadero_falso(oracion):
    es_verdadero = random.choice([True, False])
    
    if es_verdadero:
        pregunta = oracion
    else:
        doc = nlp(oracion)
        for token in doc:
            if token.pos_ in ["NOUN", "VERB", "ADJ"]:
                sinonimos = [t.text for t in token.children if t.dep_ == "conj"]
                if sinonimos:
                    oracion = oracion.replace(token.text, random.choice(sinonimos))
                    break
        pregunta = oracion
    
    return {
        'tipo': 'verdadero_falso',
        'pregunta': pregunta,
        'respuesta_correcta': es_verdadero
    }

def generar_distractores(oracion):
    doc = nlp(oracion)
    distractores = []
    
    for token in doc:
        if token.pos_ in ["NOUN", "VERB", "ADJ"]:
            sinonimos = [t.text for t in token.children if t.dep_ == "conj"]
            if sinonimos:
                distractor = oracion.replace(token.text, random.choice(sinonimos))
                if distractor not in distractores:
                    distractores.append(distractor)
    
    return distractores[:3]  # Devolver hasta 3 distractores

# Función principal para generar cuestionarios
def generar_cuestionarios(datos_json, modelo_cnn, device):
    cuestionarios = {}
    
    for item in datos_json['quiz']:
        categoria = item['materia']
        texto = item['texto']
        fuente = item['fuente']
        
        if categoria not in cuestionarios:
            cuestionarios[categoria] = []
        
        oraciones = sent_tokenize(texto)
        preguntas = []
        
        for oracion in oraciones:
            pregunta = generar_pregunta(oracion, modelo_cnn, device)
            preguntas.append(pregunta)
        
        cuestionarios[categoria].append({
            'fuente': fuente,
            'preguntas': preguntas
        })
    
    return cuestionarios

# Configuración y entrenamiento del modelo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VOCAB_SIZE = len(bert_tokenizer.vocab)
EMBEDDING_DIM = 300
N_FILTERS = 100
FILTER_SIZES = [3, 4, 5]
OUTPUT_DIM = 2
DROPOUT = 0.5
PAD_IDX = bert_tokenizer.pad_token_id

model = TextCNN(VOCAB_SIZE, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX).to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# Cargar y preparar datos
datos_json = cargar_datos_json('tutorApp/static/json/quiz.json')
textos = [item['texto'] for item in datos_json['quiz']]
etiquetas = [random.randint(0, 1) for _ in range(len(textos))]  # Etiquetas aleatorias para demostración

# Tokenizar y codificar los textos
encoded_texts = [bert_tokenizer.encode(texto, truncation=True, max_length=512, padding='max_length') for texto in textos]

# Convertir a tensores
text_tensors = torch.tensor(encoded_texts)
label_tensors = torch.tensor(etiquetas)

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(text_tensors, label_tensors, test_size=0.2, random_state=42)

# Crear DataLoader
train_data = list(zip(X_train, y_train))
test_data = list(zip(X_test, y_test))

# Entrenar el modelo
N_EPOCHS = 5
for epoch in range(N_EPOCHS):
    train_model(model, train_data, optimizer, criterion, device)
    valid_loss = evaluate_model(model, test_data, criterion, device)
    print(f'Epoch: {epoch+1:02}, Valid Loss: {valid_loss:.3f}')

# Generar cuestionarios
cuestionarios = generar_cuestionarios(datos_json, model, device)