# Generador de Quizzes
# por Franco Benassi
import json
import random
import torch
import torch.nn as nn
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
    def __init__(self, embedding_dim, n_filters, filter_sizes, output_dim, dropout):
        super().__init__()
        self.bert = bert_model
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=n_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        with torch.no_grad():
            embedded = self.bert(text)[0]
        embedded = embedded.permute(0, 2, 1)
        conved = [nn.functional.relu(conv(embedded)) for conv in self.convs]
        pooled = [nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

# Función para entrenar el modelo
def train_model(model, iterator, optimizer, criterion, device):
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch['input_ids'].to(device))
        loss = criterion(predictions, batch['labels'].to(device))
        loss.backward()
        optimizer.step()

# Función para evaluar el modelo
def evaluate_model(model, iterator, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch['input_ids'].to(device))
            loss = criterion(predictions, batch['labels'].to(device))
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
    tokens = bert_tokenizer(oracion, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = modelo_cnn(tokens['input_ids'])
    tipo_pregunta = torch.argmax(outputs).item()
    
    if tipo_pregunta == 0:  # Pregunta de alternativas
        return generar_pregunta_alternativas(oracion)
    else:  # Pregunta de verdadero/falso
        return generar_pregunta_verdadero_falso(oracion)

def generar_pregunta_alternativas(oracion):
    doc = nlp(oracion)
    
    sujeto = next((token for token in doc if token.dep_ == "nsubj"), None)
    verbo = next((token for token in doc if token.pos_ == "VERB"), None)
    objeto = next((token for token in doc if token.dep_ in ["dobj", "iobj"]), None)
    
    if sujeto and verbo:
        if objeto:
            pregunta = f"¿Qué {verbo.lemma_} {sujeto.text}?"
        else:
            pregunta = f"¿Qué hace {sujeto.text}?"
    else:
        entidades = [ent.text for ent in doc.ents]
        if entidades:
            entidad = random.choice(entidades)
            pregunta = f"¿Qué se menciona sobre {entidad}?"
        else:
            pregunta = "¿Cuál es el tema principal de esta oración?"
    
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
        pregunta = modificar_oracion_falsa(oracion)
    
    return {
        'tipo': 'verdadero_falso',
        'pregunta': pregunta,
        'respuesta_correcta': es_verdadero
    }

def modificar_oracion_falsa(oracion):
    doc = nlp(oracion)
    palabras_clave = [token for token in doc if token.pos_ in ["NOUN", "VERB", "ADJ", "NUM"]]
    
    if palabras_clave:
        palabra_a_modificar = random.choice(palabras_clave)
        
        if palabra_a_modificar.pos_ == "NUM":
            try:
                nuevo_valor = str(int(palabra_a_modificar.text) + random.choice([-1, 1]) * random.randint(1, 5))
                oracion_modificada = oracion.replace(palabra_a_modificar.text, nuevo_valor)
            except ValueError:
                oracion_modificada = oracion.replace(palabra_a_modificar.text, "un número diferente")
        elif palabra_a_modificar.pos_ == "NOUN":
            antonyms = ["ausencia de " + palabra_a_modificar.text, "escasez de " + palabra_a_modificar.text]
            oracion_modificada = oracion.replace(palabra_a_modificar.text, random.choice(antonyms))
        elif palabra_a_modificar.pos_ in ["VERB", "ADJ"]:
            antonyms = ["raramente " + palabra_a_modificar.text, "ocasionalmente " + palabra_a_modificar.text]
            oracion_modificada = oracion.replace(palabra_a_modificar.text, random.choice(antonyms))
        else:
            oracion_modificada = oracion
    else:
        oracion_modificada = "La información presentada en el texto original no es precisa."
    
    return oracion_modificada

def generar_distractores(oracion):
    doc = nlp(oracion)
    distractores = []
    
    entidades = list(doc.ents)
    palabras_clave = [token for token in doc if token.pos_ in ["NOUN", "VERB", "ADJ", "NUM"]]
    
    for _ in range(3):
        if entidades:
            entidad = random.choice(entidades)
            distractor = oracion.replace(entidad.text, "otra entidad")
        elif palabras_clave:
            palabra = random.choice(palabras_clave)
            if palabra.pos_ == "NOUN":
                distractor = oracion.replace(palabra.text, "otro concepto")
            elif palabra.pos_ == "VERB":
                distractor = oracion.replace(palabra.text, "otra acción")
            elif palabra.pos_ == "ADJ":
                distractor = oracion.replace(palabra.text, "otra característica")
            else:
                distractor = oracion.replace(palabra.text, "otro valor")
        else:
            distractor = "Esta información no se menciona en el texto original."
        
        if distractor not in distractores and distractor != oracion:
            distractores.append(distractor)
    
    while len(distractores) < 3:
        distractor = "Esta información no se relaciona directamente con el texto original."
        if distractor not in distractores:
            distractores.append(distractor)
    
    return distractores[:3]

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
EMBEDDING_DIM = 768  # Dimensión de embeddings de BERT
N_FILTERS = 100
FILTER_SIZES = [3, 4, 5]
OUTPUT_DIM = 2
DROPOUT = 0.5

model = TextCNN(EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT).to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# Cargar y preparar datos
datos_json = cargar_datos_json('tutorApp/static/json/quiz.json')
textos = [item['texto'] for item in datos_json['quiz']]
etiquetas = [random.randint(0, 1) for _ in range(len(textos))]  # Etiquetas aleatorias para demostración

# Tokenizar y codificar los textos
encoded_texts = bert_tokenizer(textos, padding=True, truncation=True, max_length=512, return_tensors="pt")
label_tensors = torch.tensor(etiquetas)

# Crear dataset
dataset = [{'input_ids': encoded_texts['input_ids'][i], 'labels': label_tensors[i]} for i in range(len(textos))]

# Dividir datos en entrenamiento y prueba
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

# Crear DataLoader
train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=16)

# Entrenar el modelo
N_EPOCHS = 5
for epoch in range(N_EPOCHS):
    train_model(model, train_loader, optimizer, criterion, device)
    valid_loss = evaluate_model(model, test_loader, criterion, device)
    print(f'Epoch: {epoch+1:02}, Valid Loss: {valid_loss:.3f}')