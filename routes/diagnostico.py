import nltk 
from nltk.probability import FreqDist 
from nltk.tokenize  import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt_tab')
nltk.download('stopwords')

texto = """ 
Dolor al orinar y sensación de ardor en la vejiga y he tenido fiebre baja y malestar general.
""" 

tokens = word_tokenize(texto.lower())

try:
    stop_words = set(stopwords.words('spanish'))
except:
    print("No se pudieron cargar las stopwords en español. Asegúrese de tener el paquete de stopwords descargado.")
    strop_words = set() 
    
tokens_filtrados = [word for word in tokens if word.isalnum() and word not in stop_words]   
frecuencia = FreqDist(tokens_filtrados)
print("palabras claves")
for palabra,frecuencia in frecuencia.most_common(5):
    print(f"{palabra}: {frecuencia}")

