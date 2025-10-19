from textblob import TextBlob 
from nltk.corpus import stopwords
from nltk.tokenize  import word_tokenize
import nltk 
nltk.download('stopwords')

texto = """ 
Tengo dolor al orinar y molestia en la vejiga desde hace dos días.
""" 
def resumen_nlp(texto: str):
    blob = TextBlob(texto) 

    frases = blob.sentences 

    stop_words = set(stopwords.words('spanish'))

    def procesar_frase(frase):
        palabras = word_tokenize(str(frase).lower())
        palabras_filtradas = [palabra for palabra in palabras if palabra.isalnum() and palabra not in stop_words]
        return ' '.join(palabras_filtradas) 

    resumen = "".join([procesar_frase(frase) for frase in frases[:2]])
    print(resumen) 
    return resumen
        
resumen_nlp(texto)