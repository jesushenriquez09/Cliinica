import nltk
import spacy
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
from transformers import pipeline
import json

# Descargar recursos necesarios
nltk.download('punkt')
nltk.download('stopwords')

class ProcessadorTexto:
    def __init__(self):
        # Inicializar modelos
        try:
            self.nlp = spacy.load("es_core_news_md")
        except:
            print("Necesitas instalar el modelo de spaCy: python -m spacy download es_core_news_md")
            self.nlp = None
            
        try:
            self.translator = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")
        except:
            print("Error al cargar el modelo de traducción")
            self.translator = None
            
        # Cargar stopwords
        try:
            self.stop_words = set(stopwords.words('spanish'))
        except:
            print("Error al cargar stopwords en español")
            self.stop_words = set()

    def generar_resumen(self, texto):
        """Genera un resumen usando TextBlob"""
        try:
            blob = TextBlob(texto)
            # Tomar la primera oración como resumen
            return str(blob.sentences[0]) if blob.sentences else texto[:100]
        except Exception as e:
            print(f"Error al generar resumen: {str(e)}")
            return texto[:100]

    def extraer_palabras_clave(self, texto):
        """Extrae palabras clave usando NLTK"""
        try:
            # Tokenización y filtrado
            tokens = word_tokenize(texto.lower())
            tokens_filtrados = [word for word in tokens if word.isalnum() and word not in self.stop_words]
            
            # Análisis de frecuencia
            frecuencia = FreqDist(tokens_filtrados)
            palabras_clave = {}
            for palabra, freq in frecuencia.most_common(5):
                palabras_clave[palabra] = freq
                
            return palabras_clave
        except Exception as e:
            print(f"Error al extraer palabras clave: {str(e)}")
            return {}

    def extraer_entidades(self, texto):
        """Extrae entidades nombradas usando spaCy"""
        if not self.nlp:
            return []
            
        try:
            doc = self.nlp(texto)
            entidades = []
            for ent in doc.ents:
                entidades.append({
                    "texto": ent.text,
                    "tipo": ent.label_
                })
            return entidades
        except Exception as e:
            print(f"Error al extraer entidades: {str(e)}")
            return []

    def traducir_texto(self, texto):
        """Traduce el texto de español a inglés"""
        if not self.translator:
            return texto
            
        try:
            traduccion = self.translator(texto, max_length=512)[0]['translation_text']
            return traduccion
        except Exception as e:
            print(f"Error al traducir: {str(e)}")
            return texto

    def procesar_texto(self, texto):
        """Procesa el texto aplicando todas las funcionalidades"""
        resultado = {
            "texto_original": texto,
            "resumen": self.generar_resumen(texto),
            "palabras_clave": self.extraer_palabras_clave(texto),
            "entidades": self.extraer_entidades(texto),
            "traduccion": self.traducir_texto(texto)
        }
        return resultado

# Ejemplo de uso
if __name__ == "__main__":
    procesador = ProcessadorTexto()
    
    texto_ejemplo = """
    Tengo dolor al orinar y molestia en la vejiga desde hace dos días.
    La fiebre comenzó ayer por la noche y he tomado paracetamol para controlarla.
    """
    
    resultado = procesador.procesar_texto(texto_ejemplo)
    print("\nResultados del procesamiento:")
    print("\nTexto original:")
    print(resultado["texto_original"])
    print("\nResumen:")
    print(resultado["resumen"])
    print("\nPalabras clave:")
    for palabra, frecuencia in resultado["palabras_clave"].items():
        print(f"{palabra}: {frecuencia}")
    print("\nEntidades encontradas:")
    for entidad in resultado["entidades"]:
        print(f"{entidad['texto']} - {entidad['tipo']}")
    print("\nTraducción:")
    print(resultado["traduccion"])