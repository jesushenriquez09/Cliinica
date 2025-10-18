import spacy
from datetime import timedelta
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Response, Body
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from textblob import TextBlob
import torch
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.security.oauth2 import OAuth2PasswordRequestForm
from config.db import  SessionLocal, get_db
from passlib.context import CryptContext
from modelo import oauth
from modelo.oauth import get_current_user
from models.db_p import  Users , Historial, Diagnostico, Citas
from sqlalchemy.orm import Session
from models import db_p
from modelo import m_pro
from modelo.m_user import Login, Token, users
from modelo.token import create_access_token 
from nltk.probability import FreqDist 
from nltk.tokenize  import word_tokenize
from nltk.corpus import stopwords
import json

# Inicialización lazy de modelos
nlp = None
sentiment_analyzer = None

def get_nlp():
    global nlp
    if nlp is None:
        try:
            nlp = spacy.load("es_core_news_md")
        except Exception:
            # Do not attempt to download at runtime (can block startup). Leave nlp as None
            nlp = None
    return nlp

def get_sentiment_analyzer():
    global sentiment_analyzer
    if sentiment_analyzer is None:
        try:
            sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="finiteautomata/beto-sentiment-analysis",
                tokenizer="finiteautomata/beto-sentiment-analysis"
            )
        except Exception:
            # If transformers or the model are not available, don't block startup; return None
            sentiment_analyzer = None
    return sentiment_analyzer

# Configurar el router con tags para mejor documentación
nlp_route = APIRouter(
    prefix="/nlp",
    tags=["NLP"],
    responses={404: {"description": "Not found"}},
)


@nlp_route.post("/", response_model=m_pro.vhBase, 
                 summary="Procesar texto con NLP",
                 description="Analiza un texto en español utilizando técnicas de NLP para extraer información relevante")
async def process_text(
    texto: m_pro.nlp_create = Body(..., example={"texto_original": "Este es un ejemplo de texto para analizar", "cita_id": None, "diagnostico": None}),
    cita_id: Optional[int] = Body(None),
    diagnostico: Optional[str | int] = Body(None),
    db: Session = Depends(get_db),
    current_user: Users = Depends(get_current_user)
):
    """
    Procesa un texto utilizando técnicas avanzadas de NLP:
    
    - Genera un resumen
    - Detecta entidades nombradas
    - Extrae palabras clave
    - Analiza el sentimiento
    - Traduce si es necesario
    
    Returns:
        m_pro.vhBase: Resultado del análisis
    """
    try:
        texto_original = texto.texto_original
        
        # 1. Obtener modelos de forma lazy
        spacy_nlp = get_nlp()
        if spacy_nlp is None:
            raise HTTPException(status_code=503, detail=("Modelo spaCy 'es_core_news_md' no disponible. "
                                                       "Instálalo localmente con: python -m spacy download es_core_news_md"))
        doc = spacy_nlp(texto_original)
        
        # 2. Generar resumen (usando la primera oración)
        resumen = str(list(doc.sents)[0]) if len(list(doc.sents)) > 0 else texto_original[:100]
        
        # 3. Traducción (si el texto no está en español, traducir a español)
        try:
            blob = TextBlob(texto_original)
            detected_lang = blob.detect_language()
            traduccion = str(blob.translate(to='es')) if detected_lang != 'es' else texto_original
        except:
            traduccion = texto_original
            
        # 4. Entidades nombradas con spaCy
        entidades = {}
        for ent in doc.ents:
            if ent.label_ not in entidades:
                entidades[ent.label_] = []
            entidades[ent.label_].append(ent.text)
        entidades = str(entidades)
        
        # 5. Palabras clave usando spaCy
        palabras_claves = {}
        for token in doc:
            if not token.is_stop and not token.is_punct and token.pos_ in ['NOUN', 'ADJ', 'VERB']:
                if token.lemma_ not in palabras_claves:
                    palabras_claves[token.lemma_] = 1
                else:
                    palabras_claves[token.lemma_] += 1
        
        # Ordenar por frecuencia y tomar los top 5
        palabras_claves = str(dict(sorted(palabras_claves.items(), key=lambda x: x[1], reverse=True)[:5]))
        
        # 6. Análisis de sentimiento usando transformers (carga lazy)
        try:
            analyzer = get_sentiment_analyzer()
            if analyzer is None:
                raise Exception("Modelo de sentiment-analysis no disponible. Instala transformers y el modelo BETO.")
            sentiment_result = analyzer(texto_original)[0]
            label = sentiment_result['label']
            score = sentiment_result['score']
            
            if label == 'POS':
                sentimiento = "positivo"
            elif label == 'NEG':
                sentimiento = "negativo"
            else:
                sentimiento = "neutral"
        except Exception as e:
            print(f"Error en análisis de sentimiento: {str(e)}")
            sentimiento = "neutral"
            
        # Resolver diagnostico (puede ser id o nombre)
        diagnostico_id = None
        if diagnostico is not None:
            # si es int, se asume id
            if isinstance(diagnostico, int):
                exists = db.query(Diagnostico).filter(Diagnostico.id == diagnostico).first()
                if not exists:
                    raise HTTPException(status_code=400, detail="Diagnostico id no encontrado")
                diagnostico_id = diagnostico
            else:
                # buscar por nombre exacto
                exists = db.query(Diagnostico).filter(Diagnostico.diagnostico == diagnostico).first()
                if not exists:
                    raise HTTPException(status_code=400, detail="Diagnostico nombre no encontrado")
                diagnostico_id = exists.id

        # Validar cita si se proporciona
        if cita_id is not None:
            cita_obj = db.query(Citas).filter(Citas.id == cita_id).first()
            if not cita_obj:
                raise HTTPException(status_code=400, detail="Cita no encontrada")

        # 7. Crear el objeto de respuesta
        resultado = m_pro.vhBase(
            texto_original=texto_original,
            resumen=resumen,
            traduccion=traduccion,
            entidades=entidades,
            palabras_claves=palabras_claves,
            sentimiento=sentimiento,
            diagnosticos_id=diagnostico_id,
            cita_id=cita_id,
            user_id=current_user.id
        )

        # 8. Guardar en la base de datos
        db_item = Historial(
            user_id=current_user.id,
            texto_original=resultado.texto_original,
            resumen=resultado.resumen,
            traduccion=resultado.traduccion,
            entidades=resultado.entidades,
            palabras_claves=resultado.palabras_claves,
            sentimiento=resultado.sentimiento,
            diagnosticos_id=diagnostico_id,
            cita_id=cita_id
        )

        db.add(db_item)
        db.commit()
        db.refresh(db_item)

        # mapear a modelo de respuesta y devolver
        resultado.user_id = db_item.user_id
        return resultado
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando el texto: {str(e)}")
    
    # 3. Keywords extraction
    stop_words = set(stopwords.words('spanish'))
    words = word_tokenize(text_original.lower())
    keywords = [word for word in words if word.isalnum() and word not in stop_words]
    freq_dist = FreqDist(keywords)
    palabras_claves = str(dict(freq_dist.most_common(5)))
    
    # 4. Sentiment Analysis
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        sentimiento = "positivo"
    elif sentiment < 0:
        sentimiento = "negativo"
    else:
        sentimiento = "neutral"
    
    # 5. Create response dictionary
    response = {
        "text_original": text_original,
        "resumen": str(blob.sentences[0]) if blob.sentences else text_original[:100],
        "traduccion": traduccion,
        "entidades": entidades,
        "palabras_claves": palabras_claves,
        "sentimiento": sentimiento,
        "diagnosticos_id": None  # Optional field that can be updated later
    }
    
    