import spacy
import nltk
from datetime import timedelta
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Response, Body
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM
from textblob import TextBlob
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from routes.sentimientos_nlp import analisis_sentimientos_nlp
from routes.resumen_nlp import resumen_nlp
from routes.entidad_nlp import entidades_nlp
from routes.traduccion_nlp import traduccion_nlp
from routes.palabras_claves import nlp_palabras_claves
from config.db import SessionLocal, get_db
from modelo.oauth import get_current_user
from models.db_p import Users, Historial, Diagnostico, Citas, Entidad
from sqlalchemy.orm import Session
from modelo import m_pro
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
import torch
import json

# 🚀 Configurar el router
nlp_route = APIRouter(
    prefix="/nlp",
    tags=["NLP"],
    responses={404: {"description": "Not found"}},
)

# 🧠 Cargar modelo biomédico español
tokenizer = AutoTokenizer.from_pretrained("PlanTL-GOB-ES/roberta-base-biomedical-clinical-es")
model = AutoModelForMaskedLM.from_pretrained("PlanTL-GOB-ES/roberta-base-biomedical-clinical-es")

# Pipeline de completado biomédico
pipe = pipeline("fill-mask", model=model, tokenizer=tokenizer)

# Modelo de embeddings semánticos (para similitud entre diagnósticos)
modelo_embeddings = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')


@nlp_route.post("/procesar_texto")
async def process_text(
    texto: m_pro.nlp_create = Body(...),
    cita_id: Optional[int] = Body(None),
    diagnostico: Optional[str | int] = Body(None),
    db: Session = Depends(get_db),
    current_user: Users = Depends(get_current_user),
):
    try:
        texto_original = texto.texto_original
        resumen = resumen_nlp(texto.texto_original)["texto_es"]
        texto_traducido = traduccion_nlp(texto.texto_original)
        entidades = entidades_nlp(texto.texto_original)
        palabras_claves = nlp_palabras_claves(texto.texto_original)["palabras_claves"]
        sentimiento = analisis_sentimientos_nlp(texto.texto_original)

        # 🩺 2️⃣ Generar diagnóstico probable con modelo biomédico
        masked_text = f"El paciente presenta {texto_traducido}. Diagnóstico probable: <mask>."
        resultado = pipe(masked_text)[0]
        diagnostico_generado = resultado["token_str"].strip()

        # 🧩 3️⃣ Cargar diagnósticos desde la base de datos
        diagnosticos_db = db.query(Diagnostico).all()
        nombres_diagnosticos = [d.diagnostico for d in diagnosticos_db]

        # 🧮 4️⃣ Calcular embeddings y similitudes
        embeddings_diag = modelo_embeddings.encode(nombres_diagnosticos, convert_to_tensor=True)
        embedding_generado = modelo_embeddings.encode(diagnostico_generado, convert_to_tensor=True)

        similitudes = util.cos_sim(embedding_generado, embeddings_diag)[0]
        mejor_idx = torch.argmax(similitudes).item()
        confianza = similitudes[mejor_idx].item()

        # 🏥 5️⃣ Seleccionar diagnóstico más parecido
        diag_seleccionado = diagnosticos_db[mejor_idx]
        diagnostico_filtrado = diag_seleccionado.diagnostico
        diagnostico_id = diag_seleccionado.id

        # ⚠️ 6️⃣ Si la similitud es baja, usar “Sin diagnóstico definido”
        if confianza < 0.45:
            sin_diag = (
                db.query(Diagnostico)
                .filter(Diagnostico.diagnostico.ilike("%sin diagnóstico%"))
                .first()
            )
            if sin_diag:
                diagnostico_filtrado = sin_diag.diagnostico
                diagnostico_id = sin_diag.id
            else:
                diagnostico_filtrado = "Sin diagnóstico definido"
                diagnostico_id = None
        
        
        db_item = Historial(
            user_id=current_user.id,
            texto_original=texto.texto_original,
            resumen=resumen,  # string puro
            traduccion=texto_traducido,  # texto limpio
            entidades=json.dumps(entidades, ensure_ascii=False),  # lista de entidades
            palabras_claves=json.dumps(palabras_claves, ensure_ascii=False),  # lista de strings
            sentimiento=json.dumps(sentimiento, ensure_ascii=False),  # estructura pequeña
            diagnosticos_id=diagnostico_id,
            cita_id=cita_id,
       )

        db.add(db_item)
        db.commit()
        db.refresh(db_item)   

        return {
            "mensaje": "Diagnóstico procesado correctamente.",
            "diagnostico_generado": diagnostico_generado,
            "diagnostico_filtrado": diagnostico_filtrado,
            "diagnostico_id": diagnostico_id,
            "confianza": round(confianza, 3),
            "traduccion": texto_traducido,
            "texto_original": texto_original,
            "resumen": resumen,
            "entidades": entidades,
            "palabras_claves": palabras_claves,
            "sentimiento": sentimiento,
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error al procesar texto: {str(e)}")
