from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from config.db import get_db
from models.db_p import Diagnostico
from typing import List

router = APIRouter(
    prefix="/diagnosticos",
    tags=["Diagnosticos"],
)

class DiagnosticoSchema(BaseModel):
    id: int
    diagnostico: str

    class Config:
        from_attributes = True
    

diagnosticos_excretor = [
    "Infección urinaria (cistitis)",
    "Uretritis (inflamación de la uretra)",
    "Pielonefritis aguda (infección renal)",
    "Cálculos renales (litiasis renal)",
    "Litiasis vesical (piedras en la vejiga)",
    "Insuficiencia renal aguda",
    "Insuficiencia renal crónica",
    "Glomerulonefritis (inflamación de los glomérulos)",
    "Nefropatía diabética",
    "Hidronefrosis (obstrucción del riñón)",
    "Cistitis intersticial",
    "Prostatitis (inflamación de la próstata)",
    "Retención urinaria aguda",
    "Enuresis (incontinencia urinaria)",
    "Nefrolitiasis recurrente",
    "Hematuria microscópica",
    "Proteinuria (proteínas en la orina)",
    "Síndrome nefrótico",
    "Hipertensión secundaria a enfermedad renal",
    "Infección del tracto urinario recurrente",
    "Obstrucción ureteral por cálculo",
    "Nefritis tubulointersticial",
    "Cistitis hemorrágica",
    "Uropatía obstructiva congénita",
    "Incontinencia urinaria de esfuerzo",
    "Incontinencia urinaria por urgencia",
    "Cálculo de urato (ácido úrico)",
    "Cálculo de oxalato de calcio",
    "Infección renal por bacterias resistentes",
    "Pielonefritis crónica",
    "Nefropatía por medicamentos (tóxica)",
    "Quistes renales simples o múltiples"
]

@router.post("/seed")
def seed_diagnosticos(db: Session = Depends(get_db)):
    inserted = []
    for nombre in diagnosticos_excretor:
        exists = db.query(Diagnostico).filter(Diagnostico.diagnostico == nombre).first()
        if not exists:
            d = Diagnostico(diagnostico=nombre)
            db.add(d)
            inserted.append(nombre)
    db.commit()
    return {"inserted": inserted, "count": len(inserted)}
