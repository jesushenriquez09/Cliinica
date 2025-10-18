from pydantic import BaseModel
from typing import Optional


class vhBase(BaseModel):
    texto_original: str
    resumen: str
    traduccion: str
    entidades: str
    palabras_claves: str
    sentimiento: str
    diagnosticos_id: Optional[int]
    cita_id: Optional[int]
    user_id: Optional[int]


class vhcreate(vhBase):
    pass


class nlp_create(BaseModel):
    texto_original: str


class vh(vhBase):
    id: int
    user_id: Optional[int]

    class Config:
        from_attributes = True
