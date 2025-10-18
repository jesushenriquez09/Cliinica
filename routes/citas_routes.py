from datetime import datetime
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from config.db import get_db
from models.db_p import Citas, Users, Historial
from modelo.oauth import get_current_user
from pydantic import BaseModel
from typing import Optional

citas_router = APIRouter(
    prefix="/citas",
    tags=["Citas"],
    responses={404: {"description": "Not found"}},
)

# Esquemas Pydantic
class CitaBase(BaseModel):
    descripcion: str
    patient_id: int
    medico_id: int
    fecha_cita: datetime

class CitaCreate(CitaBase):
    pass

class CitaResponse(CitaBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True

@citas_router.post("/", response_model=CitaResponse)
def create_cita(cita: CitaCreate, db: Session = Depends(get_db), current_user: Users = Depends(get_current_user)):
    """
    Crear una nueva cita médica.
    Se requiere autenticación y los usuarios deben tener los roles apropiados.
    """
    # Verificar que el paciente y médico existan y tengan los roles correctos
    paciente = db.query(Users).filter(Users.id == cita.patient_id).first()
    medico = db.query(Users).filter(Users.id == cita.medico_id).first()
    
    if not paciente or not medico:
        raise HTTPException(
            status_code=404,
            detail="Paciente o médico no encontrado"
        )

    # Verificar que el médico tenga rol de médico (rol_id = 1)
    if medico.rol_id != 1:
        raise HTTPException(
            status_code=400,
            detail=f"El usuario {medico.username} (ID: {medico.id}) no es un médico"
        )

    # Verificar que el paciente tenga rol de paciente (rol_id = 2)
    if paciente.rol_id != 2:
        raise HTTPException(
            status_code=400,
            detail=f"El usuario {paciente.username} (ID: {paciente.id}) no es un paciente"
        )

    # Restringir creación de citas: sólo pacientes (rol_id == 2) pueden crear citas para sí mismos.
    if current_user.rol_id == 2:
        if current_user.id != cita.patient_id:
            raise HTTPException(
                status_code=403,
                detail="Como paciente, solo puedes crear citas para ti mismo"
            )
    else:
        # Médicos y otros roles no pueden crear citas
        raise HTTPException(status_code=403, detail="Solo pacientes pueden crear citas")
    
    # Crear un diccionario con los datos de la cita
    cita_data = cita.model_dump()
    
    # Crear la cita
    db_cita = Citas(**cita_data)
    db.add(db_cita)
    db.commit()
    db.refresh(db_cita)
    return db_cita

@citas_router.get("/{cita_id}", response_model=CitaResponse)
def get_cita(cita_id: int, db: Session = Depends(get_db), current_user: Users = Depends(get_current_user)):
    """
    Obtener una cita específica por su ID.
    """
    cita = db.query(Citas).filter(Citas.id == cita_id).first()
    if not cita:
        raise HTTPException(status_code=404, detail="Cita no encontrada")
    return cita

@citas_router.get("/", response_model=List[CitaResponse])
def get_citas(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: Users = Depends(get_current_user)
):
    """
    Obtener lista de citas.
    Si el usuario es médico, obtiene sus citas asignadas.
    Si el usuario es paciente, obtiene sus citas programadas.
    Si el usuario es admin, obtiene todas las citas.
    """
    # En este sistema: rol_id == 1 -> Médico, rol_id == 2 -> Paciente
    if current_user.rol_id == 1:  # Médico
        citas = db.query(Citas).filter(Citas.medico_id == current_user.id).offset(skip).limit(limit).all()
    elif current_user.rol_id == 2:  # Paciente
        citas = db.query(Citas).filter(Citas.patient_id == current_user.id).offset(skip).limit(limit).all()
    else:
        raise HTTPException(status_code=403, detail="Rol de usuario no permitido para ver citas")
    return citas

@citas_router.put("/{cita_id}", response_model=CitaResponse)
def update_cita(
    cita_id: int,
    cita_update: CitaCreate,
    db: Session = Depends(get_db),
    current_user: Users = Depends(get_current_user)
):
    """
    Actualizar una cita existente.
    """
    db_cita = db.query(Citas).filter(Citas.id == cita_id).first()
    if not db_cita:
        raise HTTPException(status_code=404, detail="Cita no encontrada")
    
    # Verificar permisos: solo el médico asignado puede modificar la cita
    # En este sistema rol_id == 1 -> Médico
    if current_user.rol_id != 1 or current_user.id != db_cita.medico_id:
        # si no es médico asignado, denegar
        raise HTTPException(
            status_code=403,
            detail="No tiene permiso para modificar esta cita"
        )
    
    # Actualizar los campos
    for var, value in vars(cita_update).items():
        setattr(db_cita, var, value) if value else None
    
    db.commit()
    db.refresh(db_cita)
    return db_cita

@citas_router.delete("/{cita_id}")
def delete_cita(
    cita_id: int,
    db: Session = Depends(get_db),
    current_user: Users = Depends(get_current_user)
):
    """
    Eliminar una cita.
    Solo admins y médicos pueden eliminar citas.
    """
    db_cita = db.query(Citas).filter(Citas.id == cita_id).first()
    if not db_cita:
        raise HTTPException(status_code=404, detail="Cita no encontrada")
    
    # Verificar permisos: solo el médico asignado puede eliminar la cita
    if current_user.rol_id != 1 or current_user.id != db_cita.medico_id:
        raise HTTPException(
            status_code=403,
            detail="No tiene permiso para eliminar esta cita"
        )
    
    db.delete(db_cita)
    db.commit()
    return {"message": "Cita eliminada correctamente"}

@citas_router.get("/{cita_id}/historial", response_model=List[dict])
def get_historial_cita(
    cita_id: int,
    db: Session = Depends(get_db),
    current_user: Users = Depends(get_current_user)
):
    """
    Obtener el historial médico asociado a una cita específica.
    """
    # Verificar que la cita existe
    cita = db.query(Citas).filter(Citas.id == cita_id).first()
    if not cita:
        raise HTTPException(status_code=404, detail="Cita no encontrada")
    
    # Verificar permisos: médico o paciente asociado pueden ver el historial
    if not ((current_user.rol_id == 1 and current_user.id == cita.medico_id) or (current_user.rol_id == 2 and current_user.id == cita.patient_id)):
        raise HTTPException(
            status_code=403,
            detail="No tiene permiso para ver este historial"
        )
    
    # Obtener el historial
    historial = db.query(Historial).filter(Historial.cita_id == cita_id).all()
    return historial