from sqlalchemy import TIMESTAMP, Integer, String, Column, ForeignKey, text
from sqlalchemy.orm import relationship
from config.db import Base

class Rol(Base):
    __tablename__ = "rol"
    id = Column(Integer, primary_key=True)
    rol = Column(String(255), nullable=False)
    users = relationship("Users", back_populates="rol")


class Diagnostico(Base):
    __tablename__ = "diagnosticos"
    id = Column(Integer, primary_key=True)
    # DB column is named 'diagnosticos' per migration; map the attribute 'diagnostico' to that column
    diagnostico = Column('diagnosticos', String(255), nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=text("now()"))
    historiales = relationship("Historial", back_populates="diagnostico", cascade="all, delete")

    
class Users(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String(255), nullable=False, unique=True)
    email = Column(String(255), nullable=False, unique=True)
    password = Column(String(255), nullable=False)
    phone_number = Column(String(12))
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=text("now()"))
    rol_id = Column(Integer, ForeignKey("rol.id", ondelete="CASCADE"), nullable=False)
    rol = relationship("Rol", back_populates="users")

    citas_paciente = relationship("Citas", foreign_keys="Citas.patient_id", back_populates="paciente")
    citas_medico = relationship("Citas", foreign_keys="Citas.medico_id", back_populates="medico")
    historial_medico = relationship("Historial", back_populates="paciente")


class Citas(Base):
    __tablename__ = "citas"
    id = Column(Integer, primary_key=True)
    descripcion = Column(String(255), nullable=False, unique=True)
    patient_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    medico_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    fecha_cita = Column(TIMESTAMP(timezone=True), nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=text("now()"))

    # Relaciones
    paciente = relationship("Users", foreign_keys=[patient_id], back_populates="citas_paciente")
    medico = relationship("Users", foreign_keys=[medico_id], back_populates="citas_medico")
    historial_medico = relationship("Historial", back_populates="cita", uselist=True, cascade="all, delete-orphan")


class Historial(Base):
    __tablename__ = "historial"
    id = Column(Integer, primary_key=True)
    texto_original = Column(String(255), nullable=False)
    resumen = Column(String(255), nullable=False)
    traduccion = Column(String(255), nullable=False)
    entidades = Column(String(255), nullable=False)
    palabras_claves = Column(String(255), nullable=False)
    sentimiento = Column(String(255), nullable=False)
    diagnosticos_id = Column(Integer, ForeignKey("diagnosticos.id", ondelete="CASCADE"), nullable=False)
    cita_id = Column(Integer, ForeignKey("citas.id", ondelete="SET NULL"), nullable=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=text("now()"))

    # Relaciones
    cita = relationship("Citas", foreign_keys=[cita_id], back_populates="historial_medico")
    paciente = relationship("Users", foreign_keys=[user_id], back_populates="historial_medico")
    imagenes = relationship("Imagenes", back_populates="historial_medico")
    diagnostico = relationship("Diagnostico", back_populates="historiales")



class Imagenes(Base):
    __tablename__ = "imagenes"
    id = Column(Integer, primary_key=True)
    historial_medico_id = Column(Integer, ForeignKey("historial.id", ondelete="CASCADE"), nullable=False)
    url = Column(String(255), nullable=False)
    descripcion = Column(String(255), nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=text("now()"))

    historial_medico = relationship("Historial", back_populates="imagenes")
