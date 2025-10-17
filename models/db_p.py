from sqlalchemy import TIMESTAMP, Integer, String, Table, Column, text, true, ForeignKey
from sqlalchemy.orm import relationship
from config.db import Base


class Vehiculos(Base):
    __tablename__ = "vehiculos"
    id = Column(Integer, primary_key=True)
    name_product = Column(String(255), nullable=False)
    placa = Column(String(255), nullable=False)
    color = Column(String(300), nullable=False)
    price = Column(Integer, nullable=False)
    stock = Column(Integer, nullable=False)
    status_id = Column(Integer, ForeignKey("status.id", ondelete="CASCADE"), nullable=False)
    category_id = Column(Integer, ForeignKey("categorys.id", ondelete="CASCADE"), nullable=False)
    modelo_id = Column(Integer, ForeignKey("Autos_models.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=text('now()'))
    
    
    
class Model_Auto(Base):
    __tablename__ = "Autos_models"
    id = Column(Integer, primary_key=True)
    modelo = Column(String(255), nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=text('now()'))
    
    

class rol(Base):
    __tablename__ = "rol"
    id = Column(Integer, primary_key=True)
    rol = Column(String(255), nullable=False)
class status(Base):
    __tablename__ = "status"
    id = Column(Integer, primary_key=True)
    status = Column(String(255), nullable=False)
    
class venta(Base):
    __tablename__ = "venta"
    id = Column(Integer, primary_key=True)
    total = Column(Integer, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=text('now()'))

class compra(Base):
    __tablename__ = "compra"
    id = Column(Integer, primary_key=True)
    total = Column(Integer, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=text('now()'))
    
    
class Category(Base):
    __tablename__ = "categorys"
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=text('now()'))

class Detalle_Compra(Base):
    __tablename__ = "Detalle_Compra"
    id = Column(Integer, primary_key=True)
    cantidad = Column(Integer, nullable=False)
    subtotal = Column(Integer, nullable=False)
    vehiculo_id = Column(Integer, ForeignKey("vehiculos.id", ondelete="CASCADE"), nullable=False)
    compra_id = Column(Integer, ForeignKey("compra.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=text('now()'))

class Detalle_Venta(Base):
    __tablename__ = "Detalle_Venta"
    id = Column(Integer, primary_key=True)
    cantidad = Column(Integer, nullable=False)
    subtotal = Column(Integer, nullable=False)
    vehiculo_id = Column(Integer, ForeignKey("vehiculos.id", ondelete="CASCADE"), nullable=False)
    venta_id = Column(Integer, ForeignKey("venta.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=text('now()'))


class Users(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String(255), nullable=False, unique=True)
    password = Column(String(255), nullable=False)
    phone_number = Column(String(12))
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=text('now()'))
    rol_id = Column(Integer, ForeignKey("rol.id", ondelete="CASCADE"), nullable=False)



