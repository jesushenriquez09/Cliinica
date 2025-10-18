from fastapi import FastAPI  
#from routes.products import  VH
from routes.Usuario import UsuarioRouter
from routes.nlp_routes import nlp_route 
from routes.citas_routes import citas_router 
from routes.diagnosticos_routes import router as router_diagnosticos
from alembic import command
from alembic.config import Config
#from models import db_p
from config.db import engine, Base, SessionLocal
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI() 

Base.metadata.create_all(bind=engine, checkfirst=True)

#alembic_cfg = Config("alembic.ini")
#command.ensure_version(alembic_cfg)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        '*'
    ],  # En producción, especifica los dominios permitidos en lugar de "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#app.include_router(VH)
app.include_router(UsuarioRouter)
app.include_router(nlp_route)
app.include_router(citas_router)
app.include_router(router_diagnosticos)