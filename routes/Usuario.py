from datetime import timedelta
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.security.oauth2 import OAuth2PasswordRequestForm
from config.db import  SessionLocal, get_db
from passlib.context import CryptContext
from modelo import oauth
from modelo.oauth import get_current_user
from models.db_p import  Users
from sqlalchemy.orm import Session
from models import db_p
from modelo import m_pro
from modelo.m_user import Login, Token, users
from modelo.token import create_access_token
import json


UsuarioRouter = APIRouter()


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto") 

@UsuarioRouter.post("/usuario")
def get_user(user: users,db: Session = Depends(get_db)):
    existe = db.query(Users).filter(Users.username == user.username).first()
    if existe:
        return JSONResponse("usuario ya se encuentra en uso")
    if existe is None:
            hashed_password = pwd_context.hash(user.password)
            user.password = hashed_password
            db_item = Users(**user.model_dump())
            db.add(db_item)
            db.commit()
            db.refresh(db_item)
    else:
        raise HTTPException(
            status_code=404, detail="product with this name already exists "
         )
    #vh_query = db.query(Users).filter(Users.id == db_item.id).first()
    return  db_item
@UsuarioRouter.post("/usuario/login")
def get_user(user_credentials:OAuth2PasswordRequestForm=Depends(),db: Session = Depends(get_db)):
    user = db.query(Users).filter(Users.username == user_credentials.username).first()
    rol_name = user.rol.rol
    print(rol_name)
    if not user or not pwd_context.verify(user_credentials.password, user.password):
        return JSONResponse("Incorrect username or password")
        raise HTTPException(409, "Incorrect username or password")

    access_token =  create_access_token(data={"user_id": user.id, "role": user.rol_id})
    token = {"access_token": access_token, "username":user.username, "userID":user.id, "token_type": "bearer", "role": rol_name}
    return JSONResponse(token)