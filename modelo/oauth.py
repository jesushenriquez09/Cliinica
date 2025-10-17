from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from config.db import get_db
from modelo.token import verify_token
from sqlalchemy.orm import Session

from models.db_p import Users


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/usuario/login")


async def get_current_user(token: str = Depends(oauth2_scheme),db: Session = Depends(get_db)):
      credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
      ) 
      token = verify_token(token,credentials_exception) 
      user = db.query(Users).filter(Users.id == token.id).first()
        
      return  user