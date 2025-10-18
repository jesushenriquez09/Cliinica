from datetime import datetime, timedelta, timezone
from typing import Optional
from jose import JWTError, jwt
from modelo.m_user import TokenData 


SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 120

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(days=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt
def verify_token(token:str, credentials_exception):
    try:
        print(f"Token recibido: {token}")
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        id: Optional[int] = payload.get("user_id")
        role = payload.get("role")
        
        # Asegurarse de que el rol sea una cadena
        role_str = str(role) if role is not None else None
        
        if id is None:
            raise credentials_exception
        
        token_data = TokenData(id=id, role=role_str)
        #print(f"TokenData generado: {token_data}")
    except JWTError as e:
        print(f"Error decodificando token: {e}")
        raise credentials_exception
    return token_data