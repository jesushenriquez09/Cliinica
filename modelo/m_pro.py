from typing import Optional
from pydantic import BaseModel

class vhBase(BaseModel):
    name_product: str
    price: Optional[int]
    stock: Optional[int]
    placa: str
    color: str
    status_id: Optional[int]
    category_id: Optional[int]
    modelo_id: Optional[int]

class vhcreate(vhBase):
 pass

class vh(vhBase):
    id:int 
    user_id: Optional[int]

    class Config:
        from_attributes = True
      


    


    
    
