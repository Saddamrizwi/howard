from typing import List, Optional
from pydantic import BaseModel

class SignUpRequest(BaseModel):
    email: str
    password: str
    username: str

class LoginResponse(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class DataUpload(BaseModel):
    file:str 

class ReviewRequest(BaseModel):
    id: str       

class ReviewFavourite(BaseModel):
    chat_id: str
    favourite: str    

class ReviewResponse(BaseModel):
    email_id: str
    query: str
    llm: str
    name: str
    chat_id: str

class GetChat(BaseModel):
    email_id: str
    chat_id: str