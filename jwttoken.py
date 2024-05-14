#here is the jwttoken.py

from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
# from . import schemas 

SECRET_KEY = "ba524430b687c4076dbb695f98967ce68ad68cc299d1130b980edc089a88ab9e"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 180
RESET_TOKEN_EXPIRE_MINUTES=5


from fastapi import HTTPException, status

credentials_exception = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Incorrect Token or Token Expired",
    headers={"WWW-Authenticate": "Bearer"},
)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(access_token: str, credentials_exception):
    try:
        payload = jwt.decode(access_token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None or payload["exp"] < datetime.utcnow().timestamp():
            raise credentials_exception
        token_data = schemas.TokenData(email=email)
    except JWTError:
        raise credentials_exception
    return token_data.dict()