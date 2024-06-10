from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session
from user.main import  engine
import user
from user import models,main
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
# import router


# user.models.Base.metadata.create_all(bind=engine)


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#user
app.include_router(user.main.router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, workers=4)

    