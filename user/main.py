import requests
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request,Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
import psycopg2
import os
import uvicorn
from uuid import uuid4
from fastapi.responses import HTMLResponse
import json
import tempfile
from langchain_community.llms import HuggingFaceHub
from . import  models
from fastapi import Request
from . import  models, schemas,hashing,oauth2
from .jwttoken import  create_access_token
from fastapi import APIRouter
from fastapi import FastAPI
from database import engine
from sqlalchemy import create_engine, Table, Column, Integer, String, Text, Boolean, ForeignKey, DateTime, MetaData
from sqlalchemy.orm import Session
from typing import List 
from .schemas import SignUpRequest,LoginResponse,DataUpload,ReviewRequest,ReviewFavourite,ReviewResponse,GetChat

router = APIRouter()

# Database connection parameters
DB_HOST = 'localhost'
DB_PORT = 5432
DB_USER = 'postgres'
DB_PASSWORD = 'King#123'
DB_NAME = 'latest_llm'


def establish_db_connection():
    """Establishes connection to the PostgreSQL database and creates necessary tables."""
    conn = psycopg2.connect(
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    # Create sessions table if not exists
    session_table = """ 
    CREATE TABLE IF NOT EXISTS session_table6 (
        id SERIAL PRIMARY KEY,
        email VARCHAR(255) UNIQUE,
        password VARCHAR(255),
        username VARCHAR(255)
    );"""
    with conn.cursor() as cur:
        cur.execute(session_table)
    conn.commit()
    return conn

def close_db_connection(conn):
    """Closes the database connection."""
    if conn:
        conn.close()

def num_tokens_from_string(string: str) -> int:
    if not string:
        return 0
    # Detect language
    language = detect(string)
    if language == 'en':
        # English processing
        tokens = word_tokenize(string)
        stop_words = set(stopwords.words('english'))
        tokens = [token.lower() for token in tokens if token.lower() not in stop_words and token.isalnum()]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    elif language == 'zh-cn' or language == 'zh-tw':
        # Chinese processing
        tokens = jieba.lcut(string)
        tokens = [token for token in tokens if len(token.strip()) > 0] 
    else:
        # Unsupported language
        return 0
    unique_tokens = set(tokens)
    return len(unique_tokens)   

def get_embedding_cost(num_tokens):
    return num_tokens_from_string(num_tokens)/1000*0.01

def get_credit(user_id, num_tokens):
    conn = establish_db_connection()
    with conn.cursor() as cur:
        cur.execute("SELECT amount_left FROM transaction WHERE user_id = %s", (user_id,))
        rows = cur.fetchall()
        if rows:
            amount_left = rows[0][0]  # Assuming there is only one row per user_id
            new_credit = float(amount_left) - float(get_embedding_cost(num_tokens))
            cur.execute("UPDATE public.transaction SET amount_left = %s WHERE user_id = %s", (new_credit, user_id))
            conn.commit()
            conn.close()
            return new_credit    
        else:
            return None

def topic_model(text):
    text = text.strip()
    llm = HuggingFaceHub(repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
                        model_kwargs={"temperature": 0.1, "max_length":10},
                        huggingfacehub_api_token='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',)
                        #    huggingfacehub_api_token='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',)
                        #    huggingfacehub_api_token='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',)
                        #    huggingfacehub_api_token='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',)
                        #    huggingfacehub_api_token='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',)
    template = """ Summarize whole text and give a suitable title from this text in just 3 words only.
    {text}
    TITLE OF SUMMARY:
    """    
    prompt = PromptTemplate(template=template, input_variables=["text"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    topic = llm_chain.run(text)
    title = topic.split("TITLE OF SUMMARY:\n")[1].split("\n")[0]
    zabbar = title.replace('"', '').strip()
    return zabbar

def transaction_record(user_id, type_is_paid):
    try:
        conn = establish_db_connection()
        with conn.cursor() as cur:
            amount = 100 if type_is_paid.lower() == "true" else 20
            created_date = datetime.now()
            expiry_date = created_date + timedelta(days=30)
            # Insert the record
            insert_query = """
                INSERT INTO transaction (user_id, session_id, created_date, expiry_date, type, amount, amount_left)
                VALUES (%s, %s, %s, %s, %s, %s, %s);
            """
            cur.execute(insert_query, (
                user_id, 
                1, 
                created_date, 
                expiry_date, 
                type_is_paid, 
                amount, 
                amount 
            ))
        conn.commit()
        conn.close()
    except Exception as e:
        return (f"An error occurred: {e}")

def create_session(email, password, username):
    """Creates a new session for the user if not already registered."""
    conn = establish_db_connection()
    # Check if user with the provided email already exists
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM session_table6 WHERE email = %s", (email,))
        existing_user = cur.fetchone()
        if existing_user:
            conn.close()
            raise ValueError("User is already registered. Please log in.")
        else:
            # If user does not exist, proceed with creating a new session
            cur.execute("INSERT INTO session_table6 (email, password, username) VALUES (%s, %s, %s) RETURNING id",
                        (email, password, username))
            session_id = str(uuid4())
            id = cur.fetchone()[0]
    conn.commit()
    conn.close()
    return id, session_id

def validate_credentials(email, password):
    try:
        """Validates user credentials."""
        conn = establish_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT id, username FROM session_table6 WHERE email = %s AND password = %s", (email, password))
            rows = cur.fetchall()
            if rows:
                for row in rows:
                    id = row[0]
                    username = row[1]
        conn.close()
        if id is None:
            raise HTTPException(status_code=404, detail="User not found or invalid credentials.")
        else:
            return id, username 
    except Exception as e:
        #pass
        raise HTTPException(status_code=500, detail="Internal Server Error")    
     

def create_chat_id(conn, email_id):
    try:
        # Generate new chat_id
        chat_id = str(uuid4())
        # Store chat_id in the corresponding chat history table
        with conn.cursor() as cur:
            cur.execute("INSERT INTO chat_history7 (email_id, chat_id) VALUES (%s, %s)", (email_id, chat_id))
        conn.commit()
        return chat_id
    except Exception as e:
        pass
        # raise HTTPException(status_code=500, detail="Internal Server Error") 
    

def create_chat_history(conn):
    try:
        # Create chat history table if it doesn't exist
        chat_table_query = """
            CREATE TABLE IF NOT EXISTS chat_history7 (
                id SERIAL PRIMARY KEY,
                chat_id UUID DEFAULT uuid_generate_v4(),
                email_id INTEGER REFERENCES session_table6(id),
                all_messages TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                favourite BOOLEAN DEFAULT FALSE,
                topic TEXT
            );"""
        with conn.cursor() as cur:
            cur.execute(chat_table_query)
        conn.commit()
    except Exception as e:
        pass
        # raise HTTPException(status_code=500, detail="Internal Server Error")   

def create_price_record(conn):
    try:
        # Create chat history table if it doesn't exist
        price_record_query = """
            CREATE TABLE IF NOT EXISTS transaction (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES session_table6(id),
                session_id INTEGER NOT NULL,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expiry_date TIMESTAMP,
                type BOOLEAN,
                amount NUMERIC,
                amount_left NUMERIC
            );"""
        with conn.cursor() as cur:
            cur.execute(price_record_query)
        conn.commit()
    except Exception as e:
        pass
        # raise HTTPException(status_code=500, detail="Internal Server Error")               

# Establish database connection and create table/function
conn = establish_db_connection()
create_chat_history(conn)
create_price_record(conn)
close_db_connection(conn)

def update_favourite_status(chat_id, favourite):
    try:
        """Updates the favourite status in the database."""
        favourite = favourite.lower()
        with establish_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE public.chat_history7
                    SET favourite = %s
                    WHERE chat_id = %s;
                """, (favourite, chat_id))
        conn.commit()
    except Exception as e:
        pass
        # raise HTTPException(status_code=500, detail="Internal Server Error")     

def get_favourite(chat_id):
    try:
        with establish_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT favourite
                    FROM public.chat_history7
                    WHERE chat_id = %s;
                """, (chat_id,))
                rows = cur.fetchall()
        return rows
    except Exception as e:
        pass
        # raise HTTPException(status_code=500, detail="Internal Server Error") 

def upload_file(file, api_url):
    """Uploads a file to the specified API URL."""
    form_data = {"files": (file.filename, file.file)}
    response = requests.post(api_url, files=form_data)
    return response.json()    

def query_chat(query, llm):
    """Queries chat API with the given question."""
    if llm.lower() == "chinese":
        API_URL = "https://flowiseai-railway-production-51c3.up.railway.app/api/v1/prediction/eee04920-13ad-43d1-969e-1a22bfab992c"
        payload = {"question": query}
        response = requests.post(API_URL, json=payload)
    elif llm.lower() == "open ai":
        API_URL = "https://flowiseai-railway-production-51c3.up.railway.app/api/v1/prediction/fd18150b-eb44-455f-a789-aca906ff3ce9"
        payload = {"question": query}
        response = requests.post(API_URL, json=payload) 
    else:
        if llm.lower() == "open source":
            API_URL = "https://flowiseai-railway-production-51c3.up.railway.app/api/v1/prediction/c9e1bada-3ba0-4f45-b543-9ac19250bfa8"
            # API_URL = "https://flowiseai-railway-production-51c3.up.railway.app/api/v1/prediction/4552bc8a-84ba-4b40-86f5-a1148729f815"
            payload = {"question": query}
            response = requests.post(API_URL, json=payload)     
    return response.json()    

@router.get("/Registration")
async def login_form():
    return HTMLResponse("""
        <form method="post">
        <label for="email">Enter your Gmail address:</label><br>
        <input type="email" id="email" name="email" required><br>
        <label for="password">Enter your password:</label><br>
        <input type="password" id="password" name="password" required><br>
        <label for="username">Enter your username:</label><br>
        <input type="text" id="username" name="username" required><br>
        <input type="submit" value="Submit">
        </form>
    """)

@router.post("/sign_up",tags=['authetication'])
async def sign_up(sign_up_request: SignUpRequest):
    try:
        
        id, session_id = create_session(sign_up_request.email, sign_up_request.password, sign_up_request.username) 
        return {"id": id, "session_id": session_id, "message": "User signed up successfully"}
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/login",tags=['authetication'])
async def login(loginresponse: OAuth2PasswordRequestForm = Depends()):  
    email = loginresponse.username
    password = loginresponse.password
    
    # Validate credentials
    id = validate_credentials(email, password)
    
    # Create access token
    access_token = create_access_token(data={"sub": email})
    
    return {"access_token": access_token, "token_type": "bearer"} 

@router.post("/logout")
async def logout(id: str):
    try:
        # Check if session_id exists in the database
        conn = establish_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM session_table6 WHERE id = %s", (id,))
            session_data = cur.fetchone()
        close_db_connection(conn)
        if session_data:
            return {"message": "Logout successful"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        pass
        # raise HTTPException(status_code=500, detail="Internal Server Error")     

@router.delete("/delete_id")
async def delete_session(email_id: str):
    # Establish database connection
    conn = establish_db_connection()
    # Delete session
    with conn.cursor() as cur:
        cur.execute("DELETE FROM session_table6 WHERE email_id = %s", (email_id,))
    conn.commit()
    close_db_connection(conn)
    return {"message": "Session deleted successfully"}


@router.get("/dashboard/{session_id}")
async def dashboard(id: str):
    # Check if session_id exists in the database
    conn = establish_db_connection()
    with conn.cursor() as cur:
        cur.execute("SELECT username FROM session_table6 WHERE id = %s", (id,))
        session_data = cur.fetchone()
    close_db_connection(conn)
    if session_data:
        name = session_data[0]
        # Here you can implement logic to fetch user's data using the email
        return {"message": f"Welcome to your dashboard, {name.capitalize()}!"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@router.post("/upload_data")
async def upload_data(file: UploadFile = File(...)):
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension in ['.pdf', '.docx', '.doc', '.pptx', '.ppt']:
        api_url = {
            '.pdf': "https://flowiseai-railway-production-51c3.up.railway.app/api/v1/vector/upsert/4552bc8a-84ba-4b40-86f5-a1148729f815",
            '.docx': "https://flowiseai-railway-production-51c3.up.railway.app/api/v1/vector/upsert/f6079b47-29d1-4032-a7ce-055990b3661b",
            '.doc': "https://flowiseai-railway-production-51c3.up.railway.app/api/v1/vector/upsert/f6079b47-29d1-4032-a7ce-055990b3661b",
            '.pptx': "https://flowiseai-railway-production-51c3.up.railway.app/api/v1/vector/upsert/048ecc50-cbf6-4b0d-a6c7-983dfc81815b",
            '.ppt': "https://flowiseai-railway-production-51c3.up.railway.app/api/v1/vector/upsert/048ecc50-cbf6-4b0d-a6c7-983dfc81815b"
        }.get(file_extension)
        return upload_file(file, api_url)
    else:
        return {"message": f"Try Another File: Processing skipped for {file.filename}"}

@router.post("/get_history")
async def get_review(review_request: ReviewRequest,current_user:schemas.LoginResponse=Depends(oauth2.get_current_user)):
    email_id = review_request.id
    try:
        chat_data = []
        with establish_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT chat_id, all_messages, favourite, topic, timestamp
                    FROM public.chat_history7
                    WHERE email_id = %s
                    ORDER BY chat_id DESC;        ;
                """, (email_id,))
                rows = cur.fetchall()
                for row in rows:
                    chat_id = row[0]
                    favourite = row[2]
                    topic = row[3]
                    if topic is None:
                        all_messages = row[1]
                        if all_messages:  # Check if all_messages is not None
                            messages = json.loads(all_messages)  # Parse JSON data
                            if len(messages) >= 3:  # Check if there are at least 3 messages
                                text = messages[2]["message"]
                                topic = topic_model(text)
                                # Inserting the topic into the database where chat_id matches
                                with conn.cursor() as cur:
                                    cur.execute("""
                                        UPDATE public.chat_history7
                                        SET topic = %s
                                        WHERE chat_id = %s ;
                                        """, (topic, chat_id))
                    timestamp = row[4]                
                    chat_data.append({
                        "chat_id": chat_id,
                        "topic": topic,
                        "favourite": favourite,
                        "time": timestamp
                        })   
        conn.commit()
        conn.close()  # Close connection               
        close_db_connection(conn)                  
        return  chat_data
    except Exception as e:
        pass
        # raise HTTPException(status_code=500, detail="Internal Server Error") 
    
@router.post("/get_favourite")
async def get_review(review_favourite: ReviewFavourite,current_user:schemas.LoginResponse=Depends(oauth2.get_current_user)):
    chat_id = review_favourite.chat_id
    try:
        chat_data = []
        # Update favourite status if requested and there's a mismatch
        if review_favourite.favourite.lower() != "true" and review_favourite.favourite.lower() != "false":
            favourite = get_favourite(chat_id)
        else:   
            if review_favourite.favourite.lower() != get_favourite(chat_id):
                favourite = update_favourite_status(chat_id, review_favourite.favourite) 
        with establish_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT chat_id, all_messages, favourite, topic
                    FROM public.chat_history7
                    WHERE chat_id = %s;
                """, (chat_id,))
                rows = cur.fetchall()
                for row in rows:
                    chat_id = row[0]
                    topic = row[3]
                    if topic is None:
                        all_messages = row[1]
                        if all_messages:  # Check if all_messages is not None
                            messages = json.loads(all_messages)  # Parse JSON data
                            if len(messages) >= 3:  # Check if there are at least 3 messages
                                text = messages[2]["message"]
                                topic = topic_model(text)
                                # Inserting the topic into the database where chat_id matches
                                with conn.cursor() as cur:
                                    cur.execute("""
                                        UPDATE public.chat_history7
                                        SET topic = %s
                                        WHERE chat_id = %s ;
                                        """, (topic, chat_id))
                    favourite = row[2]    
                    chat_data.append({
                        "chat_id": chat_id,
                        "topic": topic,
                        "favourite": favourite
                    })
        conn.commit()
        conn.close()  # Close connection             
        close_db_connection(conn)            
        return chat_data
    except Exception as e:
        pass
        # raise HTTPException(status_code=500, detail="Internal Server Error") 
    

@router.post("/process_answer")
async def process_answer(review_response: ReviewResponse,current_user:schemas.LoginResponse=Depends(oauth2.get_current_user)):
    email_id = review_response.email_id
    query = review_response.query
    llm = review_response.llm
    name = review_response.name
    chat_id = review_response.chat_id
    type = review_response.type
    page_contents = []
    try:
        # transaction_record(email_id, type)
        output = query_chat(query, llm) 
        parts = output['text'].split("Human:")[0].strip()#.split("AI Assistant:")[1]
        metadata = output['sourceDocuments']
        for doc in metadata:
            title = doc.get('metadata', {}).get('pdf', {}).get('Title', 'No Title')
            page_cont = doc.get('pageContent', 'No Content')
            page_content = ' '.join(page_cont.strip().split()[:8])
            page_info = {
                "title": title,
                "content": page_content
            }
            page_contents.append(page_info)
        one_chat = query + parts
        credit_left = get_credit(email_id, one_chat)
        # Check if chat_id is empty, if yes, create a new chat_id
        if chat_id == "":
            with establish_db_connection() as conn:
                chat_id = create_chat_id(conn, email_id)
            conn.commit()  
        # Fetch existing chat history or initialize if none exists
        with establish_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT all_messages, favourite, topic
                    FROM public.chat_history7
                    WHERE email_id = %s AND chat_id = %s;
                """, (email_id, chat_id))
                existing_row = cur.fetchone()
                favourite = existing_row[1] if existing_row else None
                history_json = existing_row[0] if existing_row else None
                if history_json:
                    conversation = json.loads(history_json)
                else:
                    conversation = []
                    conversation.append({
                        "message": "Hi, I am ChatBot AI Assistant!",
                        "type": "bot",
                        "image": "https://picsum.photos/50/50",
                        "name": "AI ChatBot"
                    })
                
                # Append user's message to the conversation
                user_message = {
                    "message": f"{query}",
                    "type": "user",
                    "image": "https://picsum.photos/50/50",
                    "name": name if name else "User"
                }
                conversation.append(user_message)
                
                # Append new message to the conversation
                new_message = {
                    "message": f"{parts}",
                    "type": "bot",
                    "image": "https://picsum.photos/50/50",
                    "name": "AI ChatBot"
                }
                conversation.append(new_message)
                topic = existing_row[2] if existing_row else None
                if topic is None:
                    if len(conversation) >= 3:  # Check if there are at least 3 messages
                        text = conversation[2]["message"]
                        topic = topic_model(text)
                # Convert the conversation back to JSON string
                history_json = json.dumps(conversation)        
                # Insert or update the row in the database
                if existing_row:
                    cur.execute("""
                        UPDATE public.chat_history7
                        SET all_messages = %s, topic = %s
                        WHERE email_id = %s AND chat_id = %s;
                    """, (history_json, topic, email_id, chat_id))
                else:
                    cur.execute("""
                        INSERT INTO public.chat_history7 (email_id, chat_id, all_messages)
                        VALUES (%s, %s, %s, %s);
                    """, (email_id, chat_id, history_json, topic))
            conn.commit()   
        close_db_connection(conn)    
        # Return updated chat history along with chat ID
        return {
            "answer": parts,
            "data": history_json,
            "chatId": chat_id,
            "llm": llm,
            "favourite": favourite,
            "metadata": page_contents,
            "new_credit": credit_left
        }
    except Exception as e:
        raise HTTPException(detail=f"{e}")
    
@router.get("/new_chat")
async def new_chat(email_id: str):
    email_id: str
    try:
        with establish_db_connection() as conn:
            with conn.cursor() as cur:
                chat_id = create_chat_id(conn, email_id)
            conn.commit()
        close_db_connection(conn)    
        return chat_id        
    except Exception as e:
        pass
        # raise HTTPException(status_code=500, detail="Internal Server Error") 
    

@router.post("/get_conversation")
async def get_chat(get_chat: GetChat,current_user:schemas.LoginResponse=Depends(oauth2.get_current_user)):
    email_id = get_chat.email_id
    chat_id = get_chat.chat_id
    try:
        with establish_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT all_messages, timestamp, favourite
                    FROM public.chat_history7
                    WHERE email_id = %s AND chat_id = %s;
                        """, (email_id, chat_id,))
                rows = cur.fetchall()
                if rows:
                    chat, timestamp, favourite = rows[0]
                else:
                    chat, timestamp, favourite = "", "", ""
        close_db_connection(conn)            
        return chat, timestamp, favourite
    except Exception as e:
        pass
        # raise HTTPException(status_code=500, detail="Internal Server Error")
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)










# Dependency to get the database session
# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

# @router.get("/")
# def read_root():
#     return {"Hello": "World"}


# @router.post("/sign_up",tags=["authentication"],status_code=200,response_model=schemas.SignUpRequest)  
# def create_user(request:schemas.SignUpRequest,db: Session = Depends(get_db)):
#     new_user=models.User(username=request.username,email=request.email,password=hashing.Hash.bcrypt(request.password))
#     print("cvfgvcgsfvfycsfvsfsvcsfdv---------->",new_user)
#     existing_email = db.query(models.User).filter(models.User.email == request.email).first()
#     print("hgdggvgfvgarcgcvv",existing_email)
#     if existing_email:
#         raise HTTPException(status_code=400, detail="Email already registered")
#     db.add(new_user)
#     db.commit()
#     db.refresh(new_user)
#     return new_user





#here is the api for create chat..


# @router.post("/process_answer")
# async def process_answer(
#     # current_user: schemas.TokenData = Depends(oauth2.get_current_user),
#     query: str = Form(...),
#     llm: str = Form(...),
#     chat_id: int = Form(...),
#     db: Session = Depends(get_db)
#     ):
#     # parts = output['text'].split("Human:")[0]):
#     # user_id = review_response.email_id
#     if not chat_id:
#         new_chat = models.ChatHistory(email_id=email_id)
#         print("fcgssdhcad",)
#         db.add(new_chat)
#         db.commit()
#         db.refresh(new_chat)
#         chat_id = new_chat.chat_id

#     chat_history = db.query(ChatHistory).filter_by(email_id=email_id, chat_id=chat_id).first()

#     if chat_history:
#         conversation = json.loads(chat_history.all_messages) if chat_history.all_messages else []
#         favourite = chat_history.favourite
#     else:
#         conversation = [{
#             "message": "Hi, I am ChatBot AI Assistant!",
#             "type": "bot",
#             "image": "https://picsum.photos/50/50",
#             "name": "AI ChatBot"
#         }]
#         favourite = False

#     user_message = {
#         "message": f"{query}",
#         "type": "user",
#         "image": "https://picsum.photos/50/50",
#         "name": name if name else "User"
#     }
#     conversation.append(user_message)

#     new_message = {
#         "message": f"{parts}",
#         "type": "bot",
#         "image": "https://picsum.photos/50/50",
#         "name": "AI ChatBot"
#     }
#     conversation.append(new_message)
#     history_json = json.dumps(conversation)

#     if chat_history:
#         chat_history.all_messages = history_json
#         db.commit()
#     else:
#         new_chat = ChatHistory(
#             email_id=email_id,
#             chat_id=chat_id,
#             all_messages=history_json
#         )
#         db.add(new_chat)
#         db.commit()

#     return {
#         "answer": parts,
#         "data": history_json,
#         "chatId": chat_id,
#         "llm": llm,
#         "favourite": favourite
#     }



