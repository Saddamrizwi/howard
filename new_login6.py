import requests
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psycopg2
import os
import uvicorn
from uuid import uuid4
from fastapi.responses import HTMLResponse
import json
import tempfile
import requests
from langchain_community.llms import HuggingFaceHub


###############################################################
# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)   

# Database connection parameters
DB_HOST = 'monorail.proxy.rlwy.net'
DB_PORT = 39635
DB_USER = 'postgres'
DB_PASSWORD = '65jJ4lAqgQaP7u7ZVnIjBYiUgry82ZDP'
DB_NAME = 'railway'

class SignUpRequest(BaseModel):
    email: str
    password: str
    username: str

class LoginResponse(BaseModel):
    email: str
    password: str

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

def topic_model(text):
    llm = HuggingFaceHub(repo_id="facebook/bart-large-cnn",
                            task="summarization",
                            model_kwargs={"temperature":0.1, "max_length":15},
                            huggingfacehub_api_token='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')
    return llm(text)        

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
    """Validates user credentials."""
    conn = establish_db_connection()
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM session_table6 WHERE email = %s AND password = %s", (email, password))
        id = cur.fetchone()
    conn.close()
    if id is None:
        raise HTTPException(status_code=404, detail="User not found or invalid credentials.")
    else:
        return id[0]  # Returning the first element of the fetched row

def create_chat_id(conn, email_id):
    # Generate new chat_id
    chat_id = str(uuid4())
    # Store chat_id in the corresponding chat history table
    with conn.cursor() as cur:
        cur.execute("INSERT INTO chat_history6 (email_id, chat_id) VALUES (%s, %s)", (email_id, chat_id))
    conn.commit()
    return chat_id

def create_chat_history(conn):
    # Create chat history table if it doesn't exist
    chat_table_query = """
        CREATE TABLE IF NOT EXISTS chat_history6 (
            id SERIAL PRIMARY KEY,
            chat_id UUID DEFAULT uuid_generate_v4(),
            email_id INTEGER REFERENCES session_table6(id),
            all_messages TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            favourite BOOLEAN DEFAULT FALSE
        );"""
    with conn.cursor() as cur:
        cur.execute(chat_table_query)
    conn.commit()

# Establish database connection and create table/function
conn = establish_db_connection()
create_chat_history(conn)
close_db_connection(conn)

def update_favourite_status(chat_id, favourite):
    """Updates the favourite status in the database."""
    favourite = favourite.lower()
    with establish_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE public.chat_history6
                SET favourite = %s
                WHERE chat_id = %s;
            """, (favourite, chat_id))
    conn.commit()

def get_favourite(chat_id):
    with establish_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT favourite
                FROM public.chat_history6
                WHERE chat_id = %s;
            """, (chat_id,))
            rows = cur.fetchall()
    return rows

def upload_file(file, api_url):
    """Uploads a file to the specified API URL."""
    form_data = {"files": (file.filename, file.file)}
    response = requests.post(api_url, files=form_data)
    return response.json()    

def query_chat(query, llm):
    if llm.lower() == "open source":
        """Queries chat API with the given question."""
        # API_URL = "https://flowiseai-railway-production-51c3.up.railway.app/api/v1/prediction/c9e1bada-3ba0-4f45-b543-9ac19250bfa8"
        API_URL = "https://flowiseai-railway-production-51c3.up.railway.app/api/v1/prediction/4552bc8a-84ba-4b40-86f5-a1148729f815"
        payload = {"question": query}
        response = requests.post(API_URL, json=payload)
    else:
        """Queries chat API with the given question."""
        API_URL = "https://flowiseai-railway-production-51c3.up.railway.app/api/v1/prediction/fd18150b-eb44-455f-a789-aca906ff3ce9"
        payload = {"question": query}
        response = requests.post(API_URL, json=payload)     
    return response.json()    

@app.get("/Registration")
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

@app.post("/sign_up")
async def sign_up(sign_up_request: SignUpRequest):
    try:
        id, session_id = create_session(sign_up_request.email, sign_up_request.password, sign_up_request.username)
        return {"id": id, "session_id": session_id, "message": "User signed up successfully"}
    except ValueError as e:
        raise HTTPException(status_code=200, detail="User Already Exists")
    except Exception as e:
        raise HTTPException(status_code=422, detail="Unprocessable entity")

@app.post("/login")
async def login(loginresponse: LoginResponse):  
    email = loginresponse.email
    password = loginresponse.password
    # Establish database connection
    conn = establish_db_connection()
    # Create session
    id = validate_credentials(email, password)
    # Close database connection
    close_db_connection(conn)
    return {"message": "Login successful", "email_id": id}    

@app.post("/logout")
async def logout(id: str):
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

@app.delete("/delete_id")
async def delete_session(email_id: str):
    # Establish database connection
    conn = establish_db_connection()
    # Delete session
    with conn.cursor() as cur:
        cur.execute("DELETE FROM session_table6 WHERE email_id = %s", (email_id,))
    conn.commit()
    close_db_connection(conn)
    return {"message": "Session deleted successfully"}


@app.get("/dashboard/{session_id}")
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

@app.post("/upload_data")
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

@app.post("/get_history")
async def get_review(review_request: ReviewRequest):
    email_id = review_request.id
    chat_data = []
    topic = None  # Initialize topic as None in case no messages are found
    with establish_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT chat_id, all_messages, favourite
                FROM public.chat_history6
                WHERE email_id = %s;
            """, (email_id,))
            rows = cur.fetchall()
            for row in rows:
                chat_id = row[0]
                favourite = row[2]
                all_messages = row[1]
                if all_messages:  # Check if all_messages is not None
                    messages = json.loads(all_messages)  # Parse JSON data
                    if len(messages) >= 3:  # Check if there are at least 3 messages
                        # Accessing the third message
                        text = messages[2]["message"]
                        topic = topic_model(text)
                chat_data.append({
                    "chat_id": chat_id,
                    "topic": topic,
                    "favourite": favourite
                    })        
    return  chat_data

@app.post("/get_favourite")
async def get_review(review_favourite: ReviewFavourite):
    chat_id = review_favourite.chat_id
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
                SELECT chat_id, all_messages, favourite
                FROM public.chat_history6
                WHERE chat_id = %s;
            """, (chat_id,))
            rows = cur.fetchall()
            for row in rows:
                chat_id = row[0]
                all_messages = row[1]
                if all_messages:  # Check if all_messages is not None
                    messages = json.loads(all_messages)  # Parse JSON data
                    if len(messages) >= 3:  # Check if there are at least 3 messages
                        text = messages[2]["message"]
                        topic = topic_model(text)
                favourite = row[2]    
                chat_data.append({
                    "chat_id": chat_id,
                    "topic": topic,
                    "favourite": favourite
                })
    return chat_data

@app.post("/process_answer")
async def process_answer(review_response: ReviewResponse):
    email_id = review_response.email_id
    query = review_response.query
    llm = review_response.llm
    name = review_response.name
    chat_id = review_response.chat_id
    output = query_chat(query, llm) 
    parts = output['text'].split("Human:")[0]
    # Check if chat_id is empty, if yes, create a new chat_id
    if chat_id == "":
        with establish_db_connection() as conn:
            chat_id = create_chat_id(conn, email_id)
        conn.commit()  
    # Fetch existing chat history or initialize if none exists
    with establish_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT all_messages, favourite
                FROM public.chat_history6
                WHERE email_id = %s AND chat_id = %s;
            """, (email_id, chat_id))
            existing_row = cur.fetchone()
            favourite = existing_row[1]
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
            # Convert the conversation back to JSON string
            history_json = json.dumps(conversation)

            # Insert or update the row in the database
            if existing_row:
                cur.execute("""
                    UPDATE public.chat_history6
                    SET all_messages = %s
                    WHERE email_id = %s AND chat_id = %s;
                """, (history_json, email_id, chat_id))
            else:
                cur.execute("""
                    INSERT INTO public.chat_history6 (email_id, chat_id, all_messages)
                    VALUES (%s, %s, %s);
                """, (email_id, chat_id, history_json))
        conn.commit()   
    # Return updated chat history along with chat ID
    return {
        "answer": parts,
        "data": history_json,
        "chatId": chat_id,
        "llm": llm,
        "favourite": favourite
    }

@app.get("/new_chat")
async def new_chat(email_id: str):
    email_id: str
    with establish_db_connection() as conn:
        with conn.cursor() as cur:
            chat_id = create_chat_id(conn, email_id)
        conn.commit()
    return chat_id        

@app.post("/get_conversation")
async def get_chat(get_chat: GetChat):
    email_id = get_chat.email_id
    chat_id = get_chat.chat_id
    with establish_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT all_messages, timestamp, favourite
                FROM public.chat_history6
                WHERE email_id = %s AND chat_id = %s;
                    """, (email_id, chat_id,))
            rows = cur.fetchall()
            chat, timestamp, favourite = rows[0] if rows else ("", "")
    return chat, timestamp, favourite

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
