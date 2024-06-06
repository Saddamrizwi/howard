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
from langchain_community.llms import HuggingFaceEndpoint
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
from .schemas import SignUpRequest,LoginResponse,DataUpload,ReviewRequest,ReviewFavourite,ReviewResponse,GetChat,TransactionResponse
from datetime import datetime, timedelta
from langdetect import detect
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import jieba
from langchain_core.runnables.base import RunnableSequence, Runnable
from langchain.prompts import PromptTemplate
import http.client
import base64
from typing import Dict


##################################
router = APIRouter()

free_usage_counts: Dict[str, int] = {}

def establish_db_connection():
    """Establishes connection to the PostgreSQL database and creates necessary tables."""
    conn = psycopg2.connect(
        database="latest_llm",
        user="postgres",
        password="King#123",
        host="localhost",
        port="5432"
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
            return      
        else:
            return None

def topic_model(text):
    text = text.strip()
    # Updated LLM instantiation using HuggingFaceEndpoint
    llm = HuggingFaceEndpoint(
        endpoint_url="https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct",
        huggingfacehub_api_token='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',
        temperature=0.1,
        max_new_tokens=10)
    # Define the prompt template
    template = """ Summarize whole text and give a suitable title from this text in just 3 words only.
    {text}
    TITLE OF SUMMARY:
    """
    prompt = PromptTemplate(template=template, input_variables=["text"])
    # Create a runnable sequence
    runnable_sequence = RunnableSequence(first=prompt, last=llm)
    # Run the sequence
    result = runnable_sequence.invoke({"text": text}).split("\n")[0].replace('"', '').strip()
    return result

def transaction_record_1(session_id, created_date, payment_status, amount_total, currency, payment_intent, user_id):
    print("cfiwvfgweuwegiwe",session_id,created_date, payment_status, amount_total, currency, payment_intent, user_id)
    try:
        conn = establish_db_connection()
        with conn.cursor() as cur:
            if payment_status.lower() == "paid":
                user_type = 'active'
            else:
                amount_total = 0    
            expiry_date = created_date + timedelta(days=30)
            # Insert the record
            insert_query = """
                UPDATE public.transaction
                SET session_id = %s, created_date = %s, expiry_date = %s, user_type = %s, payment_intent = %s, amount = %s, amount_left = %s, currency = %s
                WHERE user_id = %s
            """
            cur.execute(insert_query, (
                session_id,
                created_date,
                expiry_date,
                user_type,
                payment_intent,
                amount_total,
                amount_total,
                currency,
                user_id
            ))
        print("vtqcuweyfgscvgjdutfgejwvdSyzwsdvhfw",insert_query)
        conn.commit()
        conn.close()
    except Exception as e:
        return f"An error occurred: {e}" 

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
    id = None  # Initialize id to ensure it has a default value
    username = None  # Initialize username
    role_id = None  # Initialize role_id
    with conn.cursor() as cur:
        cur.execute("SELECT id, username, role_id FROM session_table6 WHERE email = %s AND password = %s", (email, password))
        rows = cur.fetchall()
        if rows:
            for row in rows:
                id = row[0]
                username = row[1]
                role_id = row[2]
    conn.close()
    if id is None:
        raise HTTPException(status_code=404, detail="User not found or invalid credentials.")
    else:
        return id, username, role_id     
     

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
            # Define the enum type if it doesn't exist
        enum_query = """
            DO $$ BEGIN
                IF NOT EXISTS(SELECT 1 FROM pg_type WHERE typname = 'user_type_enum') THEN
                    CREATE TYPE user_type_enum AS ENUM ('active', 'inactive', 'free');
                END IF;
            END $$;
            """
        with conn.cursor() as cur:
            cur.execute(enum_query)

        # Create chat price table if it doesn't exist
        price_record_query = """
            CREATE TABLE IF NOT EXISTS transaction (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES session_table6(id),
                payment_intent VARCHAR(255),
                currency VARCHAR(10),
                session_id VARCHAR(255) NOT NULL,
                created_date TIMESTAMP,
                expiry_date TIMESTAMP,
                user_type user_type_enum NOT NULL,
                amount NUMERIC,
                amount_left NUMERIC
            );
            """
        with conn.cursor() as cur:
            cur.execute(price_record_query)
        
        conn.commit()
    except Exception as e:
        pass 
        raise HTTPException(status_code=500, detail="Internal Server Error")               

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
    if llm.lower() == "deepseek-ai":
        API_URL = "https://flowiseai-railway-production-51c3.up.railway.app/api/v1/prediction/eee04920-13ad-43d1-969e-1a22bfab992c"
        payload = {"question": query}
        response = requests.post(API_URL, json=payload)
    elif llm.lower() == "open ai":
        API_URL = "https://flowiseai-railway-production-51c3.up.railway.app/api/v1/prediction/fd18150b-eb44-455f-a789-aca906ff3ce9"
        payload = {"question": query}
        response = requests.post(API_URL, json=payload) 
    else:
        # if llm.lower() == "llama3 8b":
        API_URL = "https://flowiseai-railway-production-51c3.up.railway.app/api/v1/prediction/c9e1bada-3ba0-4f45-b543-9ac19250bfa8"
        # API_URL = "https://flowiseai-railway-production-51c3.up.railway.app/api/v1/prediction/4552bc8a-84ba-4b40-86f5-a1148729f815"
        payload = {"question": query}
        response = requests.post(API_URL, json=payload)     
    return response.json() 

def get_answer(email_id, query, llm, chat_id):    
    page_contents = []
    try:
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
        all_answers = parts + '\n' + 'metadata: ' + ', '.join([f"{page['title']}: {page['content']}\n" for page in page_contents])   
        one_chat = query + parts
        credit_left = get_credit(email_id, one_chat)  
        # Check if chat_id is empty, if yes, create a new chat_id
        if chat_id == "":
            with    () as conn:
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
                    "name": "User"
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
            "whole_answer": all_answers,
            "answer": parts,
            "data": history_json,
            "chatId": chat_id,
            "llm": llm,
            "favourite": favourite,
            "metadata": page_contents,
            "new_credit": credit_left
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}")

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

@router.post("/sign_up",tags=['authentication'])
async def sign_up(sign_up_request: schemas.SignUpRequest):
    try:
        
        id, session_id = create_session(sign_up_request.email, sign_up_request.password, sign_up_request.username) 
        return {"id": id, "session_id": session_id, "message": "User signed up successfully"}
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/login", tags=['authentication'])
async def login(loginresponse: OAuth2PasswordRequestForm = Depends()):
    email = loginresponse.username
    password = loginresponse.password
    # Validate credentials
    user = validate_credentials(email, password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    user_id, user_name, role_id = user
    conn = establish_db_connection()
    subscription_status = 'free'
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM public.transaction WHERE user_id = %s ORDER BY id ASC", (user_id,))
            rows = cur.fetchall()
            print("rowsss", rows)
            if not rows:
                # Insert default record if no transaction exists for the user
                created_date = datetime.now()
                expiry_date = created_date + timedelta(days=30)
                session_id = None
                amount = 5
                amount_left = 5
                currency = None
                payment_intent = None
                user_type = 'free'
                free_usage_counts = 3

                insert_query = """
                INSERT INTO transaction (user_id, session_id, created_date, expiry_date, user_type, amount, amount_left, payment_intent, currency, free_usage_counts)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                """
                cur.execute(insert_query, (
                    user_id, 
                    session_id, 
                    created_date, 
                    expiry_date, 
                    user_type,
                    amount,
                    amount_left,
                    payment_intent,
                    currency,
                    free_usage_counts  
                ))
                conn.commit()
            else:
                for row in rows:
                    print('row', row)
                    if isinstance(row, tuple):
                        user_type = row[7]
                        print('user_type', user_type)
                        subscription_status=user_type
                                
            access_token = create_access_token(data={"sub": email})
            print("access_token", access_token)
            hello={
                "access_token": access_token, 
                "token_type": "bearer", 
                "email_id": user_id, 
                "username": user_name, 
                "user_type": subscription_status,
                "role_id": role_id
            }
            print("hellooooooooooooooooooooooooooooooooooooooooooooooooooooooooo",hello)
            return {
                "access_token": access_token, 
                "token_type": "bearer", 
                "email_id": user_id, 
                "username": user_name, 
                "user_type": subscription_status,
                "role_id": role_id
            }
    except Exception as e:
        conn.rollback()
        print(f"Database error: {e}")
    finally:
        conn.close()
    
    
    # # Create access token
    # access_token = create_access_token(data={"sub": email})
    # return {"access_token": access_token, "token_type": "bearer", "email_id": user_id, "username": user_name, "type_is_paid": subscription_status}  

@router.post("/transaction")
async def transaction(transactionresponse: TransactionResponse):
    email_id = transactionresponse.email_id
    session_id = transactionresponse.session_id
    created_date = datetime.strptime(transactionresponse.created_date, "%Y-%m-%d %H:%M:%S.%f")
    type_is_paid=transactionresponse.type_is_paid
    try:
        transaction_record_1(email_id, session_id, created_date, type_is_paid)
        return{"message": "Database updated successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")     
 

secret_key='sk_test_51PEnAKLD2jJqby2hSGIkRf9h4otYfKDoEI6R53EArk6RrnXzj5SuXy3ztFDkpOBjuoGAdhMlTGVu0IFJpsD61ebP00SMSMwP0D' 

@router.get("/webhook_callback")
async def webhook_callback(request: Request):
    query_params = dict(request.query_params)
    session_id = request.query_params.get('session_id')
    user_id = request.query_params.get('user')
    print("gckvuwwec",user_id)

    if not session_id or not user_id:
        raise HTTPException(status_code=400, detail="Missing session_id or user_id")

    stripe_url = f"https://api.stripe.com/v1/checkout/sessions/{session_id}"
    encoded_secret_key = base64.b64encode(f"{secret_key}:".encode()).decode()

    conn = http.client.HTTPSConnection("api.stripe.com")
    payload = ''
    headers = {
        'Authorization': f'Basic {encoded_secret_key}'
    }
    conn.request("GET", f"/v1/checkout/sessions/{session_id}", payload, headers)
    res = conn.getresponse()
    data = res.read()

    # Decode the response from Stripe
    stripe_response = data.decode("utf-8")
    try:
        stripe_data = json.loads(stripe_response)
    except json.JSONDecodeError as e:
        return {"error": "Failed to parse JSON response from Stripe"}

    session_id = stripe_data.get("id")
    payment_status = stripe_data.get("payment_status")
    payment_intent = stripe_data.get("payment_intent")
    amount_total = stripe_data.get("amount_total")
    currency = stripe_data.get("currency")
    created_date = datetime.now()

    try:
        transaction_record_1(session_id, created_date, payment_status, amount_total, currency, payment_intent, user_id)
        redirect_url = "http://20.84.59.3:3000/payment_status/success"          
        return redirect(redirect_url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

    return {"message": stripe_response}

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
                    ORDER BY timestamp DESC;        ;
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
    chat_id = review_response.chat_id
    page_contents = []
    try:
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
                    "name": "User"
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
        raise HTTPException(status_code=500, detail=f"{e}")


    
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