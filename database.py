# from sqlalchemy import create_engine
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker

# SQLALCHEMY_DATABASE_URL = "postgresql://postgres:King#123@localhost/llm"

# engine = create_engine(SQLALCHEMY_DATABASE_URL)
# print("jvfufcuyvauu")
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base = declarative_base()

from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String

# Database URL
SQLALCHEMY_DATABASE_URL = "postgresql://postgres:King#123@localhost/latest_llm"
    
# Create the SQLAlchemy engine
engine = create_engine(SQLALCHEMY_DATABASE_URL)

# Create a metadata object
metadata = MetaData()

# Define your table
sessions_table = Table(
    'sessions',
    metadata,
    Column('id', Integer, primary_key=True),
    Column('email', String(255), unique=True),
    Column('password', String(255)),
    Column('username', String(255))
)

# Create all tables in the database
metadata.create_all(engine)

def establish_db_connection():
    """Establishes connection to the PostgreSQL database."""
    conn = engine.connect()
    return conn

def close_db_connection(conn):
    """Closes the database connection."""
    if conn:
        conn.close()