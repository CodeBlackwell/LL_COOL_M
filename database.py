import psycopg2
from psycopg2 import sql

# Replace these with your PostgreSQL credentials
DB_HOST = "localhost"
DB_NAME = "student_performance_db"
DB_USER = "user_name"
DB_PASSWORD = "password"
DB_PORT = 5432

# Function to create a connection
def get_connection():
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        return conn
    except Exception as e:
        print("Unable to connect to PostgreSQL")
        print(e)
        return None
