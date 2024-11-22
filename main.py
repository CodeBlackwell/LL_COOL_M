from fastapi import FastAPI, File, UploadFile
import pandas as pd
from database import get_connection
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from transformers import RobertaTokenizer
from data_processing import load_student_data, preprocess_data, tokenize_data
from model_training import initialize_model, train_model

app = FastAPI()

# Initialize the RoBERTa tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Load the data at startup
student_data_df = load_student_data('Student_performance_data _.csv')

# Preprocess the data
X_train, X_val, y_train, y_val = preprocess_data(student_data_df)

# Tokenize the training and validation data
X_train_tokens, y_train_tokens = tokenize_data(X_train, y_train)
X_val_tokens, y_val_tokens = tokenize_data(X_val, y_val)

# Initialize the model
model = initialize_model()

# Train the model
train_dataset = torch.utils.data.TensorDataset(X_train_tokens['input_ids'], y_train_tokens)
val_dataset = torch.utils.data.TensorDataset(X_val_tokens['input_ids'], y_val_tokens)

trained_model = train_model(model, train_dataset, val_dataset)

@app.get("/")
def read_root():
    return {"message": "School District Analytics Dashboard API is up and running!"}

@app.get("/create-table")
def create_table():
    create_table_query = """
    CREATE TABLE IF NOT EXISTS student_performance (
        student_id SERIAL PRIMARY KEY,
        age INTEGER,
        gender INTEGER,
        ethnicity INTEGER,
        parental_education INTEGER,
        study_time_weekly FLOAT,
        absences INTEGER,
        tutoring INTEGER,
        parental_support INTEGER,
        extracurricular INTEGER,
        sports INTEGER,
        music INTEGER,
        volunteering INTEGER,
        gpa FLOAT,
        grade_class FLOAT
    );
    """
    conn = get_connection()
    if conn:
        cursor = conn.cursor()
        try:
            cursor.execute(create_table_query)
            conn.commit()
            return {"message": "Student performance table created successfully"}
        except Exception as e:
            return {"error": str(e)}
        finally:
            cursor.close()
            conn.close()
    else:
        return {"error": "Connection to PostgreSQL failed"}

@app.get("/delete-table")
def delete_table():
    delete_table_query = """
    DROP TABLE IF EXISTS student_performance;
    """
    conn = get_connection()
    if conn:
        cursor = conn.cursor()
        try:
            cursor.execute(delete_table_query)
            conn.commit()
            return {"message": "Student performance table deleted successfully"}
        except Exception as e:
            return {"error": str(e)}
        finally:
            cursor.close()
            conn.close()
    else:
        return {"error": "Connection to PostgreSQL failed"}

@app.get("/student-data")
def get_student_data():
    # Return the student data as JSON for now
    return student_data_df.to_dict(orient='records')

@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...)):
    try:
        # Read the CSV file into a Pandas DataFrame
        df = pd.read_csv(file.file)

        # Optional: Validate columns
        expected_columns = [
            "gender", "race/ethnicity", "parental level of education", 
            "lunch", "test preparation course", "math score", 
            "reading score", "writing score"
        ]
        
        if not all(col in df.columns for col in expected_columns):
            return {"error": "Invalid CSV format. Please ensure all expected columns are present."}

        # Rename columns to match the table schema
        df.rename(columns={
            "race/ethnicity": "race_ethnicity",
            "parental level of education": "parental_education",
            "test preparation course": "test_prep",
            "math score": "math_score",
            "reading score": "reading_score",
            "writing score": "writing_score"
        }, inplace=True)

        # Connect to PostgreSQL and insert the data
        conn = get_connection()
        if conn:
            cursor = conn.cursor()
            try:
                for _, row in df.iterrows():
                    insert_query = """
                    INSERT INTO student_performance (
                        gender, race_ethnicity, parental_education, 
                        lunch, test_prep, math_score, 
                        reading_score, writing_score
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    cursor.execute(insert_query, (
                        row['gender'], row['race_ethnicity'], row['parental_education'],
                        row['lunch'], row['test_prep'], row['math_score'],
                        row['reading_score'], row['writing_score']
                    ))
                conn.commit()
                return {"message": "CSV data uploaded successfully"}
            except Exception as e:
                conn.rollback()
                return {"error": str(e)}
            finally:
                cursor.close()
                conn.close()
        else:
            return {"error": "Connection to PostgreSQL failed"}

    except Exception as e:
        return {"error": str(e)}

@app.post("/chat")
async def chat(prompt: str):
    # Query the OpenAI model with the provided prompt
    response = query_openai_model(prompt)
    
    # Return the response from the model
    return {"response": response}
