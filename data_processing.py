import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from transformers import RobertaTokenizer

# Function to load the CSV data
def load_student_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Function to preprocess the CSV data
def preprocess_data(df):
    # Select relevant features and target variable
    features = df.drop(columns=['StudentID', 'GPA'])
    target = df['GPA']
    
    # Normalize the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(features_scaled, target, test_size=0.2, random_state=42)
    
    return X_train, X_val, y_train, y_val

# Initialize the RoBERTa tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Function to tokenize the data
def tokenize_data(X, y):
    # Tokenize the input features
    inputs = tokenizer(X.tolist(), padding=True, truncation=True, return_tensors='pt')
    
    # Convert target to tensor
    labels = torch.tensor(y.tolist())
    
    return inputs, labels
