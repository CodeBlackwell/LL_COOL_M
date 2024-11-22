from fastapi import FastAPI, File, UploadFile
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW, get_scheduler
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
import os

app = FastAPI()

# Initialize the RoBERTa tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Define Dataset class
class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

def load_student_data(file_path):
    df = pd.read_csv(file_path)
    print("Columns in the dataset:", df.columns)  # Debugging step
    # Example preprocessing: Binary target based on GPA
    df["target"] = df["GPA"] > 3.0  # Set threshold as needed
    return df

def preprocess_data(df):
    # Selecting features and target
    X = df[["StudyTimeWeekly", "Absences", "Tutoring", "ParentalSupport", "Extracurricular"]]
    y = df["target"].astype(int)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val

def tokenize_data(X, y=None, max_length=128):
    # Convert numeric features to text strings
    X_text = X.apply(lambda row: " ".join(map(str, row)), axis=1)
    encodings = tokenizer(
        X_text.tolist(), truncation=True, padding=True, max_length=max_length, return_tensors="pt"
    )
    return CustomDataset(encodings, y.tolist()) if y is not None else encodings
# Initialize the model
def initialize_model(num_labels=2):
    return RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=num_labels)

# Fine-tune the model
def fine_tune_model(model, train_dataset, val_dataset, epochs=3, batch_size=16, lr=2e-5):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            inputs = {key: val.to(device) for key, val in batch.items()}
            labels = inputs.pop("labels")

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        lr_scheduler.step()
        print(f"Epoch {epoch + 1}: Loss = {total_loss / len(train_loader)}")

    # Validation loop
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            inputs = {key: val.to(device) for key, val in batch.items()}
            labels = inputs.pop("labels")

            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Validation Accuracy: {accuracy}")

    return model

# Load and preprocess data
student_data_df = load_student_data("./Student_performance_data _.csv")
X_train, X_val, y_train, y_val = preprocess_data(student_data_df)

# Tokenize data
train_dataset = tokenize_data(X_train, y_train)
val_dataset = tokenize_data(X_val, y_val)

# Initialize and fine-tune model
model = initialize_model()
fine_tuned_model = fine_tune_model(model, train_dataset, val_dataset, epochs=3)

# Save the model and tokenizer
model_dir = "./fine_tuned_model"
os.makedirs(model_dir, exist_ok=True)
fine_tuned_model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)

@app.get("/")
def read_root():
    return {"message": "School District Analytics Dashboard API is up and running!"}

@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        return {"message": f"Uploaded {len(df)} rows successfully"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict/")
async def predict(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = fine_tuned_model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=-1).item()
    return {"prediction": prediction}
