import torch
from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments

# Function to initialize the RoBERTa model
def initialize_model():
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=1)
    return model

# Function to train the model
def train_model(model, train_dataset, val_dataset):
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    trainer.train()
    return model
