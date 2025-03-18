#!/usr/bin/env python
import argparse
import csv
import logging
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Set seed for reproducibility
def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

set_seed()

class MarchMadnessDataset(Dataset):
    """Dataset for March Madness matchup prediction."""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def load_historical_data(file_path):
    """Load historical March Madness data from CSV file"""
    logger.info(f"Loading historical data from {file_path}")
    
    # If the file doesn't exist, we'll create a sample dataset for demonstration
    if not os.path.exists(file_path):
        logger.warning(f"File {file_path} not found. Creating sample data...")
        return create_sample_data()
    
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} historical games")
    return df

def create_sample_data():
    """Create a sample dataset for demonstration purposes"""
    logger.info("Creating sample historical tournament data")
    
    # Sample teams with different performance metrics
    teams = {
        "Duke": {"win_rate": 0.85, "ppg": 82.3, "def_rating": 90.1, "sos": 10.2},
        "Kentucky": {"win_rate": 0.82, "ppg": 79.5, "def_rating": 91.2, "sos": 9.8},
        "Kansas": {"win_rate": 0.80, "ppg": 78.2, "def_rating": 93.5, "sos": 9.5},
        "UConn": {"win_rate": 0.79, "ppg": 77.8, "def_rating": 92.3, "sos": 8.9},
        "UNC": {"win_rate": 0.78, "ppg": 80.1, "def_rating": 94.2, "sos": 9.2},
        "Gonzaga": {"win_rate": 0.90, "ppg": 84.5, "def_rating": 89.8, "sos": 7.5},
        "Michigan": {"win_rate": 0.76, "ppg": 75.3, "def_rating": 95.1, "sos": 8.5},
        "Villanova": {"win_rate": 0.83, "ppg": 77.2, "def_rating": 91.5, "sos": 9.1},
        "Auburn": {"win_rate": 0.77, "ppg": 76.8, "def_rating": 93.8, "sos": 8.7},
        "Houston": {"win_rate": 0.81, "ppg": 73.5, "def_rating": 82.0, "sos": 7.8}
    }
    
    # Generate 200 sample games with realistic outcomes
    records = []
    seeds = {team: i % 16 + 1 for i, team in enumerate(teams.keys())}
    
    for _ in range(200):
        team1, team2 = random.sample(list(teams.keys()), 2)
        
        # Create feature description for the matchup
        team1_stats = teams[team1]
        team2_stats = teams[team2]
        
        # Determine the winner based on team stats (simplified for example)
        team1_score = (
            team1_stats["win_rate"] * 40 + 
            team1_stats["ppg"] / 100 - 
            team2_stats["def_rating"] / 100 + 
            team1_stats["sos"] / 20 +
            random.uniform(-0.1, 0.1)  # Add noise
        )
        
        team2_score = (
            team2_stats["win_rate"] * 40 + 
            team2_stats["ppg"] / 100 - 
            team1_stats["def_rating"] / 100 + 
            team2_stats["sos"] / 20 +
            random.uniform(-0.1, 0.1)  # Add noise
        )
        
        team1_win = team1_score > team2_score
        
        # Add to records
        records.append({
            "year": random.randint(2010, 2023),
            "round": random.choice(["First Round", "Second Round", "Sweet 16", "Elite 8"]),
            "region": random.choice(["East", "West", "South", "Midwest"]),
            "team1": team1,
            "team2": team2,
            "seed1": seeds[team1],
            "seed2": seeds[team2],
            "winner": team1 if team1_win else team2,
            "team1_win": team1_win
        })
    
    return pd.DataFrame(records)

def prepare_data_for_finetuning(df):
    """Prepare data for fine-tuning"""
    logger.info("Preparing data for fine-tuning")
    
    texts = []
    labels = []
    
    for _, row in df.iterrows():
        seed1 = row['seed1']
        seed2 = row['seed2']
        team1 = row['team1']
        team2 = row['team2']
        
        # Create a textual description of the matchup
        text = f"#{seed1} {team1} vs #{seed2} {team2} in the {row['round']} of the {row['year']} tournament in the {row['region']} region."
        
        # Label is 1 if team1 wins, 0 if team2 wins
        label = 1 if row['team1_win'] else 0
        
        texts.append(text)
        labels.append(label)
    
    # Split data into train and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    logger.info(f"Train set: {len(train_texts)} examples")
    logger.info(f"Validation set: {len(val_texts)} examples")
    
    return train_texts, val_texts, train_labels, val_labels

def fine_tune_model(train_dataloader, val_dataloader, model, tokenizer, device, epochs=4, model_save_path="fine_tuned_model"):
    """Fine-tune the model on March Madness data"""
    logger.info(f"Fine-tuning model for {epochs} epochs on {device}")
    
    # Set up optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    # Set up learning rate scheduler
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Track best accuracy
    best_accuracy = 0.0
    
    # Training loop
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")
        
        # Training
        model.train()
        train_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc="Training")
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            train_loss += loss.item()
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            scheduler.step()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_dataloader)
        logger.info(f"Average training loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        predictions = []
        true_labels = []
        
        progress_bar = tqdm(val_dataloader, desc="Validation")
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass (no gradients)
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
            
            loss = outputs.loss
            val_loss += loss.item()
            
            # Get predictions
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            # Add to lists
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_dataloader)
        
        # Calculate accuracy
        accuracy = (np.array(predictions) == np.array(true_labels)).mean()
        
        logger.info(f"Validation Loss: {avg_val_loss:.4f}")
        logger.info(f"Validation Accuracy: {accuracy:.4f}")
        
        # Print classification report
        report = classification_report(true_labels, predictions)
        logger.info(f"\n{report}")
        
        # Save model if it's the best so far
        if accuracy > best_accuracy:
            logger.info(f"Accuracy improved from {best_accuracy:.4f} to {accuracy:.4f}. Saving model...")
            best_accuracy = accuracy
            
            # Save tokenizer and model
            tokenizer.save_pretrained(model_save_path)
            model.save_pretrained(model_save_path)

def main():
    parser = argparse.ArgumentParser(description="Fine-tune BERT for March Madness predictions")
    parser.add_argument("--data_file", type=str, default="historical_march_madness.csv",
                      help="Path to historical March Madness data CSV file")
    parser.add_argument("--epochs", type=int, default=4,
                      help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                      help="Training batch size")
    parser.add_argument("--max_length", type=int, default=512,
                      help="Maximum sequence length for tokenization")
    parser.add_argument("--model_save_path", type=str, default="fine_tuned_model",
                      help="Path to save the fine-tuned model")
    args = parser.parse_args()
    
    # Load historical data
    df = load_historical_data(args.data_file)
    
    # Prepare data for fine-tuning
    train_texts, val_texts, train_labels, val_labels = prepare_data_for_finetuning(df)
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModelForSequenceClassification.from_pretrained(
        'bert-base-uncased', 
        num_labels=2  # Binary classification: team1 wins (1) or team2 wins (0)
    )
    
    # Create datasets and dataloaders
    train_dataset = MarchMadnessDataset(
        train_texts, train_labels, tokenizer, max_length=args.max_length
    )
    val_dataset = MarchMadnessDataset(
        val_texts, val_labels, tokenizer, max_length=args.max_length
    )
    
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size
    )
    
    # Determine device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model.to(device)
    
    # Fine-tune the model
    fine_tune_model(
        train_dataloader, 
        val_dataloader, 
        model,
        tokenizer,
        device, 
        epochs=args.epochs,
        model_save_path=args.model_save_path
    )
    
    logger.info("Fine-tuning complete!")

if __name__ == "__main__":
    main() 