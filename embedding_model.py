import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertModel
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import pickle

class FraudNarrativeDataset(Dataset):
    """A dataset class for fraud narratives."""
    def __init__(self, narratives, labels, tokenizer, max_length=256):
        self.narratives = narratives
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.narratives)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.narratives[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class GenAIEmbeddingModel(nn.Module):
    """A DistilBERT-based model for generating embeddings and classifying fraud."""
    def __init__(self, model_name='distilbert-base-uncased', num_classes=2):
        super(GenAIEmbeddingModel, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_embedding)
        logits = self.classifier(x)
        return logits, cls_embedding

class EmbeddingTrainer:
    """A class to train the embedding model."""
    def __init__(self, model, tokenizer, device, logger):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.logger = logger
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=2e-5)

    def train_epoch(self, loader):
        self.model.train()
        for batch in loader:
            self.optimizer.zero_grad()
            logits, _ = self.model(
                batch['input_ids'].to(self.device),
                batch['attention_mask'].to(self.device)
            )
            loss = self.criterion(logits, batch['label'].to(self.device))
            loss.backward()
            self.optimizer.step()
        self.logger.info("Training epoch completed.")

    def get_embeddings(self, narratives, batch_size=32):
        self.model.eval()
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(narratives), batch_size):
                batch_narratives = narratives[i:i + batch_size]
                encodings = self.tokenizer(
                    batch_narratives, max_length=256, padding='max_length',
                    truncation=True, return_tensors='pt'
                )
                _, embedding_batch = self.model(
                    encodings['input_ids'].to(self.device),
                    encodings['attention_mask'].to(self.device)
                )
                embeddings.append(embedding_batch.cpu().numpy())
        return np.vstack(embeddings)

def run_embedding_pipeline(data_dir, model_dir, device):
    """The main function to run the embedding pipeline."""
    logger = logging.getLogger(__name__)
    data_path = data_dir / 'fraud_narratives_combined.csv'
    df = pd.read_csv(data_path)
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    dataset = FraudNarrativeDataset(
        df['narrative'].tolist(), df['fraud_label'].tolist(), tokenizer
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = GenAIEmbeddingModel().to(device)
    trainer = EmbeddingTrainer(model, tokenizer, device, logger)
    
    logger.info("Starting model training...")
    for epoch in range(3):
        trainer.train_epoch(loader)
        logger.info(f"Epoch {epoch + 1} completed.")
    
    embeddings = trainer.get_embeddings(df['narrative'].tolist())
    
    output_path = data_dir / 'fraud_embeddings.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump({'embeddings': embeddings, 'labels': df['fraud_label'].tolist()}, f)
    
    logger.info(f"Embeddings saved to {output_path}")
    torch.save(model.state_dict(), model_dir / 'fraud_embedding_model.pt')
    logger.info("Model saved.")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    current_dir = Path(__file__).parent
    data_dir = (current_dir.parent / 'generated').resolve()
    model_dir = (current_dir.parent / 'models').resolve()
    model_dir.mkdir(exist_ok=True)
    
    run_embedding_pipeline(data_dir, model_dir, device)
