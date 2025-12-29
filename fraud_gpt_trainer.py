"""
Step 4: GPT-2 Fine-tuning with LoRA
Fine-tunes GPT-2 model on fraud narratives using LoRA for efficient training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from peft import get_peft_model, LoraConfig, TaskType
import pandas as pd
import logging
from pathlib import Path
from typing import List


class FraudNarrativeGPT2Dataset(Dataset):
    """Dataset for GPT-2 narrative generation"""
    
    def __init__(self, narratives: List[str], tokenizer, max_length=256):
        self.narratives = narratives
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.narratives)
    
    def __getitem__(self, idx):
        narrative = self.narratives[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            narrative,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # For language modeling, labels are same as input_ids
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids
        }


class FraudGPT2Trainer:
    """Trains GPT-2 with LoRA for fraud narrative generation"""
    
    def __init__(self, model_name='gpt2', device=None):
        self.logger = logging.getLogger(__name__)
        
        # Device detection - Force CUDA on RTX 3050
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')  # Force GPU 0
                torch.cuda.set_device(0)
            else:
                raise RuntimeError("CUDA not available! RTX 3050 not detected. Please check GPU drivers.")
        else:
            self.device = device
        
        self.logger.info(f"Using device: {self.device}")
        
        # Print GPU info
        if self.device.type == 'cuda':
            self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"CUDA Version: {torch.version.cuda}")
            self.logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            # Enable memory optimization for 6GB VRAM RTX 3050
            torch.cuda.empty_cache()
            self.logger.info("GPU memory cache cleared for optimization")
        
        self.logger.info(f"PyTorch Version: {torch.__version__}")
        
        # Load base model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        base_model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=['c_attn']
        )
        
        # Apply LoRA
        self.model = get_peft_model(base_model, lora_config)
        self.model.to(self.device)
        
        self.logger.info("LoRA adapter applied to GPT-2")
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-5)
    
    def train_epoch(self, train_loader, epoch):
        """
        Train for one epoch
        
        Args:
            train_loader: DataLoader
            epoch: Epoch number
            
        Returns:
            Average loss
        """
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                self.logger.info(f"Epoch {epoch} [{batch_idx + 1}] Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def save_model(self, output_dir):
        """
        Save LoRA adapter and tokenizer
        
        Args:
            output_dir: Directory to save files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save LoRA adapter
        adapter_dir = output_path / 'fraud_pattern_generator_lora'
        self.model.save_pretrained(str(adapter_dir))
        self.logger.info(f"Saved LoRA adapter: {adapter_dir}")
        
        # Save tokenizer
        tokenizer_dir = output_path / 'gpt2_tokenizer'
        self.tokenizer.save_pretrained(str(tokenizer_dir))
        self.logger.info(f"Saved tokenizer: {tokenizer_dir}")


class GPT2Pipeline:
    """Orchestrates GPT-2 LoRA fine-tuning"""
    
    def __init__(self, data_dir='generated', model_dir='models'):
        # Resolve paths relative to parent directory (project root)
        current_dir = Path(__file__).parent
        self.data_dir = (current_dir.parent / data_dir).resolve()
        self.model_dir = (current_dir.parent / model_dir).resolve()
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Device setup - Force CUDA on RTX 3050
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            torch.cuda.set_device(0)
            torch.cuda.empty_cache()
        else:
            raise RuntimeError("CUDA not available! RTX 3050 not detected. Please check GPU drivers.")
    
    def run(self, input_file='fraud_narratives_combined.csv', epochs=3, batch_size=8):
        """
        Execute GPT-2 LoRA fine-tuning pipeline
        
        Args:
            input_file: Narratives CSV file
            epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            Path to saved model
        """
        try:
            # Load narratives
            input_path = self.data_dir / input_file
            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {input_path}")
            
            self.logger.info(f"Loading narratives from {input_path}")
            df = pd.read_csv(input_path)
            
            narratives = df['narrative'].tolist()
            
            self.logger.info(f"Loaded {len(narratives)} narratives")
            
            # Initialize trainer
            trainer = FraudGPT2Trainer(device=self.device)
            
            # Create dataset and dataloader
            dataset = FraudNarrativeGPT2Dataset(narratives, trainer.tokenizer)
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Train model
            self.logger.info(f"Training GPT-2 LoRA for {epochs} epochs with batch_size={batch_size}")
            for epoch in range(1, epochs + 1):
                avg_loss = trainer.train_epoch(train_loader, epoch)
                self.logger.info(f"Epoch {epoch} completed - Avg Loss: {avg_loss:.4f}")
            
            # Save model
            trainer.save_model(self.model_dir)
            
            self.logger.info("GPT-2 LoRA fine-tuning completed")
            return str(self.model_dir / 'fraud_pattern_generator_lora')
            
        except Exception as e:
            self.logger.error(f"Error in GPT-2 pipeline: {e}")
            raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    pipeline = GPT2Pipeline()
    # This will run after Step 3 is complete
    output_path = pipeline.run()
    print(f"GPT-2 LoRA model saved: {output_path}")