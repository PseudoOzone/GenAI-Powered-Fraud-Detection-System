"""
Streamlit UI for GenAI Fraud Detection System
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import pickle
from transformers import DistilBertTokenizer, GPT2Tokenizer, GPT2LMHeadModel
from peft import PeftModel
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from genai_embedding_model import FraudEmbeddingModel


class FraudDetectionUI:
    """Streamlit UI for fraud detection"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding_model = None
        self.embedding_tokenizer = None
        self.gpt2_model = None
        self.gpt2_tokenizer = None
        self.embeddings_data = None
        
        self.load_models()
    
    def load_models(self):
        """Load trained models"""
        # Resolve paths relative to parent directory (project root)
        current_dir = Path(__file__).parent
        models_dir = (current_dir.parent / 'models').resolve()
        generated_dir = (current_dir.parent / 'generated').resolve()
        
        # Load embedding model
        try:
            self.embedding_tokenizer = DistilBertTokenizer.from_pretrained(
                str(models_dir / 'embedding_tokenizer')
            )
            
            embedding_model = FraudEmbeddingModel()
            state_dict = torch.load(
                str(models_dir / 'fraud_embedding_model.pt'),
                map_location=self.device
            )
            embedding_model.load_state_dict(state_dict)
            self.embedding_model = embedding_model.to(self.device)
            self.embedding_model.eval()
            
        except Exception as e:
            st.error(f"Error loading embedding model: {e}")
        
        # Load GPT-2 with LoRA
        try:
            self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained(
                str(models_dir / 'gpt2_tokenizer')
            )
            
            base_model = GPT2LMHeadModel.from_pretrained('gpt2')
            self.gpt2_model = PeftModel.from_pretrained(
                base_model,
                str(models_dir / 'fraud_pattern_generator_lora')
            )
            self.gpt2_model.to(self.device)
            self.gpt2_model.eval()
            
        except Exception as e:
            st.warning(f"GPT-2 LoRA model not available: {e}")
        
        # Load embeddings data
        try:
            embeddings_file = generated_dir / 'fraud_embeddings.pkl'
            if embeddings_file.exists():
                with open(embeddings_file, 'rb') as f:
                    self.embeddings_data = pickle.load(f)
        except Exception as e:
            st.warning(f"Embeddings data not available: {e}")
    
    def detect_fraud(self, narrative):
        """Detect fraud probability for a narrative"""
        if self.embedding_model is None:
            return None, None
        
        # Tokenize
        encodings = self.embedding_tokenizer(
            narrative,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        
        with torch.no_grad():
            logits, _ = self.embedding_model(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=1)
            fraud_prob = probabilities[0, 1].item()
            
        return fraud_prob, logits
    
    def generate_narrative(self, seed_text, max_length=100):
        """Generate fraud narrative using GPT-2"""
        if self.gpt2_model is None:
            return "GPT-2 model not available"
        
        input_ids = self.gpt2_tokenizer.encode(seed_text, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            output = self.gpt2_model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9
            )
        
        narrative = self.gpt2_tokenizer.decode(output[0], skip_special_tokens=True)
        return narrative
    
    def run(self):
        """Main Streamlit app"""
        st.set_page_config(
            page_title="GenAI Fraud Detection",
            page_icon="🔍",
            layout="wide"
        )
        
        st.title("🔍 GenAI-Powered Fraud Detection System")
        st.markdown("Detecting financial fraud using transformer embeddings and LLM analysis")
        
        # Sidebar
        st.sidebar.header("Navigation")
        page = st.sidebar.radio(
            "Select Page",
            ["Dashboard", "Single Transaction", "Batch Analysis", "Model Info"]
        )
        
        if page == "Dashboard":
            self.show_dashboard()
        elif page == "Single Transaction":
            self.show_single_transaction()
        elif page == "Batch Analysis":
            self.show_batch_analysis()
        elif page == "Model Info":
            self.show_model_info()
    
    def show_dashboard(self):
        """Dashboard page"""
        st.header("Dashboard")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Embedding Model", "DistilBERT" if self.embedding_model else "Not Loaded")
        
        with col2:
            st.metric("LLM Model", "GPT-2 LoRA" if self.gpt2_model else "Not Loaded")
        
        with col3:
            st.metric("Device", self.device.type.upper())
        
        st.divider()
        
        if self.embeddings_data:
            embeddings = self.embeddings_data['embeddings']
            labels = self.embeddings_data['labels']
            
            st.subheader("Training Data Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Narratives", len(labels))
            
            with col2:
                fraud_count = sum(labels)
                st.metric("Fraud Cases", fraud_count)
            
            with col3:
                legitimate_count = len(labels) - fraud_count
                st.metric("Legitimate Cases", legitimate_count)
    
    def show_single_transaction(self):
        """Single transaction analysis page"""
        st.header("Single Transaction Analysis")
        
        st.subheader("Enter Transaction Details")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            amount = st.number_input("Amount ($)", min_value=0.0, value=1000.0)
        
        with col2:
            merchant = st.text_input("Merchant Name", "ElectroStore")
        
        with col3:
            category = st.selectbox(
                "Category",
                ["Electronics", "Jewelry", "Travel", "Clothing", "Food & Drink", "Other"]
            )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            location = st.text_input("Location", "New York")
        
        with col2:
            age = st.number_input("Customer Age", min_value=18, max_value=100, value=35)
        
        with col3:
            card_type = st.selectbox("Card Type", ["Standard", "Credit", "Debit", "Platinum"])
        
        if st.button("Analyze Transaction", type="primary"):
            # Generate narrative
            narrative = f"Transaction of ${amount:.2f} at {merchant} ({category}) in {location}. "
            narrative += f"{'High-value transaction. ' if amount > 5000 else ''}"
            narrative += f"{'Premium cardholder. ' if card_type == 'Platinum' else ''}"
            narrative += f"{'Elderly customer. ' if age > 65 else ''}"
            narrative += f"{'Young customer. ' if age < 25 else ''}"
            
            st.info(f"**Narrative:** {narrative}")
            
            # Detect fraud
            fraud_prob, _ = self.detect_fraud(narrative)
            
            if fraud_prob is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    fraud_percent = fraud_prob * 100
                    risk_level = "🔴 High Risk" if fraud_percent > 70 else "🟡 Medium Risk" if fraud_percent > 40 else "🟢 Low Risk"
                    
                    st.metric("Fraud Probability", f"{fraud_percent:.2f}%")
                    st.write(f"**Risk Level:** {risk_level}")
                
                with col2:
                    st.progress(fraud_prob)
                    st.write(f"Confidence Score: {fraud_prob:.4f}")
            
            # Generate explanation narrative
            if self.gpt2_model:
                st.subheader("Generated Explanation")
                seed = f"Transaction analysis: {narrative}"
                explanation = self.generate_narrative(seed, max_length=100)
                st.write(explanation)
    
    def show_batch_analysis(self):
        """Batch analysis page"""
        st.header("Batch Transaction Analysis")
        
        uploaded_file = st.file_uploader("Upload CSV with transactions", type="csv")
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(df)} transactions")
            
            if st.button("Analyze Batch", type="primary"):
                progress_bar = st.progress(0)
                results = []
                
                for idx, row in df.iterrows():
                    # Generate narrative
                    narrative = f"Transaction of ${row.get('amount', 0):.2f} at "
                    narrative += f"{row.get('merchant_name', 'unknown')} "
                    narrative += f"in {row.get('location', 'unknown')}"
                    
                    # Detect fraud
                    fraud_prob, _ = self.detect_fraud(narrative)
                    
                    results.append({
                        'transaction_id': row.get('transaction_id', idx),
                        'amount': row.get('amount', 0),
                        'merchant': row.get('merchant_name', 'unknown'),
                        'fraud_probability': fraud_prob if fraud_prob else 0,
                        'risk_level': "High" if fraud_prob > 0.7 else "Medium" if fraud_prob > 0.4 else "Low"
                    })
                    
                    progress_bar.progress((idx + 1) / len(df))
                
                results_df = pd.DataFrame(results)
                st.dataframe(results_df)
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Results",
                    data=csv,
                    file_name="fraud_analysis_results.csv",
                    mime="text/csv"
                )
    
    def show_model_info(self):
        """Model information page"""
        st.header("Model Information")
        
        st.subheader("Embedding Model")
        if self.embedding_model:
            st.write("✓ **DistilBERT** loaded successfully")
            st.write("- Architecture: Transformer-based sequence classification")
            st.write("- Input: Fraud narratives (max 256 tokens)")
            st.write("- Output: Binary classification (Fraud/Legitimate)")
        else:
            st.write("✗ Embedding model not loaded")
        
        st.divider()
        
        st.subheader("LLM Model")
        if self.gpt2_model:
            st.write("✓ **GPT-2 with LoRA** loaded successfully")
            st.write("- Base Model: GPT-2")
            st.write("- Fine-tuning: LoRA (Low-Rank Adaptation)")
            st.write("- Task: Fraud narrative generation")
        else:
            st.write("✗ GPT-2 model not loaded")
        
        st.divider()
        
        st.subheader("System Info")
        st.write(f"**Device:** {self.device}")
        
        if torch.cuda.is_available():
            st.write(f"**GPU:** {torch.cuda.get_device_name(0)}")
            st.write(f"**CUDA Version:** {torch.version.cuda}")
            st.write(f"**GPU Memory:** {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            st.write("**GPU:** Not available (using CPU)")


if __name__ == "__main__":
    ui = FraudDetectionUI()
    ui.run()
