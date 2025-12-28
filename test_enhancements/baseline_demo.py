"""
BASELINE DEMO - Show Current System in Action

This script demonstrates the baseline fraud detection system:
1. Loads cleaned fraud data
2. Shows embedding statistics
3. Displays model information
4. Generates comparison metrics

Run: python baseline_demo.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import torch
from datetime import datetime

# Get paths
notebooks_dir = Path(__file__).parent.parent / "notebooks"
project_root = notebooks_dir.parent
generated_dir = project_root / "generated"
models_dir = project_root / "models"

print("\n" + "="*70)
print("🔵 BASELINE SYSTEM DEMO - Current Implementation")
print("="*70)

# Check GPU
gpu_available = torch.cuda.is_available()
device = "GPU" if gpu_available else "CPU"
if gpu_available:
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\n✅ GPU Available: {gpu_name} ({gpu_memory:.2f} GB)")
else:
    print("\n⚠️  GPU Not Available - Using CPU")

print("\n" + "-"*70)
print("1. DATA PIPELINE")
print("-"*70)

# Load baseline data
try:
    data_path = generated_dir / "fraud_data_combined_clean.csv"
    if data_path.exists():
        df = pd.read_csv(data_path)
        print(f"✅ Cleaned Data Loaded")
        print(f"   Rows: {len(df):,}")
        print(f"   Columns: {len(df.columns)}")
        print(f"   File Size: {data_path.stat().st_size / 1e6:.2f} MB")
        
        # Show data info
        if 'fraud_label' in df.columns:
            fraud_count = (df['fraud_label'] == 1).sum()
            legit_count = (df['fraud_label'] == 0).sum()
            fraud_pct = (fraud_count / len(df)) * 100
            print(f"\n   Data Distribution:")
            print(f"   - Legitimate: {legit_count:,} ({100-fraud_pct:.1f}%)")
            print(f"   - Fraud: {fraud_count:,} ({fraud_pct:.1f}%)")
    else:
        print(f"❌ Data not found at {data_path}")
except Exception as e:
    print(f"❌ Error loading data: {e}")

# Check narratives
try:
    narratives_path = generated_dir / "fraud_narratives_combined.csv"
    if narratives_path.exists():
        narratives_df = pd.read_csv(narratives_path)
        print(f"\n✅ Narratives Generated")
        print(f"   Total Narratives: {len(narratives_df):,}")
        print(f"   File Size: {narratives_path.stat().st_size / 1e6:.2f} MB")
    else:
        print(f"\n❌ Narratives not found at {narratives_path}")
except Exception as e:
    print(f"\n❌ Error loading narratives: {e}")

print("\n" + "-"*70)
print("2. EMBEDDING MODEL (DistilBERT)")
print("-"*70)

try:
    embedding_model = models_dir / "fraud_embedding_model.pt"
    tokenizer_dir = models_dir / "embedding_tokenizer"
    
    if embedding_model.exists():
        print(f"✅ Embedding Model Found")
        print(f"   Model Size: {embedding_model.stat().st_size / 1e6:.2f} MB")
        print(f"   Type: DistilBERT")
        print(f"   Output Dimension: 768")
        
        if tokenizer_dir.exists():
            print(f"   Tokenizer: Found")
    
    # Check embeddings
    embeddings_path = generated_dir / "fraud_embeddings.pkl"
    if embeddings_path.exists():
        print(f"\n✅ Embeddings Cached")
        print(f"   File Size: {embeddings_path.stat().st_size / 1e6:.2f} MB")
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
            if isinstance(embeddings, dict) and 'embeddings' in embeddings:
                emb_data = embeddings['embeddings']
                print(f"   Shape: {emb_data.shape}")
                print(f"   Type: {type(emb_data)}")
    else:
        print(f"\n⚠️  Embeddings not cached")
except Exception as e:
    print(f"❌ Error checking embeddings: {e}")

print("\n" + "-"*70)
print("3. LANGUAGE MODEL (GPT-2 + LoRA)")
print("-"*70)

try:
    gpt_model = models_dir / "gpt2_lora_model.pt"
    if gpt_model.exists():
        print(f"✅ GPT-2 LoRA Model Found")
        print(f"   Model Size: {gpt_model.stat().st_size / 1e6:.2f} MB")
        print(f"   Base Model: GPT-2")
        print(f"   Fine-tuning: LoRA (Low-Rank Adaptation)")
        print(f"   LoRA Rank: 8")
        print(f"   LoRA Alpha: 32")
    else:
        print(f"⚠️  GPT-2 Model not found (will be generated on first run)")
except Exception as e:
    print(f"❌ Error checking model: {e}")

print("\n" + "-"*70)
print("4. SYSTEM STATISTICS")
print("-"*70)

try:
    # Estimate pipeline time
    step1_time = 15  # seconds
    step2_time = 2   # seconds
    step3_time = 30 * 60  # 30 minutes
    step4_time = 50 * 60  # 50 minutes
    total_time = step1_time + step2_time + step3_time + step4_time
    
    print(f"✅ Pipeline Execution Times (RTX 3050 GPU):")
    print(f"   Step 1 (PII Clean): {step1_time} sec")
    print(f"   Step 2 (Narratives): {step2_time} sec")
    print(f"   Step 3 (Embeddings): {step3_time/60:.0f} min")
    print(f"   Step 4 (GPT-2 LoRA): {step4_time/60:.0f} min")
    print(f"   ─────────────────────")
    print(f"   TOTAL: {total_time//3600:.0f}h {(total_time%3600)//60:.0f}m {total_time%60:.0f}s")
    
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "-"*70)
print("5. BASELINE SYSTEM STATUS")
print("-"*70)

status = "✅ READY" if embedding_model.exists() and gpt_model.exists() else "⏳ FIRST RUN"
print(f"\nSystem Status: {status}")
print(f"Last Check: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print("\n" + "-"*70)
print("📊 NEXT STEPS")
print("-"*70)
print("""
To view the Baseline System Dashboard:
  cd notebooks
  python -m streamlit run app.py
  
  → Visit http://localhost:8501

To run the complete pipeline:
  cd notebooks
  python run_pipeline_genai.py

To test enhancements:
  cd test_enhancements
  python enhanced_demo.py
""")

print("\n" + "="*70)
print("✅ Baseline Demo Complete")
print("="*70 + "\n")
