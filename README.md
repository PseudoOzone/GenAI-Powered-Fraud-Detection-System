# GenAI-Powered Fraud Detection System

## Overview

A 4-step fraud detection pipeline using Generative AI (LLM-based) and transformer models. This system processes transaction data through GPU-optimized modules to detect fraudulent patterns using semantic embeddings and fine-tuned language models.

## Project Structure

```
/data                       → Original datasets (Base.csv, Variant I-V.csv)
/generated                  → Processed outputs (cleaned data, narratives, embeddings)
/models                     → Trained models (embeddings, LoRA adapter, tokenizers)
/notebooks                  → Python scripts and pipeline modules
/security                   → PII detection and cleaning modules
/uploads                    → User-uploaded CSV files (Streamlit UI)
/logs                       → Training logs and execution traces
requirements.txt            → Python package dependencies
```

## Technology Stack

- **Python 3.13**
- **PyTorch 2.0+** with CUDA support (GPU-optimized)
- **Transformers (HuggingFace)** - DistilBERT, GPT-2
- **PEFT** - LoRA fine-tuning
- **Streamlit** - Web UI
- **pandas, numpy, scikit-learn** - Data utilities

## 4-Step Pipeline Architecture

### Step 1: PII Cleaning
**Module:** `pii_cleaner.py`

- Loads all 6 datasets (Base + Variant I-V)
- Removes/masks PII (emails, phone, SSN, names)
- Combines into single clean dataset
- **Output:** `fraud_data_combined_clean.csv`

### Step 2: Fraud Narrative Generation
**Module:** `genai_narrative_generator.py`

- Converts transaction data → fraud story descriptions
- Generates both FRAUD and LEGITIMATE narratives
- Augments narratives with fraud patterns and risk indicators
- **Output:** `fraud_narratives_combined.csv`

**Example Narrative:**
```
"Transaction of $5000.50 at TechMart Electronics (Electronics) in New York. 
High-value transaction. Premium cardholder."
```

### Step 3: Fraud Embedding Model
**Module:** `genai_embedding_model.py`

- Trains DistilBERT model on fraud narratives
- GPU-optimized with automatic device detection
- Generates semantic embeddings for fraud patterns
- **Output:** 
  - `fraud_embeddings.pkl` (embeddings + labels)
  - `fraud_embedding_model.pt` (model weights)
  - `embedding_tokenizer/` (tokenizer files)

### Step 4: GPT-2 LoRA Fine-tuning
**Module:** `fraud_gpt_trainer.py`

- Fine-tunes GPT-2 with LoRA adapter
- Learns to generate fraud narratives
- Efficient parameter tuning with LoRA
- **Output:**
  - `fraud_pattern_generator_lora/` (LoRA weights)
  - `gpt2_tokenizer/` (tokenizer files)

## Installation & Setup

### 1. Create Virtual Environment
```bash
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify GPU Setup (Optional)
```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

## Usage

### Run Complete Pipeline

```bash
cd notebooks
python run_pipeline_genai.py
```

**What happens:**
1. Detects GPU and logs system info
2. Runs all 4 steps sequentially
3. Creates logs with timestamped output
4. Saves all models and data
5. Reports success/failure with exit code

**Output:**
- Log file: `logs/genai_pipeline_YYYYMMDD_HHMMSS.log`
- Generated data: `generated/fraud_*.csv`, `generated/*.pkl`
- Models: `models/fraud_embedding_model.pt`, `models/fraud_pattern_generator_lora/`

### Run Individual Steps

```bash
# Step 1: PII Cleaning
python -c "from pii_cleaner import PIICleaner; PIICleaner().run()"

# Step 2: Narrative Generation
python -c "from genai_narrative_generator import NarrativeGeneratorPipeline; \
    NarrativeGeneratorPipeline().run()"

# Step 3: Embedding Model
python -c "from genai_embedding_model import EmbeddingPipeline; \
    EmbeddingPipeline().run()"

# Step 4: GPT-2 LoRA
python -c "from fraud_gpt_trainer import GPT2Pipeline; \
    GPT2Pipeline().run()"
```

### Launch Streamlit UI

```bash
streamlit run notebooks/ui.py
```

Opens at `http://localhost:8501`

**Features:**
- Dashboard with model status
- Single transaction analysis
- Batch CSV upload and analysis
- Model information and system stats

## GPU Optimization

The system automatically detects and uses GPU if available:

- **Device Detection:** Checks for CUDA availability
- **Memory Management:** Batch processing to reduce footprint
- **Logging:** Prints GPU name, CUDA version, VRAM on startup
- **Fallback:** Uses CPU if GPU unavailable (with logging)

**GPU Info Logged:**
```
GPU: NVIDIA A100
CUDA Version: 12.1
GPU Memory: 80.00 GB
```

## Output Files

### Data Files (`/generated`)
- `fraud_data_combined_clean.csv` - Step 1 output (PII masked)
- `fraud_narratives_combined.csv` - Step 2 output (narratives)
- `fraud_embeddings.pkl` - Step 3 output (embeddings)
- `[dataset_name]_clean.csv` - Individual cleaned datasets

### Model Files (`/models`)
- `fraud_embedding_model.pt` - DistilBERT weights
- `embedding_tokenizer/` - DistilBERT tokenizer
- `fraud_pattern_generator_lora/` - GPT-2 LoRA adapter
- `gpt2_tokenizer/` - GPT-2 tokenizer

### Logs (`/logs`)
- `genai_pipeline_YYYYMMDD_HHMMSS.log` - Timestamped execution log

## Error Handling

The pipeline gracefully handles:
- **File Not Found:** Logs clear message and skips
- **Out of Memory:** Automatically reduces batch size
- **Model Loading Fails:** Logs checkpoint path and exits
- **GPU Memory Error:** Falls back to CPU with warning
- **CSV Parsing Error:** Logs row index and continues

## Testing Checklist

- ✓ All imports work
- ✓ GPU detected and used
- ✓ All 4 steps run without errors
- ✓ Generated files exist in correct locations
- ✓ Models saved with correct names
- ✓ Logs written to `/logs/`
- ✓ Streamlit UI runs
- ✓ CSV upload/prediction works

## Key Features

- **Generative AI Based:** Uses LLMs instead of traditional ML
- **GPU-Optimized:** Full CUDA support with device detection
- **Narrative-Driven:** Converts data to interpretable stories
- **LoRA Fine-tuning:** Efficient parameter tuning with PEFT
- **Modular Design:** Each step can run independently
- **Comprehensive Logging:** Timestamped logs with detailed execution info
- **Web UI:** Streamlit interface for interactive analysis
- **Batch Processing:** Handle multiple datasets simultaneously

## Configuration

Edit pipeline parameters in individual module `run()` methods:

```python
# Sample size for narrative generation
pipeline.run(sample_size=5000)

# Training epochs and batch size
pipeline.run(epochs=3, batch_size=16)
```

## Troubleshooting

### CUDA Not Available
```bash
# Check PyTorch installation with CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Out of Memory (GPU)
Reduce batch size in pipeline calls:
```python
pipeline.run(batch_size=8)  # Smaller batches
```

### Module Import Errors
Ensure working directory is `/notebooks`:
```bash
cd notebooks
python run_pipeline_genai.py
```

## Support & Documentation

- **DistilBERT:** https://huggingface.co/distilbert-base-uncased
- **GPT-2:** https://huggingface.co/gpt2
- **PEFT (LoRA):** https://github.com/huggingface/peft
- **PyTorch:** https://pytorch.org/
- **Streamlit:** https://streamlit.io/

## License

MIT License - See LICENSE file for details
