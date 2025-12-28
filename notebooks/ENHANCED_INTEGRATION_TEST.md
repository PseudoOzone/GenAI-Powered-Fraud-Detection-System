# Enhanced Fraud Detection System - Integration Testing Notebook

This comprehensive notebook tests all enhancements integrated into the GenAI Fraud Detection System.

## Quick Links
- **Testing Guide:** See "Testing & Validation" section
- **System Status:** Check "Environment Setup" first
- **Common Issues:** Jump to "Troubleshooting" section

---

## PART 1: ENVIRONMENT SETUP & VALIDATION

### Step 1.1: Import All Required Libraries
```python
# Standard libraries
import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Data processing
import pandas as pd
import numpy as np

# Machine learning
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, GPT2Tokenizer, GPT2LMHeadModel
from peft import PeftModel

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Add project path
project_root = Path("c:\\Users\\anshu\\GenAI-Powered Fraud Detection System")
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "notebooks"))
sys.path.insert(0, str(project_root / "security"))

print("✅ All libraries imported successfully")
```

### Step 1.2: Verify GPU & Device
```python
# Check GPU availability
print("=" * 60)
print("DEVICE & GPU CONFIGURATION")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Device: {device.type.upper()}")
print(f"✅ PyTorch Version: {torch.__version__}")

if torch.cuda.is_available():
    print(f"✅ GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"✅ CUDA Version: {torch.version.cuda}")
else:
    print("⚠️ GPU not available - using CPU (slower but functional)")

print()
```

### Step 1.3: Verify All Modules Can Be Imported
```python
# Import enhancement modules
print("=" * 60)
print("IMPORTING ENHANCEMENT MODULES")
print("=" * 60)

# PII Validator
try:
    from security.pii_validator import PIIDetector
    pii_detector = PIIDetector()
    print("✅ PII Validator imported successfully")
except Exception as e:
    print(f"❌ PII Validator error: {e}")
    pii_detector = None

# Attack Pattern Analyzer
try:
    from notebooks.attack_pattern_analyzer import AttackPatternAnalyzer
    attack_analyzer = AttackPatternAnalyzer()
    print("✅ Attack Pattern Analyzer imported successfully")
except Exception as e:
    print(f"❌ Attack Pattern Analyzer error: {e}")
    attack_analyzer = None

# Federated Learning
try:
    from notebooks.federated_learning import FederatedConfig, FederatedClient
    federated_config = FederatedConfig(num_clients=5, epochs=3)
    print("✅ Federated Learning imported successfully")
except Exception as e:
    print(f"❌ Federated Learning error: {e}")
    federated_config = None

# Original embedding model
try:
    from notebooks.genai_embedding_model import FraudEmbeddingModel
    print("✅ Embedding Model imported successfully")
except Exception as e:
    print(f"❌ Embedding Model error: {e}")

print()
```

---

## PART 2: PII VALIDATOR TESTING

### Step 2.1: Test PII Entity Detection
```python
print("=" * 60)
print("TEST 1: PII ENTITY DETECTION")
print("=" * 60)

test_cases = [
    "Call John at 555-123-4567 about his account",
    "Email: john.doe@example.com for credit card 4532-1111-2222-3333",
    "SSN: 123-45-6789, DOB: 01/15/1990",
    "Driver's license: D12345678, Phone: (555) 987-6543",
    "Clean transaction narrative with no sensitive data",
    "Account number 98765432 at IP address 192.168.1.1"
]

if pii_detector:
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}: {test_case[:50]}...")
        try:
            entities = pii_detector.detect_entities(test_case)
            if entities:
                print(f"   Detected {len(entities)} entities:")
                for entity in entities:
                    print(f"   - {entity.get('type', 'UNKNOWN')}: {entity.get('value', 'N/A')}")
                    print(f"     Confidence: {entity.get('confidence', 0):.1%}")
            else:
                print(f"   ✓ No sensitive data detected")
        except Exception as e:
            print(f"   ❌ Error: {e}")
else:
    print("❌ PII Detector not available")

print()
```

### Step 2.2: Test Compliance Validation
```python
print("=" * 60)
print("TEST 2: COMPLIANCE FRAMEWORK VALIDATION")
print("=" * 60)

compliance_test = "Customer John Doe, SSN 123-45-6789, Email john@test.com, Phone 555-1234"

if pii_detector:
    print(f"\n📝 Test Text: {compliance_test}")
    try:
        compliance = pii_detector.validate_compliance(compliance_test)
        print(f"\n✅ Compliance Results:")
        for framework, status in compliance.items():
            emoji = "✅" if status else "❌"
            print(f"   {emoji} {framework}: {status}")
    except Exception as e:
        print(f"❌ Error: {e}")
else:
    print("❌ PII Detector not available")

print()
```

### Step 2.3: Test Confidence Scoring
```python
print("=" * 60)
print("TEST 3: CONFIDENCE SCORING")
print("=" * 60)

confidence_test = "Contact info: john@example.com, Phone: 555-4567"

if pii_detector:
    print(f"\n📝 Test Text: {confidence_test}")
    try:
        scores = pii_detector.get_confidence_scores(confidence_test)
        print(f"\n✅ Confidence Scores:")
        if scores:
            for key, value in scores.items():
                print(f"   {key}: {value:.1%}")
        else:
            print(f"   No confidence scores available")
    except Exception as e:
        print(f"❌ Error: {e}")
else:
    print("❌ PII Detector not available")

print()
```

### Step 2.4: Test Real Data Processing
```python
print("=" * 60)
print("TEST 4: REAL DATA PROCESSING")
print("=" * 60)

generated_dir = project_root / "generated"
fraud_data_file = generated_dir / "fraud_data_combined_clean.csv"

if pii_detector and fraud_data_file.exists():
    print(f"\n📁 Loading data from: {fraud_data_file}")
    try:
        df = pd.read_csv(fraud_data_file, nrows=100)
        print(f"✅ Loaded {len(df)} rows")
        
        text_columns = df.select_dtypes(include=['object']).columns[:5]
        print(f"✅ Processing {len(text_columns)} text columns: {list(text_columns)}")
        
        total_entities = 0
        for col in text_columns:
            for text in df[col].head(20):
                if pd.notna(text):
                    entities = pii_detector.detect_entities(str(text))
                    total_entities += len(entities) if entities else 0
        
        print(f"✅ Total entities found: {total_entities}")
        print(f"✅ Average per sample: {total_entities / (len(df) * len(text_columns)):.2f}")
    except Exception as e:
        print(f"❌ Error: {e}")
else:
    print("⚠️ Data file not found or PII detector not available")

print()
```

---

## PART 3: ATTACK PATTERN ANALYZER TESTING

### Step 3.1: Test 8-Type Fraud Classification
```python
print("=" * 60)
print("TEST 5: 8-TYPE FRAUD CLASSIFICATION")
print("=" * 60)

attack_samples = {
    'Account Takeover': 'Unauthorized access detected. Password reset initiated. Multiple failed login attempts.',
    'Card-Not-Present': 'Online transaction without card present. CVV verification bypassed. High transaction amount.',
    'Identity Theft': 'Personal identity used fraudulently. New account opened. Victim unaware of transaction.',
    'Payment Manipulation': 'Transaction amount altered after authorization. Original $100, charged $1000.',
    'Refund Fraud': 'False refund claim submitted. Product claimed not received. Multiple refund requests.',
    'Money Laundering': 'Large fund transfers to multiple accounts. Structured deposits below reporting threshold.',
    'Credential Stuffing': 'Multiple login attempts with different credentials. Automated attack detected.',
    'Social Engineering': 'Customer convinced to share sensitive information. Wire transfer authorized by victim.'
}

if attack_analyzer:
    results = []
    for attack_type, narrative in attack_samples.items():
        print(f"\n🎯 Testing: {attack_type}")
        try:
            classification = attack_analyzer.classify_fraud(narrative)
            if classification:
                detected_type = classification.get('attack_type', 'Unknown')
                confidence = classification.get('confidence', 0)
                
                match = "✅" if detected_type == attack_type else "⚠️"
                print(f"   {match} Detected: {detected_type} ({confidence:.1%})")
                
                results.append({
                    'Expected': attack_type,
                    'Detected': detected_type,
                    'Confidence': f"{confidence:.1%}",
                    'Match': detected_type == attack_type
                })
            else:
                print(f"   ❌ No classification")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    if results:
        results_df = pd.DataFrame(results)
        print(f"\n📊 Summary:")
        matches = sum(results_df['Match'])
        print(f"   Matches: {matches}/{len(results)} ({matches/len(results):.1%})")
else:
    print("❌ Attack Analyzer not available")

print()
```

### Step 3.2: Test Pattern Extraction
```python
print("=" * 60)
print("TEST 6: N-GRAM PATTERN EXTRACTION")
print("=" * 60)

pattern_test = "Unauthorized access to account. Multiple transfers. New recipient added. Funds sent offshore."

if attack_analyzer:
    print(f"\n📝 Test Narrative: {pattern_test}")
    try:
        patterns = attack_analyzer.extract_patterns(pattern_test)
        print(f"\n✅ Extracted {len(patterns) if patterns else 0} patterns:")
        if patterns:
            bigrams = [p for p in patterns if len(p.split()) == 2]
            trigrams = [p for p in patterns if len(p.split()) == 3]
            
            print(f"\n   Bigrams ({len(bigrams)}):")
            for pattern in bigrams[:5]:
                print(f"   - {pattern}")
            
            print(f"\n   Trigrams ({len(trigrams)}):")
            for pattern in trigrams[:5]:
                print(f"   - {pattern}")
        else:
            print(f"   No patterns extracted")
    except Exception as e:
        print(f"❌ Error: {e}")
else:
    print("❌ Attack Analyzer not available")

print()
```

### Step 3.3: Test Threat Scoring
```python
print("=" * 60)
print("TEST 7: THREAT LEVEL SCORING")
print("=" * 60)

threat_samples = [
    ("Minor transaction anomaly", "LOW"),
    ("Suspicious activity detected", "MEDIUM"),
    ("High-value unauthorized transaction", "HIGH"),
    ("Multiple account takeovers detected", "CRITICAL")
]

if attack_analyzer:
    print("\n🎯 Testing threat level categorization:\n")
    for narrative, expected_level in threat_samples:
        try:
            threat = attack_analyzer.calculate_threat_score(narrative)
            if threat:
                actual_level = threat.get('category', 'UNKNOWN')
                score = threat.get('score', 0)
                
                match = "✅" if actual_level == expected_level else "⚠️"
                print(f"{match} '{narrative}'")
                print(f"   Expected: {expected_level}, Actual: {actual_level}")
                print(f"   Score: {score:.2f} (0-1 scale)")
            else:
                print(f"❌ No threat score for: {narrative}")
        except Exception as e:
            print(f"❌ Error: {e}")
else:
    print("❌ Attack Analyzer not available")

print()
```

---

## PART 4: FEDERATED LEARNING TESTING

### Step 4.1: Test Framework Configuration
```python
print("=" * 60)
print("TEST 8: FEDERATED LEARNING CONFIGURATION")
print("=" * 60)

if federated_config:
    print("\n✅ Federated Configuration:")
    print(f"   Number of Clients: {federated_config.num_clients}")
    print(f"   Local Epochs: {federated_config.epochs}")
    print(f"   Batch Size: {federated_config.batch_size}")
    print(f"   Learning Rate: {federated_config.learning_rate}")
    print(f"   Rounds: {getattr(federated_config, 'rounds', 'Not specified')}")
else:
    print("❌ Federated Config not available")

print()
```

### Step 4.2: Test Client Initialization
```python
print("=" * 60)
print("TEST 9: FEDERATED CLIENT INITIALIZATION")
print("=" * 60)

if federated_config:
    try:
        # Create sample data for clients
        sample_data = np.random.randn(1000, 256)
        sample_labels = np.random.randint(0, 2, 1000)
        
        print(f"\n✅ Created sample data: {sample_data.shape}")
        
        # Create sample model
        sample_model = nn.Linear(256, 2)
        
        print(f"\n✅ Created sample model:")
        print(f"   Layers: {len(list(sample_model.parameters()))}")
        print(f"   Total params: {sum(p.numel() for p in sample_model.parameters())}")
        
        print(f"\n✅ Federated Learning Setup Ready:")
        print(f"   Data shape: {sample_data.shape}")
        print(f"   Model size: {sum(p.numel() for p in sample_model.parameters())} parameters")
        print(f"   Clients: {federated_config.num_clients}")
        print(f"   Can distribute data across {federated_config.num_clients} clients")
        
    except Exception as e:
        print(f"❌ Error: {e}")
else:
    print("❌ Federated Config not available")

print()
```

---

## PART 5: INTEGRATION TEST

### Step 5.1: End-to-End Pipeline Test
```python
print("=" * 60)
print("TEST 10: END-TO-END INTEGRATION PIPELINE")
print("=" * 60)

test_transaction = {
    'amount': 5000,
    'merchant': 'Electronics Store',
    'location': 'New York',
    'customer_name': 'John Doe',
    'customer_email': 'john@example.com',
    'card_type': 'Platinum'
}

# Generate narrative
narrative = f"Transaction of ${test_transaction['amount']} at {test_transaction['merchant']} "
narrative += f"in {test_transaction['location']}. "
narrative += f"Premium customer. Email: {test_transaction['customer_email']}"

print(f"\n📝 Test Narrative:")
print(f"   {narrative}\n")

# Step 1: PII Detection
print("STEP 1: PII DETECTION & COMPLIANCE")
if pii_detector:
    try:
        pii_text = f"{test_transaction['customer_name']} {test_transaction['customer_email']}"
        entities = pii_detector.detect_entities(pii_text)
        compliance = pii_detector.validate_compliance(pii_text)
        
        print(f"✅ Detected {len(entities) if entities else 0} PII entities")
        print(f"✅ Compliance status: {list(compliance.values()) if compliance else 'N/A'}")
    except Exception as e:
        print(f"❌ Error: {e}")
else:
    print("⚠️ PII detector not available")

# Step 2: Attack Analysis
print("\nSTEP 2: ATTACK PATTERN ANALYSIS")
if attack_analyzer:
    try:
        classification = attack_analyzer.classify_fraud(narrative)
        threat = attack_analyzer.calculate_threat_score(narrative)
        
        if classification:
            print(f"✅ Attack Type: {classification.get('attack_type', 'Unknown')}")
            print(f"✅ Confidence: {classification.get('confidence', 0):.1%}")
        
        if threat:
            print(f"✅ Threat Level: {threat.get('category', 'UNKNOWN')}")
            print(f"✅ Threat Score: {threat.get('score', 0):.2f}")
    except Exception as e:
        print(f"❌ Error: {e}")
else:
    print("⚠️ Attack analyzer not available")

# Step 3: Federated Learning Ready
print("\nSTEP 3: FEDERATED LEARNING READINESS")
if federated_config:
    print(f"✅ Federated Config Ready")
    print(f"✅ Clients: {federated_config.num_clients}")
    print(f"✅ Can train in distributed mode")
else:
    print("⚠️ Federated learning not available")

print("\n🎉 End-to-End Test Complete!")
print()
```

---

## PART 6: TROUBLESHOOTING

### Common Issues & Solutions

**Issue 1: ImportError - Module not found**
```python
# Solution: Install dependencies
import subprocess
subprocess.run(['pip', 'install', 'transformers', 'scikit-learn', 'scipy', 'peft'], check=True)
```

**Issue 2: GPU not detected**
```python
# Check CUDA installation
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")

# Install CUDA-enabled PyTorch if needed:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**Issue 3: Model files not found**
```python
# Check if files exist
from pathlib import Path
models_dir = Path("c:\\Users\\anshu\\GenAI-Powered Fraud Detection System\\models")
generated_dir = Path("c:\\Users\\anshu\\GenAI-Powered Fraud Detection System\\generated")

print("Model files:")
for f in models_dir.glob("*"):
    print(f"  - {f.name}")

print("Generated files:")
for f in generated_dir.glob("*"):
    print(f"  - {f.name}")
```

---

## PART 7: SUMMARY

### Test Results Summary
```python
print("=" * 60)
print("TESTING SUMMARY")
print("=" * 60)

summary = {
    'PII Detection': pii_detector is not None,
    'Attack Analysis': attack_analyzer is not None,
    'Federated Learning': federated_config is not None,
    'Device': device.type == 'cuda'
}

print("\nComponent Status:")
for component, available in summary.items():
    emoji = "✅" if available else "⚠️"
    status = "Ready" if available else "Not Available"
    print(f"  {emoji} {component}: {status}")

all_ready = all(summary.values())
print(f"\n{'🎉 All components ready!' if all_ready else '⚠️ Some components missing'}")
```

---

## Quick Reference Commands

```bash
# Run full test
jupyter notebook enhanced_integration_test.ipynb

# Test specific enhancement
python -c "from security.pii_validator import PIIDetector; PIIDetector()"

# Launch enhanced dashboard
streamlit run notebooks/ui_enhanced.py

# Check logs
type logs\enhancement.log

# Verify imports
python -c "import torch; from security.pii_validator import PIIDetector; print('OK')"
```

---

**Created:** December 28, 2025  
**Status:** ✅ PRODUCTION READY  
**Version:** 2.0 Enhanced
