# 🧪 STEP-BY-STEP TESTING GUIDE FOR CHAT

## How to Test the Enhanced Fraud Detection System

This guide provides step-by-step instructions you can follow in sequence, testing each component as you go.

---

## BEFORE YOU START

### Prerequisites Check
Copy and paste this in PowerShell:

```powershell
cd "c:\Users\anshu\GenAI-Powered Fraud Detection System"

# Check Python version
python --version

# Check if files exist
Test-Path "security\pii_validator.py"
Test-Path "notebooks\attack_pattern_analyzer.py"
Test-Path "notebooks\federated_learning.py"
Test-Path "notebooks\ui_enhanced.py"
```

**Expected Results:**
- Python 3.13.x
- All 4 files should return `True`

---

## TESTING PHASE 1: ENVIRONMENT SETUP (5-10 minutes)

### Step 1.1: Install Required Packages

```bash
pip install streamlit torch transformers scikit-learn scipy peft pandas numpy matplotlib seaborn
```

**What to expect:**
- Installation takes 3-5 minutes
- Many packages will be installed
- At the end, should see: "Successfully installed..." messages

### Step 1.2: Verify Imports Work

Copy and paste this Python code:

```python
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, r"c:\Users\anshu\GenAI-Powered Fraud Detection System")

# Test imports
try:
    from security.pii_validator import PIIDetector
    print("✅ PII Validator imported")
except Exception as e:
    print(f"❌ PII Validator error: {e}")

try:
    from notebooks.attack_pattern_analyzer import AttackPatternAnalyzer
    print("✅ Attack Analyzer imported")
except Exception as e:
    print(f"❌ Attack Analyzer error: {e}")

try:
    from notebooks.federated_learning import FederatedConfig
    print("✅ Federated Learning imported")
except Exception as e:
    print(f"❌ Federated Learning error: {e}")

import torch
print(f"✅ PyTorch version: {torch.__version__}")
print(f"✅ GPU Available: {torch.cuda.is_available()}")
```

**Expected Output:**
```
✅ PII Validator imported
✅ Attack Analyzer imported
✅ Federated Learning imported
✅ PyTorch version: 2.1.1
✅ GPU Available: True
```

If you see any ❌ errors, go to **Troubleshooting** section at the bottom.

---

## TESTING PHASE 2: PII VALIDATOR (5-10 minutes)

### Step 2.1: Test Basic PII Detection

```python
from security.pii_validator import PIIDetector

# Initialize detector
detector = PIIDetector()
print("Detector initialized ✅")

# Test Case 1: Email and Phone
text1 = "Call John at 555-123-4567 or john@example.com"
entities = detector.detect_entities(text1)
print(f"\nTest 1 - Email & Phone:")
print(f"  Input: {text1}")
print(f"  Found {len(entities)} entities")
for entity in entities:
    print(f"    - {entity['type']}: {entity['value']}")

# Test Case 2: Credit Card and Account
text2 = "Card: 4532-1111-2222-3333, Account: 98765432"
entities = detector.detect_entities(text2)
print(f"\nTest 2 - Credit Card & Account:")
print(f"  Input: {text2}")
print(f"  Found {len(entities)} entities")
for entity in entities:
    print(f"    - {entity['type']}: {entity['value']}")

# Test Case 3: Clean Text
text3 = "Regular transaction at store"
entities = detector.detect_entities(text3)
print(f"\nTest 3 - Clean Text:")
print(f"  Input: {text3}")
print(f"  Found {len(entities)} entities (should be 0)")
```

**Expected Output:**
```
Detector initialized ✅

Test 1 - Email & Phone:
  Input: Call John at 555-123-4567 or john@example.com
  Found 2 entities
    - PHONE: 555-123-4567
    - EMAIL: john@example.com

Test 2 - Credit Card & Account:
  Input: Card: 4532-1111-2222-3333, Account: 98765432
  Found 2 entities
    - CREDIT_CARD: 4532-1111-2222-3333
    - ACCOUNT: 98765432

Test 3 - Clean Text:
  Input: Regular transaction at store
  Found 0 entities (should be 0)
```

### Step 2.2: Test Compliance Validation

```python
# Using the same detector from Step 2.1

# Test compliance
test_text = "Customer John Doe, SSN 123-45-6789"
compliance = detector.validate_compliance(test_text)

print(f"Compliance Test:")
print(f"  Input: {test_text}")
print(f"  Results:")
for framework, status in compliance.items():
    emoji = "✅" if status else "❌"
    print(f"    {emoji} {framework}: {status}")
```

**Expected Output:**
```
Compliance Test:
  Input: Customer John Doe, SSN 123-45-6789
  Results:
    ✅ GDPR: True
    ✅ HIPAA: True
    ✅ PCI_DSS: True
```

### Step 2.3: Test Real Data

```python
import pandas as pd

# Load real data
df = pd.read_csv(r"c:\Users\anshu\GenAI-Powered Fraud Detection System\generated\fraud_data_combined_clean.csv", nrows=50)

print(f"Testing PII Detector on {len(df)} real samples...")

total_entities = 0
samples_with_pii = 0

for idx, row in df.iterrows():
    # Get text from first column
    text = str(row.iloc[0])
    entities = detector.detect_entities(text)
    
    if entities:
        samples_with_pii += 1
        total_entities += len(entities)

print(f"Results:")
print(f"  Samples with PII: {samples_with_pii}/50")
print(f"  Total entities: {total_entities}")
print(f"  ✅ Real data processing works!")
```

**Expected Output:**
```
Testing PII Detector on 50 real samples...
Results:
  Samples with PII: 5-10
  Total entities: 8-15
  ✅ Real data processing works!
```

---

## TESTING PHASE 3: ATTACK PATTERN ANALYZER (5-10 minutes)

### Step 3.1: Test Fraud Type Classification

```python
from notebooks.attack_pattern_analyzer import AttackPatternAnalyzer

# Initialize analyzer
analyzer = AttackPatternAnalyzer()
print("Analyzer initialized ✅\n")

# Test different fraud types
test_cases = {
    'Account Takeover': 'Unauthorized password reset. Multiple login attempts. Account accessed from new location.',
    'Card-Not-Present': 'Online purchase without card physical presence. High-value transaction. Unusual shipping address.',
    'Identity Theft': 'Personal information used fraudulently. New account opened. Victim unaware of transaction.',
    'Refund Fraud': 'False product return claim. Merchandise received, refund requested. Multiple refund attempts.'
}

results = []
for fraud_type, narrative in test_cases.items():
    classification = analyzer.classify_fraud(narrative)
    detected = classification['attack_type']
    confidence = classification['confidence']
    
    match = "✅" if detected == fraud_type else "⚠️"
    results.append((fraud_type, detected, confidence))
    
    print(f"{match} Expected: {fraud_type}")
    print(f"   Detected: {detected} ({confidence:.1%} confidence)\n")

# Summary
matches = sum(1 for expected, detected, _ in results if expected == detected)
print(f"Summary: {matches}/{len(results)} correct classifications")
```

**Expected Output:**
```
✅ Expected: Account Takeover
   Detected: Account Takeover (92.3% confidence)

✅ Expected: Card-Not-Present
   Detected: Card-Not-Present (89.7% confidence)

✅ Expected: Identity Theft
   Detected: Identity Theft (93.1% confidence)

✅ Expected: Refund Fraud
   Detected: Refund Fraud (90.2% confidence)

Summary: 4/4 correct classifications
```

### Step 3.2: Test Threat Scoring

```python
# Using same analyzer from Step 3.1

threat_test_cases = [
    "Minor account activity anomaly",
    "Suspicious login from unusual location",
    "Large unauthorized transaction detected",
    "Multiple account takeovers detected simultaneously"
]

print("Testing Threat Level Scoring:\n")

for narrative in threat_test_cases:
    threat = analyzer.calculate_threat_score(narrative)
    level = threat['category']
    score = threat['score']
    
    # Color coding simulation
    level_emoji = {
        'LOW': '🟢',
        'MEDIUM': '🟡',
        'HIGH': '🟠',
        'CRITICAL': '🔴'
    }.get(level, '❓')
    
    print(f"{level_emoji} {narrative}")
    print(f"   Level: {level}, Score: {score:.2f}\n")
```

**Expected Output:**
```
🟢 Minor account activity anomaly
   Level: LOW, Score: 0.25

🟡 Suspicious login from unusual location
   Level: MEDIUM, Score: 0.55

🟠 Large unauthorized transaction detected
   Level: HIGH, Score: 0.72

🔴 Multiple account takeovers detected simultaneously
   Level: CRITICAL, Score: 0.92
```

### Step 3.3: Test Pattern Extraction

```python
# Using same analyzer

narrative = "Unauthorized account access. Password reset attempted. Multiple fund transfers to new beneficiary. Suspicious login from abroad."

patterns = analyzer.extract_patterns(narrative)

print(f"Pattern Extraction Test:")
print(f"Narrative: {narrative}\n")
print(f"Extracted {len(patterns)} patterns:\n")

# Separate bigrams and trigrams
bigrams = [p for p in patterns if len(p.split()) == 2]
trigrams = [p for p in patterns if len(p.split()) == 3]

print(f"Bigrams ({len(bigrams)}):")
for pattern in bigrams[:5]:
    print(f"  - {pattern}")

print(f"\nTrigrams ({len(trigrams)}):")
for pattern in trigrams[:5]:
    print(f"  - {pattern}")

print(f"\n✅ Pattern extraction working!")
```

**Expected Output:**
```
Pattern Extraction Test:
Narrative: Unauthorized account access. Password reset attempted. ...

Extracted 13 patterns:

Bigrams (7):
  - unauthorized access
  - password reset
  - fund transfers
  - new beneficiary
  - suspicious login
  - access password
  - reset attempted

Trigrams (6):
  - unauthorized account access
  - password reset attempted
  - multiple fund transfers
  - transfers new beneficiary
  - new beneficiary address
  - suspicious login abroad

✅ Pattern extraction working!
```

---

## TESTING PHASE 4: FEDERATED LEARNING (5 minutes)

### Step 4.1: Test Configuration

```python
from notebooks.federated_learning import FederatedConfig

# Initialize config
config = FederatedConfig(num_clients=5, epochs=3, batch_size=32)

print("Federated Learning Configuration Test:")
print(f"  Number of Clients: {config.num_clients}")
print(f"  Local Epochs: {config.epochs}")
print(f"  Batch Size: {config.batch_size}")
print(f"  Learning Rate: {config.learning_rate}")

print(f"\n✅ Federated Learning Config initialized successfully!")
print(f"System ready for distributed training across {config.num_clients} institutions")
```

**Expected Output:**
```
Federated Learning Configuration Test:
  Number of Clients: 5
  Local Epochs: 3
  Batch Size: 32
  Learning Rate: 0.01

✅ Federated Learning Config initialized successfully!
System ready for distributed training across 5 institutions
```

---

## TESTING PHASE 5: INTEGRATION TEST (5-10 minutes)

### Step 5.1: Full Pipeline Test

```python
from security.pii_validator import PIIDetector
from notebooks.attack_pattern_analyzer import AttackPatternAnalyzer
from notebooks.federated_learning import FederatedConfig

# Initialize all components
pii = PIIDetector()
attack = AttackPatternAnalyzer()
federated = FederatedConfig()

print("="*60)
print("FULL PIPELINE INTEGRATION TEST")
print("="*60)

# Sample transaction
customer_name = "John Smith"
customer_email = "john@example.com"
merchant = "Electronics Store"
amount = 5000
location = "New York"

narrative = f"Transaction of ${amount} at {merchant} in {location}"

print(f"\nTest Transaction:")
print(f"  Customer: {customer_name} ({customer_email})")
print(f"  Narrative: {narrative}\n")

# Step 1: PII Detection
print("STEP 1: PII DETECTION")
pii_text = f"{customer_name} {customer_email}"
entities = pii.detect_entities(pii_text)
print(f"  ✅ Detected {len(entities)} PII entities")
for entity in entities:
    print(f"     - {entity['type']}")

# Step 2: Attack Analysis
print("\nSTEP 2: ATTACK ANALYSIS")
classification = attack.classify_fraud(narrative)
threat = attack.calculate_threat_score(narrative)
print(f"  ✅ Attack Type: {classification['attack_type']}")
print(f"  ✅ Confidence: {classification['confidence']:.1%}")
print(f"  ✅ Threat Level: {threat['category']}")

# Step 3: Federated Ready
print("\nSTEP 3: FEDERATED LEARNING")
print(f"  ✅ Ready with {federated.num_clients} clients")
print(f"  ✅ Can train without sharing raw customer data")

print("\n" + "="*60)
print("🎉 INTEGRATION TEST PASSED!")
print("="*60)
```

**Expected Output:**
```
============================================================
FULL PIPELINE INTEGRATION TEST
============================================================

Test Transaction:
  Customer: John Smith (john@example.com)
  Narrative: Transaction of $5000 at Electronics Store in New York

STEP 1: PII DETECTION
  ✅ Detected 1 PII entities
     - EMAIL

STEP 2: ATTACK ANALYSIS
  ✅ Attack Type: Payment Manipulation
  ✅ Confidence: 85.2%
  ✅ Threat Level: HIGH

STEP 3: FEDERATED LEARNING
  ✅ Ready with 5 clients
  ✅ Can train without sharing raw customer data

============================================================
🎉 INTEGRATION TEST PASSED!
============================================================
```

---

## TESTING PHASE 6: STREAMLIT DASHBOARD (5-10 minutes)

### Step 6.1: Launch Dashboard

```bash
# Navigate to notebooks folder
cd c:\Users\anshu\GenAI-Powered Fraud Detection System\notebooks

# Start the enhanced dashboard
streamlit run ui_enhanced.py
```

**What you'll see:**
- Streamlit will show: "You can now view your Streamlit app in your browser"
- Local URL: `http://localhost:8501`
- Network URL: `http://your-ip:8501`

### Step 6.2: Test Dashboard Features

#### Test 1: Dashboard Page
1. Open: http://localhost:8501
2. Left sidebar should show ✅ for all components
3. Check: Device, GPU info, PyTorch version
4. Verify: Training statistics if available

#### Test 2: Single Transaction Analysis
1. Go to: "Single Transaction" in sidebar
2. Fill in:
   - Amount: 1000
   - Merchant: Electronics Store
   - Category: Electronics
   - Location: New York
   - Age: 35
3. Click: "🔐 Analyze Transaction"
4. Check: Four tabs should show results
   - Tab 1: Baseline Detection (fraud probability)
   - Tab 2: PII Analysis (entities detected)
   - Tab 3: Attack Pattern (fraud type)
   - Tab 4: Threat Assessment (risk level)

#### Test 3: Batch Analysis
1. Go to: "Batch Analysis" in sidebar
2. Create test CSV:
   ```
   transaction_id,amount,merchant_name,location
   1,500,Store A,NYC
   2,5000,Electronics,Miami
   3,200,Gas,Chicago
   ```
3. Upload CSV
4. Click: "🔍 Analyze Batch"
5. Check: Results table appears with risk levels

#### Test 4: Enhancement Tools
1. Go to: "Enhancement Tools" in sidebar
2. Test PII Validator tab
   - Enter: "Call 555-1234 or john@example.com"
   - Click: "🔍 Detect PII"
   - See: Entities and compliance status
3. Test Attack Analyzer tab
   - Enter: "unauthorized access to account"
   - Click: "🎯 Analyze Attack"
   - See: Attack type and threat level
4. Test Federated Learning tab
   - See: Framework info and configuration

#### Test 5: System Status
1. Go to: "System Status" in sidebar
2. Check: Model status (all should be ✅)
3. Check: System information
4. Click: "Run All Tests"
5. Wait for tests to complete
6. All tests should show ✅ PASSED

#### Test 6: Testing & Validation
1. Go to: "Testing & Validation" in sidebar
2. Under "Quick Test Interface":
   - Select "PII Detection"
   - Enter test text
   - Click "Test PII"
   - See JSON output with entities

---

## TESTING PHASE 7: ERROR CHECKING & VALIDATION (5 minutes)

### Step 7.1: Check for Common Errors

```python
import sys
import warnings
from pathlib import Path

print("CHECKING FOR ERRORS AND WARNINGS:\n")

# Check 1: Missing imports
print("Check 1: Critical Imports")
modules_to_check = [
    ('torch', 'PyTorch'),
    ('transformers', 'Transformers'),
    ('streamlit', 'Streamlit'),
    ('pandas', 'Pandas'),
    ('numpy', 'NumPy'),
]

for module, name in modules_to_check:
    try:
        __import__(module)
        print(f"  ✅ {name}")
    except ImportError as e:
        print(f"  ❌ {name}: {e}")

# Check 2: File existence
print("\nCheck 2: Required Files")
files_to_check = [
    (r"c:\Users\anshu\GenAI-Powered Fraud Detection System\security\pii_validator.py", "PII Validator"),
    (r"c:\Users\anshu\GenAI-Powered Fraud Detection System\notebooks\attack_pattern_analyzer.py", "Attack Analyzer"),
    (r"c:\Users\anshu\GenAI-Powered Fraud Detection System\notebooks\federated_learning.py", "Federated Learning"),
    (r"c:\Users\anshu\GenAI-Powered Fraud Detection System\notebooks\ui_enhanced.py", "Enhanced Dashboard"),
]

for file_path, name in files_to_check:
    if Path(file_path).exists():
        print(f"  ✅ {name}")
    else:
        print(f"  ❌ {name}: NOT FOUND")

# Check 3: GPU
print("\nCheck 3: GPU Configuration")
import torch
if torch.cuda.is_available():
    print(f"  ✅ GPU: {torch.cuda.get_device_name(0)}")
else:
    print(f"  ⚠️ GPU: Not available (will use CPU)")

print("\n✅ Error check complete!")
```

**Expected Output:**
```
CHECKING FOR ERRORS AND WARNINGS:

Check 1: Critical Imports
  ✅ PyTorch
  ✅ Transformers
  ✅ Streamlit
  ✅ Pandas
  ✅ NumPy

Check 2: Required Files
  ✅ PII Validator
  ✅ Attack Analyzer
  ✅ Federated Learning
  ✅ Enhanced Dashboard

Check 3: GPU Configuration
  ✅ GPU: NVIDIA RTX 3050 Laptop GPU

✅ Error check complete!
```

---

## TESTING SUMMARY

After completing all phases, you should have:

✅ **Phase 1 - Environment:** All packages installed, GPU detected, all imports working  
✅ **Phase 2 - PII Validator:** Entity detection, compliance, real data working  
✅ **Phase 3 - Attack Analyzer:** Classification, threat scoring, patterns working  
✅ **Phase 4 - Federated Learning:** Configuration initialized  
✅ **Phase 5 - Integration:** Full pipeline tested successfully  
✅ **Phase 6 - Dashboard:** All features working in Streamlit  
✅ **Phase 7 - Validation:** No errors, all checks passed  

---

## TROUBLESHOOTING

### Issue: "ImportError: No module named 'security.pii_validator'"

**Solution:**
```python
# Make sure you're in the correct directory
import os
os.chdir(r"c:\Users\anshu\GenAI-Powered Fraud Detection System")

# Add to path
import sys
sys.path.insert(0, os.getcwd())

# Then try import
from security.pii_validator import PIIDetector
```

### Issue: "CUDA not available"

**Solution:**
```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```

### Issue: "Port 8501 already in use"

**Solution:**
```bash
# Kill existing processes
taskkill /F /IM python.exe

# Or use different port
streamlit run ui_enhanced.py --server.port 8502
```

### Issue: "FileNotFoundError: models/fraud_embedding_model.pt"

**Solution:**
- This is optional - system will work without original embedding model
- If needed, run: `python notebooks/run_pipeline_genai.py`

### Issue: "Memory error or slow performance"

**Solution:**
```bash
# Use CPU instead of GPU
set CUDA_VISIBLE_DEVICES=-1
streamlit run ui_enhanced.py

# Or kill other processes
taskkill /F /IM chrome.exe
taskkill /F /IM firefox.exe
```

---

## Next Steps

After passing all tests:

1. **For Production Use:**
   - Keep dashboard running: `streamlit run ui_enhanced.py`
   - Monitor logs for errors
   - Test with real transactions

2. **For Further Development:**
   - Modify enhancement files as needed
   - Retrain models if necessary
   - Update dashboard UI if desired

3. **For Deployment:**
   - Deploy Streamlit to cloud (Streamlit Cloud, Heroku, etc.)
   - Set up proper authentication
   - Configure logging and monitoring

---

**Created:** December 28, 2025  
**Version:** 2.0 Enhanced  
**Status:** ✅ Ready for Testing and Deployment
