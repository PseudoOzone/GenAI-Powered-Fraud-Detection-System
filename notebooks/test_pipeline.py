"""
Quick Test Script - Verify Pipeline Modules
Tests each module with sample data before full pipeline execution
"""

import sys
import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test all module imports"""
    logger.info("=" * 60)
    logger.info("TEST 1: Module Imports")
    logger.info("=" * 60)
    
    try:
        from pii_cleaner import PIICleaner
        logger.info("✓ pii_cleaner imported")
    except Exception as e:
        logger.error(f"✗ pii_cleaner import failed: {e}")
        return False
    
    try:
        from genai_narrative_generator import NarrativeGeneratorPipeline
        logger.info("✓ genai_narrative_generator imported")
    except Exception as e:
        logger.error(f"✗ genai_narrative_generator import failed: {e}")
        return False
    
    try:
        from genai_embedding_model import EmbeddingPipeline
        logger.info("✓ genai_embedding_model imported")
    except Exception as e:
        logger.error(f"✗ genai_embedding_model import failed: {e}")
        return False
    
    try:
        from fraud_gpt_trainer import GPT2Pipeline
        logger.info("✓ fraud_gpt_trainer imported")
    except Exception as e:
        logger.error(f"✗ fraud_gpt_trainer import failed: {e}")
        return False
    
    return True


def test_data_loading():
    """Test loading actual data"""
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST 2: Data Loading")
    logger.info("=" * 60)
    
    data_dir = Path('../data')
    
    datasets = ['Base.csv', 'Variant I.csv', 'Variant II.csv', 
                'Variant III.csv', 'Variant IV.csv', 'Variant V.csv']
    
    for ds in datasets:
        try:
            # Load first 1000 rows
            df = pd.read_csv(data_dir / ds, nrows=1000)
            logger.info(f"✓ {ds}: {len(df)} rows, {len(df.columns)} columns")
            
            # Check fraud column
            if 'fraud_bool' in df.columns:
                fraud_count = df['fraud_bool'].sum()
                logger.info(f"  → Fraud cases: {fraud_count}")
            
        except Exception as e:
            logger.error(f"✗ {ds} failed: {e}")
            return False
    
    return True


def test_pii_cleaner():
    """Test PII cleaner module"""
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST 3: PII Cleaner Module")
    logger.info("=" * 60)
    
    try:
        from pii_cleaner import PIICleaner
        
        # Create small test data
        test_data = {
            'fraud_bool': [0, 1, 0],
            'customer_age': [35, 42, 28],
            'income': [50000, 75000, 45000],
            'credit_risk_score': [150, 200, 100]
        }
        test_df = pd.DataFrame(test_data)
        
        # Test cleaning
        cleaner = PIICleaner(data_dir='../data', output_dir='../generated')
        cleaned = cleaner.pii_guard.clean_dataframe(test_df)
        
        logger.info(f"✓ PII Cleaner tested on {len(cleaned)} rows")
        logger.info(f"  → Columns preserved: {len(cleaned.columns)}")
        
        return True
    except Exception as e:
        logger.error(f"✗ PII Cleaner test failed: {e}")
        return False


def test_narrative_generator():
    """Test narrative generator"""
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST 4: Narrative Generator")
    logger.info("=" * 60)
    
    try:
        from genai_narrative_generator import NarrativeGenerator
        
        # Create test profile
        test_profile = {
            'customer_age': 45,
            'income': 75000,
            'credit_risk_score': 175,
            'velocity_6h': 3000,
            'velocity_24h': 8000,
            'velocity_4w': 45000,
            'days_since_request': 5,
            'employment_status': 'Employed',
            'housing_status': 'Homeowner',
            'payment_type': 'AD',
            'device_fraud_count': 1
        }
        
        gen = NarrativeGenerator()
        narrative = gen.generate_narrative(test_profile)
        
        logger.info("✓ Narrative Generator works")
        logger.info(f"  → Generated narrative ({len(narrative)} chars):")
        logger.info(f"     {narrative[:100]}...")
        
        return True
    except Exception as e:
        logger.error(f"✗ Narrative Generator test failed: {e}")
        return False


def test_gpu_detection():
    """Test GPU detection"""
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST 5: GPU Detection")
    logger.info("=" * 60)
    
    try:
        import torch
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"✓ Device detected: {device}")
        
        if torch.cuda.is_available():
            logger.info(f"  → GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"  → CUDA Version: {torch.version.cuda}")
            logger.info(f"  → Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            logger.warning("  → GPU not available, will use CPU")
        
        return True
    except Exception as e:
        logger.error(f"✗ GPU detection failed: {e}")
        return False


def main():
    """Run all tests"""
    logger.info("\n")
    logger.info("╔" + "=" * 58 + "╗")
    logger.info("║" + " " * 15 + "PIPELINE QUICK TEST SUITE" + " " * 19 + "║")
    logger.info("╚" + "=" * 58 + "╝")
    logger.info("\n")
    
    results = {
        'Imports': test_imports(),
        'Data Loading': test_data_loading(),
        'PII Cleaner': test_pii_cleaner(),
        'Narrative Generator': test_narrative_generator(),
        'GPU Detection': test_gpu_detection()
    }
    
    logger.info("\n")
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info("")
    logger.info(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n✓ ALL TESTS PASSED - Ready to run full pipeline!")
        return 0
    else:
        logger.error("\n✗ Some tests failed - Fix issues before running pipeline")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
