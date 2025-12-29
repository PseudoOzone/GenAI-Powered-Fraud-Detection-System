"""
GenAI Fraud Detection Pipeline Coordinator
Orchestrates all 4 steps: PII Cleaning, Narrative Generation, Embedding Model, GPT-2 LoRA
"""

import torch
import sys
import logging
import traceback
from datetime import datetime
from pathlib import Path

# Import pipeline modules
from pii_cleaner import PIICleaner
from genai_narrative_generator import NarrativeGeneratorPipeline
from genai_embedding_model import EmbeddingPipeline
from fraud_gpt_trainer import GPT2Pipeline


class GenAIPipelineCoordinator:
    """Main pipeline orchestrator"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        self.logger.info("=" * 80)
        self.logger.info("GenAI Fraud Detection Pipeline Coordinator")
        self.logger.info("=" * 80)
        
        # Device info
        self.log_device_info()
    
    def setup_logging(self):
        """Setup logging to file and console"""
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'genai_pipeline_{timestamp}.log'
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        self.log_file = log_file
    
    def log_device_info(self):
        """Log GPU/CPU device information"""
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("SYSTEM INFORMATION")
        self.logger.info("=" * 80)
        
        self.logger.info(f"PyTorch Version: {torch.__version__}")
        
        if torch.cuda.is_available():
            self.logger.info(f"CUDA Available: Yes")
            self.logger.info(f"CUDA Version: {torch.version.cuda}")
            self.logger.info(f"GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                self.logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                self.logger.info(f"  Memory: {props.total_memory / 1e9:.2f} GB")
        else:
            self.logger.warning("CUDA NOT Available - Using CPU")
        
        self.logger.info("")
    
    def run_step1_pii_cleaning(self):
        """Step 1: PII Cleaning"""
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("[STEP1] PII CLEANING")
        self.logger.info("=" * 80)
        
        try:
            cleaner = PIICleaner(data_dir='data', output_dir='generated')
            output_path = cleaner.run()
            self.logger.info(f"[STEP1] ✓ Completed: {output_path}")
            return True, output_path
        except Exception as e:
            self.logger.error(f"[STEP1] ✗ Failed: {e}")
            self.logger.error(traceback.format_exc())
            return False, None
    
    def run_step2_narrative_generation(self):
        """Step 2: Fraud Narrative Generation"""
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("[STEP2] FRAUD NARRATIVE GENERATION")
        self.logger.info("=" * 80)
        
        try:
            pipeline = NarrativeGeneratorPipeline(data_dir='generated', output_dir='generated')
            output_path = pipeline.run(sample_size=5000)
            self.logger.info(f"[STEP2] ✓ Completed: {output_path}")
            return True, output_path
        except Exception as e:
            self.logger.error(f"[STEP2] ✗ Failed: {e}")
            self.logger.error(traceback.format_exc())
            return False, None
    
    def run_step3_embedding_model(self):
        """Step 3: Fraud Embedding Model"""
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("[STEP3] FRAUD EMBEDDING MODEL")
        self.logger.info("=" * 80)
        
        try:
            pipeline = EmbeddingPipeline(data_dir='generated', model_dir='models')
            output_path = pipeline.run(epochs=3, batch_size=16)
            self.logger.info(f"[STEP3] ✓ Completed: {output_path}")
            return True, output_path
        except Exception as e:
            self.logger.error(f"[STEP3] ✗ Failed: {e}")
            self.logger.error(traceback.format_exc())
            return False, None
    
    def run_step4_gpt2_lora(self):
        """Step 4: GPT-2 LoRA Fine-tuning"""
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("[STEP4] GPT-2 LoRA FINE-TUNING")
        self.logger.info("=" * 80)
        
        try:
            pipeline = GPT2Pipeline(data_dir='generated', model_dir='models')
            output_path = pipeline.run(epochs=3, batch_size=8)
            self.logger.info(f"[STEP4] ✓ Completed: {output_path}")
            return True, output_path
        except Exception as e:
            self.logger.error(f"[STEP4] ✗ Failed: {e}")
            self.logger.error(traceback.format_exc())
            return False, None
    
    def log_summary(self, results):
        """Log final summary"""
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("PIPELINE SUMMARY")
        self.logger.info("=" * 80)
        
        step_names = ["Step 1: PII Cleaning", "Step 2: Narrative Generation", 
                     "Step 3: Embedding Model", "Step 4: GPT-2 LoRA"]
        
        for i, (success, output) in enumerate(results):
            status = "✓ SUCCESS" if success else "✗ FAILED"
            self.logger.info(f"{step_names[i]}: {status}")
            if output:
                self.logger.info(f"  Output: {output}")
        
        elapsed = datetime.now() - self.start_time
        self.logger.info(f"Total Time: {elapsed}")
        self.logger.info(f"Log File: {self.log_file}")
        
        # Check all outputs
        if all(success for success, _ in results):
            self.logger.info("")
            self.logger.info("=" * 80)
            self.logger.info("✓ ALL STEPS COMPLETED SUCCESSFULLY!")
            self.logger.info("=" * 80)
            return 0
        else:
            self.logger.info("")
            self.logger.info("=" * 80)
            self.logger.info("✗ PIPELINE FAILED - Check logs for details")
            self.logger.info("=" * 80)
            return 1
    
    def run(self):
        """Execute full pipeline"""
        results = []
        
        # Step 1: PII Cleaning
        success, output = self.run_step1_pii_cleaning()
        results.append((success, output))
        if not success:
            return self.log_summary(results)
        
        # Step 2: Narrative Generation
        success, output = self.run_step2_narrative_generation()
        results.append((success, output))
        if not success:
            return self.log_summary(results)
        
        # Step 3: Embedding Model
        success, output = self.run_step3_embedding_model()
        results.append((success, output))
        if not success:
            return self.log_summary(results)
        
        # Step 4: GPT-2 LoRA
        success, output = self.run_step4_gpt2_lora()
        results.append((success, output))
        
        # Summary and exit
        return self.log_summary(results)


if __name__ == "__main__":
    coordinator = GenAIPipelineCoordinator()
    exit_code = coordinator.run()
    sys.exit(exit_code)