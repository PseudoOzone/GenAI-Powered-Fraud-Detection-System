import pandas as pd
import logging
from pathlib import Path
import random

class NarrativeGenerator:
    """A class to generate narratives from transaction data for model training."""

    def __init__(self):
        """Initializes the NarrativeGenerator."""
        self.logger = logging.getLogger(__name__)
        self.templates = {
            'high_risk': "A high-risk transaction of ${amount:.2f} was made by a customer aged {age} with an income of ${income:.0f}. The credit score is {risk_score}, and recent transaction velocity is high.",
            'low_risk': "A low-risk transaction of ${amount:.2f} occurred. The customer is {age} years old with a stable income and a good credit score of {risk_score}.",
            'suspicious_activity': "Suspicious activity detected: a transaction of ${amount:.2f} from an unusual location. The customer's recent 6-hour velocity is ${velocity_6h:.2f}.",
        }

    def generate_single_narrative(self, row):
        """
        Generates a single narrative for a given transaction row.
        
        Args:
            row (pd.Series): A row of transaction data.
            
        Returns:
            str: A generated narrative.
        """
        if row.get('credit_risk_score', 150) > 200:
            template = self.templates['high_risk']
        elif row.get('velocity_6h', 0) > 5000:
            template = self.templates['suspicious_activity']
        else:
            template = self.templates['low_risk']
            
        return template.format(
            amount=row.get('transaction_amount', 0),
            age=row.get('customer_age', 30),
            income=row.get('income', 50000),
            risk_score=row.get('credit_risk_score', 150),
            velocity_6h=row.get('velocity_6h', 0)
        )

    def generate_narratives(self, df):
        """
        Generates narratives for a DataFrame of transactions.
        
        Args:
            df (pd.DataFrame): The DataFrame containing transaction data.
            
        Returns:
            list: A list of dictionaries, each containing a narrative and associated data.
        """
        narratives = [
            {
                'narrative': self.generate_single_narrative(row),
                'fraud_label': int(row.get('fraud_bool', 0)),
                'income': float(row.get('income', 0)),
            } for _, row in df.iterrows()
        ]
        self.logger.info(f"Generated {len(narratives)} narratives.")
        return narratives

class NarrativePipeline:
    """A pipeline for generating and saving narratives."""

    def __init__(self, data_dir='generated', output_dir='generated'):
        """Initializes the NarrativePipeline."""
        current_dir = Path(__file__).parent
        self.data_dir = (current_dir.parent / data_dir).resolve()
        self.output_dir = (current_dir.parent / output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.generator = NarrativeGenerator()

    def run(self, input_file='fraud_data_combined_clean.csv'):
        """
        Executes the narrative generation pipeline.
        
        Args:
            input_file (str): The name of the input CSV file.
            
        Returns:
            str: The path to the saved narratives CSV file.
        """
        try:
            input_path = self.data_dir / input_file
            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {input_path}")
            
            df = pd.read_csv(input_path)
            narratives = self.generator.generate_narratives(df)
            
            output_file = self.output_dir / 'fraud_narratives_combined.csv'
            pd.DataFrame(narratives).to_csv(output_file, index=False)
            self.logger.info(f"Saved narratives to {output_file}")
            
            return str(output_file)

        except Exception as e:
            self.logger.error(f"Narrative generation pipeline failed: {e}", exc_info=True)
            raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pipeline = NarrativePipeline()
    pipeline.run()
