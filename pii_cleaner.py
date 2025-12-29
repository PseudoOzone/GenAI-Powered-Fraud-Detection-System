import pandas as pd
import logging
from pathlib import Path

class PIICleaner:
    """A class to clean personally identifiable information (PII) from datasets."""

    def __init__(self, data_dir='data', output_dir='generated'):
        """
        Initializes the PIICleaner with specified data and output directories.
        
        Args:
            data_dir (str): The directory where the input data is located.
            output_dir (str): The directory where the cleaned data will be saved.
        """
        current_dir = Path(__file__).parent
        self.data_dir = (current_dir.parent / data_dir).resolve()
        self.output_dir = (current_dir.parent / output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def load_datasets(self, file_names, sample_size=None):
        """
        Loads multiple datasets from CSV files into a dictionary of pandas DataFrames.
        
        Args:
            file_names (list of str): A list of CSV file names to load.
            sample_size (int, optional): The number of rows to load from each file.
            
        Returns:
            dict: A dictionary mapping file names to their corresponding DataFrames.
        """
        datasets = {}
        for file_name in file_names:
            file_path = self.data_dir / file_name
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path, nrows=sample_size)
                    datasets[file_name.replace('.csv', '')] = df
                    self.logger.info(f"Loaded {file_name}: {len(df)} rows")
                except Exception as e:
                    self.logger.error(f"Error loading {file_name}: {e}")
            else:
                self.logger.warning(f"Dataset not found: {file_path}")
        return datasets

    def clean_data(self, df, columns_to_drop):
        """
        Removes specified columns from a DataFrame to clean PII.
        
        Args:
            df (pd.DataFrame): The DataFrame to clean.
            columns_to_drop (list of str): A list of column names to drop.
            
        Returns:
            pd.DataFrame: The cleaned DataFrame.
        """
        cleaned_df = df.drop(columns=columns_to_drop, errors='ignore')
        self.logger.info(f"Removed PII columns: {columns_to_drop}")
        return cleaned_df

    def combine_data(self, datasets):
        """
        Combines multiple DataFrames into a single DataFrame.
        
        Args:
            datasets (dict): A dictionary of DataFrames to combine.
            
        Returns:
            pd.DataFrame: The combined DataFrame.
        """
        if not datasets:
            return pd.DataFrame()
        
        combined_df = pd.concat(datasets.values(), ignore_index=True)
        self.logger.info(f"Combined datasets into a single frame with {len(combined_df)} rows.")
        return combined_df

    def run(self, sample_size=None):
        """
        Executes the full PII cleaning pipeline: loads, cleans, combines, and saves data.
        
        Args:
            sample_size (int, optional): The number of rows to load from each file.
            
        Returns:
            str: The path to the saved combined and cleaned CSV file.
        """
        try:
            dataset_files = [f'Variant {i}.csv' for i in range(1, 6)] + ['Base.csv']
            datasets = self.load_datasets(dataset_files, sample_size=sample_size)

            if not datasets:
                raise FileNotFoundError("No datasets were loaded.")

            pii_columns = ['name', 'email', 'address', 'phone_number']  # Example PII columns
            cleaned_datasets = {
                name: self.clean_data(df, pii_columns) for name, df in datasets.items()
            }

            combined_data = self.combine_data(cleaned_datasets)

            output_file = self.output_dir / "fraud_data_combined_clean.csv"
            combined_data.to_csv(output_file, index=False)
            self.logger.info(f"Saved combined cleaned data to {output_file}")
            
            return str(output_file)

        except Exception as e:
            self.logger.error(f"PII cleaning pipeline failed: {e}", exc_info=True)
            raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cleaner = PIICleaner()
    cleaner.run(sample_size=50000)
