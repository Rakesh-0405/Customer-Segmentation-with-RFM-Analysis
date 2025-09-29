import pandas as pd
import numpy as np
import logging
from config.config import DATA_PROCESSED_PATH

logger = logging.getLogger(__name__)

class DataCleaner:
    def __init__(self):
        self.cleaning_steps = []
    
    def clean_data(self, df):
        """
        Clean the raw dataset for RFM analysis
        """
        logger.info("Starting data cleaning process...")
        
        data = df.copy()
        self.cleaning_steps.append("Created copy of original data")
        
        initial_shape = data.shape
        data = data.dropna(subset=['CustomerID'])
        removed_rows = initial_shape[0] - data.shape[0]
        self.cleaning_steps.append(f"Removed {removed_rows} rows with missing CustomerID")
        
        initial_shape = data.shape
        data = data.drop_duplicates()
        removed_duplicates = initial_shape[0] - data.shape[0]
        self.cleaning_steps.append(f"Removed {removed_duplicates} duplicate rows")
        
        initial_shape = data.shape
        data = data[(data['Quantity'] > 0) & (data['UnitPrice'] > 0)]
        removed_invalid = initial_shape[0] - data.shape[0]
        self.cleaning_steps.append(f"Removed {removed_invalid} rows with invalid quantities/prices")
        
        data['CustomerID'] = data['CustomerID'].astype(int)
        data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
        
        data['TotalPrice'] = data['Quantity'] * data['UnitPrice']
        self.cleaning_steps.append("Created TotalPrice column")
        
        logger.info(f"Data cleaning completed. Final shape: {data.shape}")
        logger.info("Cleaning steps applied:")
        for step in self.cleaning_steps:
            logger.info(f"  - {step}")
        
        return data
    
    def save_cleaned_data(self, df, filename='cleaned_retail_data.csv'):
        """
        Save cleaned data to processed folder
        """
        filepath = os.path.join(DATA_PROCESSED_PATH, filename)
        df.to_csv(filepath, index=False)
        logger.info(f"Cleaned data saved to: {filepath}")
        return filepath