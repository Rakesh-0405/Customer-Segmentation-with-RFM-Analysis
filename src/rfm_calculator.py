import pandas as pd
import numpy as np
import logging
from datetime import datetime
from config.config import SNAPSHOT_DATE

logger = logging.getLogger(__name__)

class RFMCalculator:
    def __init__(self, snapshot_date=None):
        self.snapshot_date = snapshot_date or SNAPSHOT_DATE
    
    def calculate_rfm(self, data):
        """
        Calculate RFM values for each customer
        """
        logger.info("Calculating RFM values...")
        
        rfm = data.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (self.snapshot_date - x.max()).days,  
            'InvoiceNo': 'nunique',    
            'TotalPrice': 'sum'        
        }).reset_index()
        
        rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
        
        logger.info(f"RFM calculation completed. Shape: {rfm.shape}")
        logger.info("\nRFM Statistics:")
        logger.info(rfm[['Recency', 'Frequency', 'Monetary']].describe())
        
        return rfm
    
    def calculate_rfm_scores(self, rfm_df):
        """
        Calculate RFM scores (1-4) for each metric
        """
        rfm_scores = rfm_df.copy()
        
        rfm_scores['R_Score'] = pd.qcut(rfm_scores['Recency'], 4, labels=[4, 3, 2, 1])
        
        rfm_scores['F_Score'] = pd.qcut(rfm_scores['Frequency'], 4, labels=[1, 2, 3, 4])
        
        rfm_scores['M_Score'] = pd.qcut(rfm_scores['Monetary'], 4, labels=[1, 2, 3, 4])
        
        rfm_scores['R_Score'] = rfm_scores['R_Score'].astype(int)
        rfm_scores['F_Score'] = rfm_scores['F_Score'].astype(int)
        rfm_scores['M_Score'] = rfm_scores['M_Score'].astype(int)
        
        rfm_scores['RFM_Score'] = rfm_scores['R_Score'] + rfm_scores['F_Score'] + rfm_scores['M_Score']
        
        return rfm_scores