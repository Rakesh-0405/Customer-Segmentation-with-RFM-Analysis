import pandas as pd
import os
from config.config import RAW_DATA_PATH
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_raw_data():
    """
    Load the raw Online Retail dataset
    Returns: pandas DataFrame
    """
    try:
        logger.info(f"Loading data from: {RAW_DATA_PATH}")
        df = pd.read_excel(RAW_DATA_PATH)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found at {RAW_DATA_PATH}")
        logger.info("Please download the dataset from UCI Repository and place it in data/raw/")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def explore_data(df):
    """
    Perform basic data exploration
    """
    logger.info("=== Data Exploration ===")
    logger.info(f"Dataset shape: {df.shape}")
    logger.info("\nData types:")
    logger.info(df.dtypes)
    logger.info("\nMissing values:")
    logger.info(df.isnull().sum())
    logger.info("\nBasic statistics:")
    logger.info(df.describe())
    
    return df