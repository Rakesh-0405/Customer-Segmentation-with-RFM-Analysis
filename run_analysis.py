"""
Main script for RFM Customer Segmentation Analysis
Run this file to execute the complete analysis
"""

import logging
import pandas as pd
from src.data_loader import load_raw_data, explore_data
from src.data_cleaner import DataCleaner
from src.rfm_calculator import RFMCalculator
from src.clustering import CustomerClustering
from src.visualizer import RFMVisualizer
from config.config import REPORTS_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main function to run the complete RFM analysis"""
    logger.info("Starting RFM Customer Segmentation Analysis")
    
    try:
        
        logger.info("Step 1: Loading data...")
        raw_data = load_raw_data()
        explore_data(raw_data)
        
        logger.info("Step 2: Cleaning data...")
        cleaner = DataCleaner()
        cleaned_data = cleaner.clean_data(raw_data)
        cleaner.save_cleaned_data(cleaned_data)
        
        logger.info("Step 3: Calculating RFM values...")
        rfm_calc = RFMCalculator()
        rfm_df = rfm_calc.calculate_rfm(cleaned_data)
        rfm_with_scores = rfm_calc.calculate_rfm_scores(rfm_df)
        
        logger.info("Step 4: Performing clustering...")
        clustering = CustomerClustering()
        rfm_scaled = clustering.prepare_data(rfm_df)
        clusters = clustering.apply_kmeans(rfm_scaled)
        
        rfm_df['Cluster'] = clusters
        rfm_with_scores['Cluster'] = clusters
        
        logger.info("Step 5: Creating visualizations...")
        visualizer = RFMVisualizer()
        visualizer.plot_rfm_distributions(rfm_df)
        visualizer.plot_clusters_2d(rfm_df)
        
        logger.info("Step 6: Generating business insights...")
        generate_business_insights(rfm_df)
        
        logger.info("Step 7: Saving results...")
        save_final_results(rfm_df, rfm_with_scores)
        
        logger.info("RFM Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        raise

def generate_business_insights(rfm_df):
    """Generate and display business insights"""
    logger.info("\n" + "="*60)
    logger.info("BUSINESS INSIGHTS")
    logger.info("="*60)
    
    cluster_analysis = rfm_df.groupby('Cluster').agg({
        'Recency': ['mean', 'std'],
        'Frequency': ['mean', 'std'],
        'Monetary': ['mean', 'std'],
        'CustomerID': 'count'
    }).round(2)
    
    logger.info("\nCluster Profiles:")
    logger.info(cluster_analysis)
    
    total_customers = len(rfm_df)
    total_revenue = rfm_df['Monetary'].sum()
    
    logger.info(f"\nTotal Customers: {total_customers:,}")
    logger.info(f"Total Revenue: £{total_revenue:,.2f}")
    logger.info(f"Average Customer Value: £{total_revenue/total_customers:,.2f}")

def save_final_results(rfm_df, rfm_with_scores):
    """Save final results to files"""
    rfm_df.to_csv(f'{REPORTS_PATH}/rfm_segmentation_results.csv', index=False)
    rfm_with_scores.to_csv(f'{REPORTS_PATH}/rfm_with_scores.csv', index=False)
    
    with open(f'{REPORTS_PATH}/analysis_summary.txt', 'w') as f:
        f.write("RFM Customer Segmentation Analysis Summary\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total Customers Analyzed: {len(rfm_df)}\n")
        f.write(f"Total Revenue: £{rfm_df['Monetary'].sum():,.2f}\n")
        f.write(f"Number of Segments: {rfm_df['Cluster'].nunique()}\n\n")
        
        f.write("Segment Distribution:\n")
        segment_dist = rfm_df['Cluster'].value_counts().sort_index()
        for segment, count in segment_dist.items():
            percentage = (count / len(rfm_df)) * 100
            f.write(f"Segment {segment}: {count} customers ({percentage:.1f}%)\n")

if __name__ == "__main__":
    main()