import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import logging
import joblib
from config.config import MODELS_PATH, N_CLUSTERS, RANDOM_STATE

logger = logging.getLogger(__name__)

class CustomerClustering:
    def __init__(self, n_clusters=N_CLUSTERS, random_state=RANDOM_STATE):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = None
    
    def prepare_data(self, rfm_df):
        """
        Prepare RFM data for clustering
        """
        logger.info("Preparing data for clustering...")
        
        rfm_log = rfm_df[['Recency', 'Frequency', 'Monetary']].copy()
        
        rfm_log['Frequency'] = np.log1p(rfm_log['Frequency'])
        rfm_log['Monetary'] = np.log1p(rfm_log['Monetary'])
        
        rfm_log['Recency'] = -rfm_log['Recency']
        
        rfm_scaled = self.scaler.fit_transform(rfm_log)
        rfm_scaled_df = pd.DataFrame(rfm_scaled, 
                                   columns=['Recency', 'Frequency', 'Monetary'],
                                   index=rfm_df.index)
        
        return rfm_scaled_df
    
    def find_optimal_clusters(self, rfm_scaled, max_k=10):
        """
        Find optimal number of clusters using elbow method and silhouette score
        """
        wcss = []
        silhouette_scores = []
        
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(rfm_scaled)
            
            wcss.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(rfm_scaled, cluster_labels))
        
        optimal_k = self.n_clusters 
        
        logger.info(f"Optimal number of clusters selected: {optimal_k}")
        
        return wcss, silhouette_scores, optimal_k
    
    def apply_kmeans(self, rfm_scaled):
        """
        Apply K-means clustering
        """
        logger.info(f"Applying K-means clustering with {self.n_clusters} clusters...")
        
        self.kmeans = KMeans(n_clusters=self.n_clusters, 
                           random_state=self.random_state, 
                           n_init=10)
        clusters = self.kmeans.fit_predict(rfm_scaled)
        
        logger.info("Clustering completed successfully")
        
        return clusters
    
    def save_model(self, filename='kmeans_model.pkl'):
        """
        Save the trained model
        """
        if self.kmeans is None:
            logger.error("No model trained yet")
            return
        
        filepath = os.path.join(MODELS_PATH, filename)
        joblib.dump({'kmeans': self.kmeans, 'scaler': self.scaler}, filepath)
        logger.info(f"Model saved to: {filepath}")
        return filepath