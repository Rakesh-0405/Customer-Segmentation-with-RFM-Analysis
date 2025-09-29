import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from config.config import PLOTS_PATH, PLOT_STYLE, COLOR_PALETTE, FIG_SIZE

plt.style.use(PLOT_STYLE)
sns.set_palette(COLOR_PALETTE)

class RFMVisualizer:
    def __init__(self):
        self.fig_size = FIG_SIZE
    
    def plot_rfm_distributions(self, rfm_df, save_plot=True):
        """
        Plot distributions of RFM values
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('RFM Distributions', fontsize=16)
        
        axes[0, 0].hist(rfm_df['Recency'], bins=50, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Recency Distribution')
        axes[0, 0].set_xlabel('Days since last purchase')
        axes[0, 0].set_ylabel('Number of Customers')
        
        axes[0, 1].hist(rfm_df['Frequency'], bins=50, alpha=0.7, color='lightgreen')
        axes[0, 1].set_title('Frequency Distribution')
        axes[0, 1].set_xlabel('Number of purchases')
        axes[0, 1].set_ylabel('Number of Customers')
        
        axes[1, 0].hist(rfm_df['Monetary'], bins=50, alpha=0.7, color='salmon')
        axes[1, 0].set_title('Monetary Distribution')
        axes[1, 0].set_xlabel('Total spending')
        axes[1, 0].set_ylabel('Number of Customers')
        
        fig.delaxes(axes[1, 1])
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(PLOTS_PATH, 'rfm_distributions.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {plot_path}")
        
        plt.show()
    
    def plot_clusters_3d(self, rfm_df, save_plot=True):
        """
        Create 3D scatter plot of clusters (if you want to get fancy)
        """
        try:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            scatter = ax.scatter(rfm_df['Recency'], 
                               rfm_df['Frequency'], 
                               rfm_df['Monetary'], 
                               c=rfm_df['Cluster'], 
                               cmap='viridis', 
                               alpha=0.6)
            
            ax.set_xlabel('Recency')
            ax.set_ylabel('Frequency')
            ax.set_zlabel('Monetary')
            ax.set_title('3D Cluster Visualization')
            plt.colorbar(scatter)
            
            if save_plot:
                plot_path = os.path.join(PLOTS_PATH, '3d_clusters.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            
            plt.show()
        except ImportError:
            print("3D plotting not available, creating 2D plots instead")
            self.plot_clusters_2d(rfm_df, save_plot)
    
    def plot_clusters_2d(self, rfm_df, save_plot=True):
        """
        Create 2D scatter plots for cluster visualization
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Customer Segments Visualization', fontsize=16)
        
        scatter1 = axes[0, 0].scatter(rfm_df['Recency'], rfm_df['Frequency'], 
                                    c=rfm_df['Cluster'], cmap='viridis', alpha=0.6)
        axes[0, 0].set_xlabel('Recency')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Recency vs Frequency')
        plt.colorbar(scatter1, ax=axes[0, 0])
        
        scatter2 = axes[0, 1].scatter(rfm_df['Recency'], rfm_df['Monetary'], 
                                    c=rfm_df['Cluster'], cmap='viridis', alpha=0.6)
        axes[0, 1].set_xlabel('Recency')
        axes[0, 1].set_ylabel('Monetary')
        axes[0, 1].set_title('Recency vs Monetary')
        plt.colorbar(scatter2, ax=axes[0, 1])
        
        scatter3 = axes[1, 0].scatter(rfm_df['Frequency'], rfm_df['Monetary'], 
                                    c=rfm_df['Cluster'], cmap='viridis', alpha=0.6)
        axes[1, 0].set_xlabel('Frequency')
        axes[1, 0].set_ylabel('Monetary')
        axes[1, 0].set_title('Frequency vs Monetary')
        plt.colorbar(scatter3, ax=axes[1, 0])
        
        cluster_counts = rfm_df['Cluster'].value_counts().sort_index()
        axes[1, 1].bar(cluster_counts.index, cluster_counts.values, color='lightcoral')
        axes[1, 1].set_xlabel('Cluster')
        axes[1, 1].set_ylabel('Number of Customers')
        axes[1, 1].set_title('Cluster Distribution')
        for i, v in enumerate(cluster_counts.values):
            axes[1, 1].text(i, v, str(v), ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(PLOTS_PATH, 'cluster_visualization.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Cluster visualization saved to: {plot_path}")
        
        plt.show()