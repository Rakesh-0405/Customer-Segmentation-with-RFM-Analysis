import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

print("🎯 RFM CUSTOMER SEGMENTATION ANALYSIS")
print("=" * 50)

def download_dataset():
    dataset_path = "Online_Retail.xlsx"
    if not os.path.exists(dataset_path):
        print("📥 Downloading dataset...")
        try:
            import requests
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
            response = requests.get(url)
            with open(dataset_path, 'wb') as f:
                f.write(response.content)
            print(f"✅ Dataset downloaded: {dataset_path}")
        except:
            print("❌ Download failed. Please manually download:")
            print("https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx")
            return None
    else:
        print(f"✅ Dataset found: {dataset_path}")
    return dataset_path

def load_and_clean_data(dataset_path):
    print("📊 Loading data...")
    df = pd.read_excel(dataset_path)
    print(f"Original data shape: {df.shape}")
    print("🧹 Cleaning data...")
    df_clean = df.dropna(subset=['CustomerID'])
    df_clean = df_clean.drop_duplicates()
    df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['UnitPrice'] > 0)]
    df_clean['CustomerID'] = df_clean['CustomerID'].astype(int)
    df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
    df_clean['TotalPrice'] = df_clean['Quantity'] * df_clean['UnitPrice']
    
    print(f"✅ Data cleaned! New shape: {df_clean.shape}")
    return df_clean

def calculate_rfm(df_clean):
    print("📈 Calculating RFM values...")
    
    snapshot_date = df_clean['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    rfm = df_clean.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days, 
        'InvoiceNo': 'nunique',    
        'TotalPrice': 'sum'       
    }).reset_index()
    
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    print(f"✅ RFM calculated for {len(rfm)} customers")
    return rfm

def perform_clustering(rfm):
    print("🔧 Preparing data for clustering...")
    
    rfm_data = rfm[['Recency', 'Frequency', 'Monetary']].copy()
    
    rfm_data['Frequency'] = np.log1p(rfm_data['Frequency'])
    rfm_data['Monetary'] = np.log1p(rfm_data['Monetary'])
    rfm_data['Recency'] = -rfm_data['Recency'] 
    
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_data)
    
    print("🎯 Applying K-means clustering...")
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(rfm_scaled)
    
    rfm['Cluster'] = clusters
    print("✅ Clustering completed!")
    return rfm

def visualize_results(rfm):
    print("📊 Creating visualizations...")
    
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('outputs/plots', exist_ok=True)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(rfm['Recency'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Recency Distribution')
    plt.xlabel('Days since last purchase')
    plt.ylabel('Number of Customers')
    
    plt.subplot(1, 3, 2)
    plt.hist(rfm['Frequency'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.title('Frequency Distribution')
    plt.xlabel('Number of purchases')
    plt.ylabel('Number of Customers')
    
    plt.subplot(1, 3, 3)
    plt.hist(rfm['Monetary'], bins=30, alpha=0.7, color='salmon', edgecolor='black')
    plt.title('Monetary Distribution')
    plt.xlabel('Total spending (£)')
    plt.ylabel('Number of Customers')
    
    plt.tight_layout()
    plt.savefig('outputs/plots/rfm_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize=(12, 8))
    
    colors = ['red', 'blue', 'green', 'orange']
    cluster_names = ['Low Value', 'Promising', 'High Value', 'At Risk']
    
    for cluster_num in range(4):
        cluster_data = rfm[rfm['Cluster'] == cluster_num]
        plt.scatter(cluster_data['Recency'], cluster_data['Frequency'], 
                   c=colors[cluster_num], label=cluster_names[cluster_num], alpha=0.6)
    
    plt.xlabel('Recency (days since last purchase)')
    plt.ylabel('Frequency (number of purchases)')
    plt.title('Customer Segments - Recency vs Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('outputs/plots/customer_segments.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize=(10, 6))
    cluster_counts = rfm['Cluster'].value_counts().sort_index()
    bars = plt.bar([f'Cluster {i}' for i in cluster_counts.index], 
                   cluster_counts.values, 
                   color=colors[:len(cluster_counts)])
    
    plt.title('Customer Distribution by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Customers')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.savefig('outputs/plots/cluster_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_insights(rfm):
    print("💡 Generating business insights...")
    print("=" * 50)
    
    total_customers = len(rfm)
    total_revenue = rfm['Monetary'].sum()
    
    print(f"📊 OVERALL METRICS:")
    print(f"• Total Customers: {total_customers:,}")
    print(f"• Total Revenue: £{total_revenue:,.2f}")
    print(f"• Average Customer Value: £{total_revenue/total_customers:,.2f}")
    
    print(f"\n👥 CLUSTER ANALYSIS:")
    cluster_stats = rfm.groupby('Cluster').agg({
        'Recency': ['mean', 'std'],
        'Frequency': ['mean', 'std'],
        'Monetary': ['mean', 'sum'],
        'CustomerID': 'count'
    }).round(2)
    
    cluster_stats.columns = ['Recency_Mean', 'Recency_Std', 
                            'Frequency_Mean', 'Frequency_Std',
                            'Monetary_Mean', 'Monetary_Sum',
                            'Customer_Count']
    
    cluster_stats['Revenue_Share'] = (cluster_stats['Monetary_Sum'] / total_revenue * 100).round(1)
    cluster_stats['Customer_Share'] = (cluster_stats['Customer_Count'] / total_customers * 100).round(1)
    
    print(cluster_stats)
    
    print(f"\n🎯 SEGMENT RECOMMENDATIONS:")
    segment_insights = {
        0: {"name": "LOW VALUE", "desc": "Infrequent buyers, haven't purchased recently", "action": "Win-back campaigns, special discounts"},
        1: {"name": "PROMISING", "desc": "Recent buyers with moderate frequency", "action": "Upsell opportunities, loyalty programs"},
        2: {"name": "HIGH VALUE", "desc": "Frequent buyers with high spending", "action": "VIP treatment, exclusive offers"},
        3: {"name": "AT RISK", "desc": "Previously good customers becoming inactive", "action": "Personalized outreach, win-back offers"}
    }
    
    for cluster in sorted(rfm['Cluster'].unique()):
        stats = segment_insights[cluster]
        count = len(rfm[rfm['Cluster'] == cluster])
        percentage = (count / total_customers) * 100
        avg_value = rfm[rfm['Cluster'] == cluster]['Monetary'].mean()
        
        print(f"\n🔹 {stats['name']} CUSTOMERS ({count} customers, {percentage:.1f}%):")
        print(f"   Description: {stats['desc']}")
        print(f"   Average Value: £{avg_value:,.2f}")
        print(f"   Recommended Action: {stats['action']}")

def save_results(rfm):
    print("💾 Saving results...")
    
    os.makedirs('outputs/reports', exist_ok=True)
    
    rfm.to_csv('outputs/reports/rfm_segmentation_results.csv', index=False)
    
    with open('outputs/reports/analysis_summary.txt', 'w') as f:
        f.write("RFM CUSTOMER SEGMENTATION ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total Customers Analyzed: {len(rfm):,}\n")
        f.write(f"Total Revenue: £{rfm['Monetary'].sum():,.2f}\n")
        f.write(f"Number of Segments: {rfm['Cluster'].nunique()}\n\n")
        
        f.write("SEGMENT DISTRIBUTION:\n")
        for cluster in sorted(rfm['Cluster'].unique()):
            count = len(rfm[rfm['Cluster'] == cluster])
            percentage = (count / len(rfm)) * 100
            avg_value = rfm[rfm['Cluster'] == cluster]['Monetary'].mean()
            f.write(f"Segment {cluster}: {count} customers ({percentage:.1f}%), Avg Value: £{avg_value:,.2f}\n")
    
    print("✅ Results saved to outputs/reports/")

def main():
    try:
        dataset_path = download_dataset()
        if not dataset_path:
            return
        
        df_clean = load_and_clean_data(dataset_path)
        
        rfm = calculate_rfm(df_clean)
        
        rfm = perform_clustering(rfm)
        
        visualize_results(rfm)
        
        generate_insights(rfm)
        
        save_results(rfm)
        
        print("\n🎉 RFM ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print("📁 Generated files:")
        print("   - outputs/plots/rfm_distributions.png")
        print("   - outputs/plots/customer_segments.png") 
        print("   - outputs/plots/cluster_distribution.png")
        print("   - outputs/reports/rfm_segmentation_results.csv")
        print("   - outputs/reports/analysis_summary.txt")
        print("\n🚀 This project is ready for your resume!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()