# 🎯 Customer Segmentation with RFM Analysis
A comprehensive data analysis project that segments customers using RFM (Recency, Frequency, Monetary) analysis and K-means clustering to provide actionable business insights for e-commerce optimization.

## 📊 Project Overview

This project analyzes customer purchasing behavior from an online retail dataset to identify distinct customer segments. Using RFM analysis and machine learning clustering, we categorize customers into strategic groups and provide targeted business recommendations for each segment.

**Key Achievements:**
- Analyzed **541,909 transactions** from **4,338 customers**
- Identified **4 distinct customer segments** contributing differently to revenue
- Discovered that **13% of customers generate 63% of total revenue**
- Provided **data-driven business strategies** for customer retention and growth

# 🪐 RFM Analysis & Customer Segmentation Architecture

## Architecture Overview

```mermaid
flowchart TD
    A[Data Preparation] --> B[RFM Feature Engineering]
    B --> C[Data Transformation]
    C --> D[RFM Analysis]
    D --> E[Clustering: K-Means]
    E --> F[Determining Optimal Clusters]
    F --> G[Segment Profiling & Analysis]
````


## 🔧 Methodology

### 1. Data Preparation

- **Data Cleaning:**  
  - Handled missing values  
  - Removed duplicates  
  - Filtered invalid entries

- **Feature Engineering:**  
  - Calculated Recency (days since last purchase)  
  - Calculated Frequency (number of purchases)  
  - Calculated Monetary (total spend per customer)

- **Data Transformation:**  
  - Applied log transformation  
  - Standardized features for clustering

---

### 2. RFM Analysis

- **Recency:**  
  - Days since customer's last purchase (lower is better)
- **Frequency:**  
  - Total number of purchases (higher is better)
- **Monetary:**  
  - Aggregate spending per customer (higher is better)

---

### 3. Machine Learning

- **K-means Clustering:**  
  - Unsupervised learning for customer segmentation  
- **Optimal Clusters:**  
  - Used elbow method and business insights to select K
- **Segment Profiling:**  
  - Analyzed mean R, F, M scores of each cluster  
  - Interpreted defining traits of each segment

  # 🪐 RFM Analysis & Customer Segmentation

---

## 🚀 Key Libraries Used

- **Pandas**: Data manipulation and analysis  
- **Scikit-learn**: Machine learning (K-means clustering)  
- **Matplotlib / Seaborn**: Data visualization  
- **NumPy**: Numerical computations  

---

## 📊 Sample Visualizations

- **RFM Distributions**: Histograms showing Recency, Frequency, Monetary value distributions  
- **Customer Segments**: Scatter plot visualizing clusters (Recency vs. Frequency)  
- **Cluster Distribution**: Bar chart showing customer count per segment  

---

## 🎯 Skills Demonstrated

### Technical Skills

- Programming: Python, Data Analysis, Machine Learning  
- Data Manipulation: Pandas, NumPy, Data Cleaning  
- ML Algorithms: K-means Clustering, Feature Engineering  
- Data Visualization: Matplotlib, Seaborn, Business Reporting  
- Statistical Analysis: RFM Methodology, Customer Segmentation  

### Business Skills

- Customer Analytics: RFM Analysis, Segmentation Strategy  
- Business Intelligence: Insight Generation, Actionable Recommendations  
- Data Storytelling: Visualization, Executive Reporting  
- Strategic Planning: Customer Retention, Revenue Optimization  

---

## 📈 Performance Metrics

- **Data Processing**: Handled 541,909 records efficiently  
- **Clustering Accuracy**: Clear separation of 4 distinct segments  
- **Business Impact**: Identified opportunities affecting 63% of revenue  
- **Scalability**: Methodology applicable to larger datasets  

---

## 🔮 Future Enhancements

- Time-series analysis for trend identification  
- Customer lifetime value prediction  
- Churn prediction modeling  
- A/B testing for recommendation strategies  
- Integration with real-time data pipelines  
