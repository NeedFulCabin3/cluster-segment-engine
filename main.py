import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def load_customer_data(file_path=None):
    """Loads customer data from CSV or generates synthetic data if none provided."""
    if file_path:
        try:
            df = pd.read_csv(file_path)
            print(f"Successfully loaded data from {file_path}")
            return df
        except Exception as e:
            print(f"Error loading file: {e}. Generating synthetic data instead.")
    
    # Synthetic dataset simulating Annual Income vs Spending Score
    print("Generating synthetic customer dataset...")
    np.random.seed(42)
    
    income = np.concatenate([
        np.random.normal(30, 8, 100),
        np.random.normal(85, 12, 100),
        np.random.normal(55, 10, 100),
        np.random.normal(25, 6, 100),
        np.random.normal(85, 10, 100)
    ])
    spending = np.concatenate([
        np.random.normal(80, 10, 100),
        np.random.normal(80, 10, 100),
        np.random.normal(50, 10, 100),
        np.random.normal(20, 8, 100),
        np.random.normal(20, 8, 100)
    ])
    
    return pd.DataFrame({
        'Annual_Income': income,
        'Spending_Score': spending
    })

def apply_elbow_method(X):
    """Calculates inertia for different values of K to find the optimal cluster count."""
    inertias = []
    K_range = range(1, 11)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        
    # Plotting the Elbow Curve
    plt.figure(figsize=(8, 5))
    plt.plot(K_range, inertias, marker='o', linestyle='-', color='b')
    plt.title('Elbow Method For Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.show()

def main():
    file_path = input("Enter path to your customer CSV file (or press Enter to use synthetic data): ").strip()
    df = load_customer_data(file_path if file_path else None)
    
    # Select features for clustering
    feature_cols = ['Annual_Income', 'Spending_Score']
    if not all(col in df.columns for col in feature_cols):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            feature_cols = numeric_cols[:2]
            print(f"Using columns {list(feature_cols)} for clustering.")
        else:
            raise ValueError("Dataset must contain at least two numeric columns.")
            
    X = df[feature_cols].values
    
    # Standardize features for K-Means distance calculations
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("\nAnalyzing optimal cluster count using the Elbow Method...")
    apply_elbow_method(X_scaled)
    
    # Get user input for optimal K based on the plot
    try:
        optimal_k = int(input("\nEnter the optimal K observed from the elbow plot (e.g., 5): "))
    except ValueError:
        print("Invalid input. Defaulting to K=5.")
        optimal_k = 5
        
    # Fit K-Means model with optimal K
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df['Cluster'] = clusters
    
    # Print cluster summary statistics
    print(f"\n--- Cluster Summary Statistics (K={optimal_k}) ---")
    print(df.groupby('Cluster')[list(feature_cols)].mean())
    
    # Plot final clusters and centroids
    plt.figure(figsize=(9, 6))
    plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', s=50, alpha=0.8)
    
    centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
    plt.scatter(centers_original[:, 0], centers_original[:, 1], c='red', marker='X', s=200, label='Centroids')
    
    plt.title(f'Customer Segmentation (K = {optimal_k})')
    plt.xlabel(feature_cols[0])
    plt.ylabel(feature_cols[1])
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()