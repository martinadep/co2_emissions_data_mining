import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler

def closest_cluster(point, centroids):
    distances = np.sum((centroids - point) ** 2, axis=1) #axes=1 to sum across features
    return np.argmin(distances)

def kmeans_sequential(data, k, max_iter=20, eps=1e-4):
    n_points, dim = data.shape
    
    indices = np.random.choice(n_points, k, replace=False)
    centroids = data[indices]

    for i in range(max_iter):
        # MAP (cluster assignment)
        labels = np.array([closest_cluster(p, centroids) for p in data])

        new_centroids = np.zeros_like(centroids)
        
        # REDUCE (update)
        for idx in range(k):
            cluster_points = data[labels == idx] # mask to select points in cluster idx
            
            if len(cluster_points) > 0:
                new_centroids[idx] = np.mean(cluster_points, axis=0)
            else:
                new_centroids[idx] = centroids[idx]

        shift = np.linalg.norm(centroids - new_centroids) # norm of centroid shifts
        centroids = new_centroids

        if shift < eps:
            print(f"Converged at iteration {i+1}")
            break

    return centroids

def run_test():
    np.random.seed(21)
    n_points = 1000000
    dim = 2
    k = 3

    print(f"--- DEBUG TEST: SEQUENTIAL ---")
    print(f"Generating {n_points} points...")
    data = np.random.randn(n_points, dim).astype("f")

    start = time.perf_counter()
    final_centroids = kmeans_sequential(data, k=k)
    end = time.perf_counter()

    print(f"CORES: 1 (Sequential)")
    print(f"TIME: {end-start:.4f}")
    print(f"Final Centroids:\n{final_centroids}")

    scikit_kmeans = KMeans(n_clusters=k, init='random', n_init=1, random_state=21)
    scikit_kmeans.fit(data)
    print(f"Scikit KMeans Centroids:\n{scikit_kmeans.cluster_centers_}")


def real_dataset_test(df_scaled):
    start = time.perf_counter()
    final_centroids = kmeans_sequential(df_scaled, k=3)
    end = time.perf_counter()

    print(f"TIME: {end-start:.4f} seconds")
    print(f"Final Centroids:\n{final_centroids}")


if __name__ == "__main__":
    # run_test()

    df = pd.read_csv('./data/final/gdp_co2_emissions.csv')
    df = df[df['Year'] == 2024]
    df = df[['log (GDP pc)', 'log (CO2 pc)']].dropna()

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    real_dataset_test(df_scaled)

    scikit_kmeans = KMeans(n_clusters=3, init='random', n_init=1, random_state=12)
    scikit_kmeans.fit(df_scaled)

    print(f"Scikit KMeans Centroids:\n{scikit_kmeans.cluster_centers_}")