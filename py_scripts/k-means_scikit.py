import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler

def run_test(n_points=1000000, dim=2, k=3):
    np.random.seed(12)

    print("--- Scikit-Learn K-Means ---")
    print(f"Generating {n_points} points...")

    data = np.random.randn(n_points, dim).astype("f")

    kmeans = KMeans(n_clusters=k, init="random", n_init=1, random_state=12)

    start = time.perf_counter()
    kmeans.fit(data)
    end = time.perf_counter()

    print(f"TIME: {end-start:.4f} seconds")
    print(f"Final Centroids:\n{kmeans.cluster_centers_}")

def real_dataset_test(df_scales, k=3):
    kmeans = KMeans(n_clusters=k, init="random", n_init=1, random_state=12)

    start = time.perf_counter()
    kmeans.fit(df_scales)
    end = time.perf_counter()

    print(f"Scikit K-Means")
    print(f"TIME: {end-start:.4f} seconds")
    print(f"Final Centroids:\n{kmeans.cluster_centers_}")
    


if __name__ == "__main__":
    # run_test()
    
    df = pd.read_csv('./data/final/gdp_co2_emissions.csv')
    df = df[df['Year'] == 2024]
    df = df[['log (GDP pc)', 'log (CO2 pc)']].dropna()

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    real_dataset_test(df_scaled)
