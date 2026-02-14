import time
import numpy as np
from sklearn.cluster import KMeans

def run_sequential_test():
    np.random.seed(42)
    n_points = 1000000
    dim = 2
    k = 3
    max_iter = 20
    eps = 1e-4

    print("--- DEBUG TEST: SEQUENTIAL (Scikit-Learn) ---")
    print(f"Generating {n_points} points...")


    data = np.random.randn(n_points, dim).astype("f")

    kmeans = KMeans(
        n_clusters=k, 
        init='random', 
        n_init=1, 
        max_iter=max_iter, 
        tol=eps, 
        random_state=42
    )

    start = time.perf_counter()
    kmeans.fit(data)
    end = time.perf_counter()

    print(f"CORES: 1 (Sequential)")
    print(f"TIME: {end-start:.4f} seconds")
    print(f"Final Centroids:\n{kmeans.cluster_centers_}")

if __name__ == "__main__":
    run_sequential_test()
