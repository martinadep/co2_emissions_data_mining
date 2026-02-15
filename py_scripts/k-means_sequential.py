import time
import numpy as np

def closest_idx(p, centroids):
    distances = np.sum((centroids - p[:, np.newaxis])**2, axis=2)
    return np.argmin(distances, axis=1)

def kmeans_sequential(data, k, max_iter=20, eps=1e-4):
    n_points, _ = data.shape
    
    indices = np.random.choice(n_points, k, replace=False)
    centroids = data[indices]

    for i in range(max_iter):
        # MAP (cluster assignment)
        labels = closest_idx(data, centroids)

        new_centroids = np.zeros_like(centroids)
        
        # REDUCE (update)
        for idx in range(k):
            cluster_points = data[labels == idx]
            
            if len(cluster_points) > 0:
                new_centroids[idx] = np.mean(cluster_points, axis=0)
            else:
                new_centroids[idx] = centroids[idx]

        shift = np.linalg.norm(centroids - new_centroids)
        centroids = new_centroids

        if shift < eps:
            print(f"Converged at iteration {i+1}")
            break

    return centroids

def run_test():
    np.random.seed(42)
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

if __name__ == "__main__":
    run_test()