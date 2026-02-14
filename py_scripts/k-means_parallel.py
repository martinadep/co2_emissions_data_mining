import time
import numpy as np
from pyspark.sql import SparkSession
import os
import sys

def kmeans_optimized(rdd, k, max_iter=20, eps=1e-4):
    centroids = np.array(rdd.takeSample(False, k))
    
    for i in range(max_iter):
        br_centroids = rdd.context.broadcast(centroids) # broadcast centroids to avoid shuffling
        
        # Map-Reduce: calcultae (sum_points, count) for every cluster
        new_stats = rdd.map(lambda p: (
            np.argmin([np.linalg.norm(p - c) for c in br_centroids.value]), # find closest centroid index
            (p, 1)
        )).reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1])).collectAsMap()
        
        new_centroids = np.copy(centroids) # centroids update
        for idx, (p_sum, count) in new_stats.items():
            new_centroids[idx] = p_sum / count
            
        shift = np.linalg.norm(centroids - new_centroids) # convergence check
        centroids = new_centroids
        
        if shift < eps:
            break
            
    return centroids


def run_scalability_test():
    # Cores assigned by PBS scheduler, default to 1 if not set
    n_cores = int(os.environ.get('PBS_NUM_PPN', 1))
    
    n_points = 1000000 
    dim = 2
    data_np = np.random.randn(n_points, dim).astype('f')
    
    spark = SparkSession.builder \
        .master(f"local[{n_cores}]") \
        .appName(f"KMeans_Scale_{n_cores}") \
        .config("spark.driver.memory", "16g") \
        .getOrCreate()
    
    rdd = spark.sparkContext.parallelize(data_np, numSlices=n_cores).cache()
    rdd.count()
    
    start = time.time()
    kmeans_optimized(rdd, k=3)
    end = time.time()
    
    print(f"\n--- FINAL RESULT ---")
    print(f"CORES: {n_cores}")
    print(f"TIME: {end - start:.4f} seconds")
    print(f"------------------------\n")
    
    spark.stop()

if __name__ == "__main__":
    run_scalability_test()