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

# STRONG SCALABILITY
def run_scalability_test():
    n_points = 100000
    dim = 2
    data_np = np.random.randn(n_points, dim)
    
    core_configs = [1, 2, 4] 
    results = {}

    for n in core_configs:
        print(f"\n--- Test with {n} Cores ---")
        spark = SparkSession.builder \
            .master(f"local[{n}]") \
            .appName("ScalabilityTest") \
            .getOrCreate()
        
        rdd = spark.sparkContext.parallelize(data_np).cache()
        rdd.count() # Force caching
        
        start = time.time()
        kmeans_optimized(rdd, k=3)
        end = time.time()
        
        duration = end - start
        results[n] = duration
        print(f"Time taken: {duration:.2f} seconds")
        
        spark.stop()

    print("\n--- SPEEDUP RESULTS ---")
    for n in core_configs:
        speedup = results[core_configs[0]] / results[n]
        print(f"Core {n}: Speedup = {speedup:.2f}x")

if __name__ == "__main__":
    run_scalability_test()