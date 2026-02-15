import time
import numpy as np
from pyspark.sql import SparkSession
import sys
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler


def closest_cluster(point, centroids):
    distances = np.sum((centroids - point) ** 2, axis=1) #axes=1 to sum across features
    return np.argmin(distances)

def kmeans_parallel(rdd, k, max_iter=20, eps=1e-4):
    curr_centroids = np.array(rdd.takeSample(False, k)) # False for no replacement

    for i in range(max_iter):
        br_centroids = rdd.context.broadcast(curr_centroids)

        # MAP (cluster assignment)
        cluster_assignments = rdd.map(lambda p: (closest_cluster(p, br_centroids.value), (p, 1))) # emit (cluster_idx, (point, count)) for each point

        # REDUCE (sum points and counts)
        cluster_sums = cluster_assignments.reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1])) # x and y are (sum_points, count)
        new_stats = cluster_sums.collectAsMap() # {cluster_idx: (sum_points, count)}

        new_centroids = np.copy(curr_centroids)

        for idx, (p_sum, count) in new_stats.items():
            new_centroids[idx] = p_sum / count

        shift = np.linalg.norm(curr_centroids - new_centroids)
        curr_centroids = new_centroids

        br_centroids.unpersist()

        if shift < eps:
            print(f"Converged at iteration {i+1}")
            break

    return curr_centroids


def run_test(n_points=1000000, dim=2, k=3, n_cores=4):
    spark = (
        SparkSession.builder.master(f"local[{n_cores}]") 
        .appName(f"KMeans_Scale_{n_cores}") 
        .config("spark.driver.memory", "16g") 
        .getOrCreate()
    )

    np.random.seed(21)

    rdd = spark.sparkContext.range(n_points, numSlices=n_cores)
    rdd = rdd.map(lambda _: np.random.randn(dim)) # Generate random points 
    rdd = rdd.cache() # Force caching
    rdd.count()  

    print(f"--- Spark K-means: {n_cores} CORES ---")
    print("Partitions in RDD:", rdd.getNumPartitions())


    start = time.perf_counter()
    final_centroids = kmeans_parallel(rdd, k=k)
    end = time.perf_counter()

    print(f"CORES: {n_cores}")
    print(f"TIME: {end-start:.4f}")
    print(f"CENTROIDS: {final_centroids}")

    spark.stop()

def real_dataset_test(df_scaled, k, n_cores):
    spark = (
        SparkSession.builder
        .master(f"local[{n_cores}]")
        .appName(f"KMeans_Scale_{n_cores}")
        .config("spark.driver.memory", "16g")
        .getOrCreate()
    )

    rdd = spark.sparkContext.parallelize(df_scaled, numSlices=n_cores)
    rdd = rdd.map(lambda x: np.array(x, dtype=float)) # numpy arrays for distance calculations
    rdd.cache()
    rdd.count()  # Force caching

    print(f"--- Spark K-Means: {n_cores} CORES ---")
    print("Partitions in RDD:", rdd.getNumPartitions())
    

    start = time.perf_counter()
    final_centroids = kmeans_parallel(rdd, k=k)
    end = time.perf_counter()

    print(f"CORES: {n_cores}")
    print(f"TIME: {end-start:.4f}")
    print(f"CENTROIDS: {final_centroids}")

    spark.stop()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        n_cores = int(sys.argv[1])
    else:
        print("Error: specify number of cores (e.g.: python k-means_parallel.py 4)")
        sys.exit(1)

    # run_test(n_cores=n_cores)
    df = pd.read_csv('./data/final/gdp_co2_emissions.csv')
    df = df[df['Year'] == 2024]
    df = df[['log (GDP pc)', 'log (CO2 pc)']].dropna()

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    real_dataset_test(df_scaled, k=3, n_cores=n_cores)

    scikit_kmeans = KMeans(n_clusters=3, init='random', n_init=1, random_state=21)
    scikit_kmeans.fit(df_scaled)
    print("Comparing with Scikit-Learn centroids...")
    print(f"Scikit KMeans Centroids:\n{scikit_kmeans.cluster_centers_}")




