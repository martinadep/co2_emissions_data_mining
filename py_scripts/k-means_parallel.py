import time
import numpy as np
from pyspark.sql import SparkSession
import os


def closest_idx(p, centroids):
    return np.argmin(np.sum((centroids - p) ** 2, axis=1))


def kmeans_optimized(rdd, k, max_iter=20, eps=1e-4):
    centroids = np.array(rdd.takeSample(False, k))

    for _ in range(max_iter):
        br_centroids = rdd.context.broadcast(centroids)

        new_stats = (
            rdd.map(lambda p: (closest_idx(p, br_centroids.value), (p, 1)))
               .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))
               .collectAsMap()
        )

        new_centroids = np.copy(centroids)

        for idx, (p_sum, count) in new_stats.items():
            new_centroids[idx] = p_sum / count

        shift = np.linalg.norm(centroids - new_centroids)
        centroids = new_centroids

        br_centroids.unpersist()

        if shift < eps:
            break

    return centroids


def run_scalability_test():
    n_cores = int(os.environ.get("PBS_NP", 1))
    print("Partitions:", rdd.getNumPartitions())
    print("Default parallelism:", spark.sparkContext.defaultParallelism)
    
    np.random.seed(42)
    n_points = 1_000_000
    dim = 2

    spark = (
        SparkSession.builder
        .master(f"local[{n_cores}]")
        .appName(f"KMeans_Scale_{n_cores}")
        .config("spark.driver.memory", "16g")
        .getOrCreate()
    )

    rdd = spark.sparkContext.range(n_points, numSlices=2*n_cores) \
    .map(lambda _: np.random.randn(dim).astype("f")) \
    .cache()

    start = time.perf_counter()
    kmeans_optimized(rdd, k=3)
    end = time.perf_counter()

    print(f"CORES: {n_cores}")
    print(f"TIME: {end-start:.4f}")

    spark.stop()


if __name__ == "__main__":
    run_scalability_test()
