import numpy as np
from sklearn import datasets
import os
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


#Completed with the assistance of ChatGPT
def gmm_clustering(X, K, max_iter=100, tol=1e-6):

    N, D = X.shape

    
    pi = np.ones(K) / K

    
    indices = np.random.choice(N, K, replace=False)
    mu = X[indices]

    
    sigma = np.array([np.eye(D) for _ in range(K)])

    
    gamma = np.zeros((N, K))

    log_likelihoods = []

    for iteration in range(max_iter):
        
        for k in range(K):
            gamma[:, k] = pi[k] * multivariate_normal.pdf(X, mean=mu[k], cov=sigma[k])
        gamma /= gamma.sum(axis=1, keepdims=True)


        N_k = gamma.sum(axis=0)
        pi = N_k / N
        mu = (gamma.T @ X) / N_k[:, np.newaxis]

        for k in range(K):
            X_centered = X - mu[k]
            gamma_k = gamma[:, k][:, np.newaxis]
            sigma_k = (gamma_k * X_centered).T @ X_centered / N_k[k]
            sigma[k] = sigma_k + 1e-6 * np.eye(D)


        log_likelihood = np.sum(
            np.log(
                np.sum(
                    [pi[k] * multivariate_normal.pdf(X, mu[k], sigma[k]) for k in range(K)],
                    axis=0
                )
            )
        )
        log_likelihoods.append(log_likelihood)

        # Check for convergence
        if iteration > 0 and abs(log_likelihood - log_likelihoods[-2]) < tol:
            break

    # Assign labels based on the highest responsibility
    labels = np.argmax(gamma, axis=1)

    return labels, iteration + 1  # Return number of iterations
def kmeans_pp(X, n_clusters=3, max_iter=300, tol=1e-4, random_state=42):
    # Set random seed
    np.random.seed(random_state)
    n_samples, _ = X.shape

    
    centroids = []
    first_centroid_idx = np.random.randint(0, n_samples)
    centroids.append(X[first_centroid_idx])

    
    for _ in range(1, n_clusters):
        distances = np.array([
            min(np.linalg.norm(x - c) ** 2 for c in centroids) for x in X
        ])
        probabilities = distances / distances.sum()
        cumulative_probabilities = np.cumsum(probabilities)
        r = np.random.rand()
        next_centroid_idx = np.where(cumulative_probabilities >= r)[0][0]
        centroids.append(X[next_centroid_idx])

    centroids = np.array(centroids)

   
    for iteration in range(max_iter):
        
        clusters = np.argmin(
            np.linalg.norm(X[:, np.newaxis] - centroids, axis=2), axis=1
        )

        
        new_centroids = np.array([
            X[clusters == k].mean(axis=0) if len(X[clusters == k]) > 0 else centroids[k]
            for k in range(n_clusters)
        ])

        if np.all(np.linalg.norm(new_centroids - centroids, axis=1) < tol):
            break

        centroids = new_centroids

    return centroids, clusters, iteration + 1
def k_means(X, K, max_iters=100):

    # Randomly initialize centroids
    centroids = X[np.random.choice(len(X), K, replace=False)]
    for i in range(max_iters):
        # Assignment step
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        # Update step
        new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(K)])
        
        # Check for convergence
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return centroids, labels, i+1  # Return the number of iterations


def run_experiments():
    n_samples_per_blob = 100
    n_features_list = [2, 3, 4, 5, 6, 7, 8]
    n_trials = 10
    cluster_std = 0.5
    K = 3  # Number of clusters

    convergence_times = {n_features: [] for n_features in n_features_list}

    for n_features in n_features_list:
        # Define centers for Gaussian blobs
        centers = n_features # Sufficiently separated centers
        for trial in range(n_trials):
            # Generate data
            X, _ = datasets.make_blobs(n_samples=10000, n_features=trial+1, centers=3, cluster_std=1.0, center_box=(-10.0, 10.0), shuffle=True, random_state=None, return_centers=False)

            # Run K-means
            _, _, iterations = k_means(X, K)
            
            # Record convergence time
            convergence_times[n_features].append(iterations)
    
    return convergence_times
def run_experiments_gmm():
    n_samples_per_blob = 100
    n_features_list = [2, 3, 4, 5, 6, 7, 8]
    n_trials = 10
    K = 3  # Number of clusters

    convergence_times = {n_features: [] for n_features in n_features_list}

    for n_features in n_features_list:
        for trial in range(n_trials):
            # Generate data
            X, _ = datasets.make_blobs(
                n_samples=10000,
                n_features=n_features,
                centers=K,
                cluster_std=1.0,
                center_box=(-10.0, 10.0),
                shuffle=True,
                random_state=None,
                return_centers=False
            )

            # Run GMM
            _, iterations = gmm_clustering(X, K)

            # Record convergence time
            convergence_times[n_features].append(iterations)

    return convergence_times
def run_experiments_kmeanspp():
    n_samples_per_blob = 100
    n_features_list = [2, 3, 4, 5, 6, 7, 8]
    n_trials = 10
    K = 3  # Number of clusters

    convergence_times = {n_features: [] for n_features in n_features_list}

    for n_features in n_features_list:
        for trial in range(n_trials):
            # Generate data
            X, _ = datasets.make_blobs(
                n_samples=10000,
                n_features=n_features,
                centers=K,
                cluster_std=1.0,
                center_box=(-10.0, 10.0),
                shuffle=True,
                random_state=None,
                return_centers=False
            )

            # Run KMeans++ algorithm
            _, _, iterations = kmeans_pp(X, n_clusters=K)

            # Record convergence time
            convergence_times[n_features].append(iterations)

    return convergence_times

def analyze_results(convergence_times):
    for n_features, times in convergence_times.items():
        mean_iterations = np.mean(times)
        std_iterations = np.std(times)
        print(f"Dimensionality: {n_features}D")
        print(f"Average Iterations to Converge: {mean_iterations:.2f}")
        print(f"Standard Deviation: {std_iterations:.2f}\n")
if __name__ == "__main__":
    print("K-means with random initialization:")
    convergence_times = run_experiments()
    analyze_results(convergence_times)

    print("K-means++ initialization:")
    convergence_times_pp = run_experiments_kmeanspp()
    analyze_results(convergence_times_pp)

    print("GMM clustering:")
    convergence_times_gmm = run_experiments_gmm()
    analyze_results(convergence_times_gmm)
