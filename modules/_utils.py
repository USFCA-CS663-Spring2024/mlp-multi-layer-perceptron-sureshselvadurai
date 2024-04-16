# _utils.py
import numpy as np


def initialize_centroids(X, n_clusters, init, random_state):
    """Initialize centroids using k-means++ or random selection.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Training instances.

    n_clusters : int
        The number of clusters.

    init : {'k-means++', 'random'}
        Method for initialization.

    random_state : int or None
        Random state.

    Returns:
    --------
    centroids : array, shape (n_clusters, n_features)
        Initial cluster centers.

    """
    if init == 'random':
        rng = np.random.default_rng(random_state)
        centroids_indices = rng.choice(X.shape[0], size=n_clusters, replace=False)
        centroids = X[centroids_indices]
    elif init == 'k-means++':
        rng = np.random.default_rng(random_state)
        centroids = [X[rng.choice(X.shape[0])]]
        for _ in range(1, n_clusters):
            distances = np.array([min(np.linalg.norm(x - c) ** 2 for c in centroids) for x in X])
            probs = distances / np.sum(distances)
            cumulative_probs = np.cumsum(probs)
            r = rng.random()
            index = np.searchsorted(cumulative_probs, r)
            centroids.append(X[index])
        centroids = np.array(centroids)
    else:
        raise ValueError("Invalid value for init. Supported values are 'random' and 'k-means++'.")
    return centroids


def balance_clusters(X, labels, centroids, balanced):
    """Balance cluster sizes by distributing data counts equally among clusters.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Training instances.

    labels : array, shape (n_samples,)
        Labels of each point.

    centroids : array, shape (n_clusters, n_features)
        Cluster centers.

    balanced : bool
        If True, ensures balanced cluster sizes by distributing data counts equally among clusters.

    Returns:
    --------
    new_labels : array, shape (n_samples,)
        Updated labels after balancing clusters.

    """

    if not balanced:
        return labels

    cluster_counts = np.bincount(labels, minlength=centroids.shape[0])
    target_count = len(X) // centroids.shape[0]
    print(cluster_counts)

    while not all(count == target_count for count in cluster_counts):
        # Calculate excess points in each cluster
        excess_points = cluster_counts - target_count
        excess_indices = np.where(excess_points > 0)[0]
        print(cluster_counts)
        print("-------")

        # Reassign excess points to other clusters
        for cluster_idx in excess_indices:
            excess_indices_cluster = np.where(labels == cluster_idx)[0]
            excess_indices_cluster = excess_indices_cluster[:excess_points[cluster_idx]]
            excess_X = X[excess_indices_cluster]
            excess_labels = np.zeros_like(excess_indices_cluster, dtype=int)
            for i, idx in enumerate(excess_indices_cluster):
                dists = np.linalg.norm(excess_X[i] - centroids, axis=1)
                sorted_labels_by_distance = np.argsort(dists)
                for lbl in sorted_labels_by_distance:
                    if cluster_counts[lbl] < target_count:
                        excess_labels[i] = lbl
                        cluster_counts[cluster_idx] -= 1
                        cluster_counts[lbl] += 1
                        break
            labels[excess_indices_cluster] = excess_labels

    return labels

def predict(X, centroids):
    """Predict the closest cluster each sample in X belongs to.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        New data points.

    centroids : array-like, shape (n_clusters, n_features)
        Cluster centers.

    Returns:
    --------
    labels : array, shape (n_samples,)
        Index of the cluster each sample belongs to.

    """
    distances = np.linalg.norm(X[:, np.newaxis, :] - centroids, axis=2)
    return np.argmin(distances, axis=1)
