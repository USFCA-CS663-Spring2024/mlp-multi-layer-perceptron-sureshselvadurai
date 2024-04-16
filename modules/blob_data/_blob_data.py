from sklearn.datasets import make_blobs
import numpy as np


def generate_blob_data(n_samples, centers, n_features=2, cluster_std=0.6, random_state=0):
    """
    Generate blob data using make_blobs function from scikit-learn.

    Args:
    - n_samples (int): The total number of points equally divided among clusters.
    - n_features (int): The number of features for each sample.
    - centers (int or array of shape [n_centers, n_features]): The number of centers to generate or the fixed center locations.
    - cluster_std (float or sequence of floats, optional): The standard deviation of the clusters.
    - random_state (int, RandomState instance or None, optional): Determines random number generation for dataset creation. Pass an int for reproducible output across multiple function calls.

    Returns:
    - X (array of shape [n_samples, n_features]): The generated samples.
    - cluster_assignments (array of shape [n_samples]): The integer labels for cluster membership of each sample.
    """
    X, cluster_assignments = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, cluster_std=cluster_std, random_state=random_state)
    return X, cluster_assignments


def generate_random_points(n_samples, n_features=2, feature_range=(-1, 1), random_state=None):
    """
    Generate random points with two features.

    Args:
    - n_samples (int): The total number of points to generate.
    - n_features (int): The number of features for each sample.
    - feature_range (tuple of floats, optional): The range (min, max) of the feature values.
    - random_state (int or RandomState instance, optional): Determines random number generation for dataset creation. Pass an int for reproducible output across multiple function calls.

    Returns:
    - X (array of shape [n_samples, n_features]): The generated samples.
    """
    rng = np.random.default_rng(random_state)
    min_value, max_value = feature_range
    X = rng.uniform(min_value, max_value, size=(n_samples, n_features))
    return X


def generate_skewed_points(n_samples, skewness_param=1.0, feature_range=(0, 1), random_state=None):
    """
    Generate very skewed points with two features using the exponential distribution.

    Args:
    - n_samples (int): The total number of points to generate.
    - skewness_param (float, optional): The parameter controlling the skewness of the distribution.
    - feature_range (tuple of floats, optional): The range (min, max) of the feature values.
    - random_state (int or RandomState instance, optional): Determines random number generation for dataset creation. Pass an int for reproducible output across multiple function calls.

    Returns:
    - X (array of shape [n_samples, 2]): The generated samples.
    """
    rng = np.random.default_rng(random_state)
    min_value, max_value = feature_range
    # Generate skewed points using the exponential distribution
    x1 = rng.exponential(scale=skewness_param, size=n_samples)
    x2 = rng.uniform(min_value, max_value, size=n_samples)
    # Combine the features
    X = np.column_stack((x1, x2))
    return X
