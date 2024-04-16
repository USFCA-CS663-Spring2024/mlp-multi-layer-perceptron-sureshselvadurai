import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np


class KMeansClustering:
    def __init__(self, optimal_num_clusters=5, num_clusters_range=(3, 4)):
        self.num_clusters_range = num_clusters_range
        self.optimal_num_clusters = optimal_num_clusters
        self.kmeans_model = None
        self.imputer = SimpleImputer(strategy='mean')
        self.scaler = StandardScaler()

    def fit_hyperparameter_tuning(self, data, sample_size=None):
        if sample_size is not None:
            data_sample = data.sample(sample_size, random_state=42)
        else:
            data_sample = data

        # Handle missing values by imputing with mean
        data_imputed = pd.DataFrame(self.imputer.fit_transform(data_sample), columns=data_sample.columns)

        # Normalize the data
        data_normalized = pd.DataFrame(self.scaler.fit_transform(data_imputed), columns=data_sample.columns)

        # Initialize lists to store silhouette scores
        silhouette_scores = []

        # Iterate over each cluster in the range
        for num_clusters in range(*self.num_clusters_range):
            # Initialize KMeans model
            kmeans_model = KMeans(n_clusters=num_clusters, random_state=42)

            # Fit the model to the data
            kmeans_model.fit(data_normalized)

            # Get cluster labels for each data point
            cluster_labels = kmeans_model.labels_

            # Calculate silhouette score
            silhouette_avg = silhouette_score(data_normalized, cluster_labels)

            # Print silhouette score
            print(f"Number of clusters: {num_clusters}, Silhouette Score: {silhouette_avg}")

            # Append silhouette score to the list
            silhouette_scores.append(silhouette_avg)

        # Find the optimal number of clusters based on silhouette score
        self.optimal_num_clusters = self.num_clusters_range[np.argmax(silhouette_scores)]
        print("optimal number of clusters : ", self.optimal_num_clusters)

    def fit(self, data):
        if self.imputer.statistics_ is None:
            raise ValueError("fit_hyperparameter_tuning method must be called before calling fit")

        # Handle missing values by imputing with mean
        data_imputed = pd.DataFrame(self.imputer.transform(data), columns=data.columns)

        # Normalize the data
        data_normalized = pd.DataFrame(self.scaler.transform(data_imputed), columns=data.columns)

        # Initialize KMeans model with optimal number of clusters
        self.kmeans_model = KMeans(n_clusters=self.optimal_num_clusters, random_state=42)

        # Fit the model to the data
        cluster_labels = self.kmeans_model.fit_predict(data_normalized)

        # Get centroids of each cluster
        cluster_centers = self.kmeans_model.cluster_centers_

        return cluster_labels, cluster_centers

    def predict_clusters(self, data):
        if self.imputer.statistics_ is None:
            raise ValueError("fit_hyperparameter_tuning method must be called before calling predict_clusters")

        # Handle missing values by imputing with mean
        data_imputed = pd.DataFrame(self.imputer.transform(data), columns=data.columns)

        # Normalize the data
        data_normalized = pd.DataFrame(self.scaler.transform(data_imputed), columns=data.columns)

        # Get cluster labels for the data
        cluster_labels = self.kmeans_model.predict(data_normalized)

        return cluster_labels

    def get_clustering_metrics(self, data):
        if self.kmeans_model is None:
            raise ValueError("fit method must be called before calling get_clustering_metrics")

        # Handle missing values by imputing with mean
        data_imputed = pd.DataFrame(self.imputer.transform(data), columns=data.columns)

        # Normalize the data
        data_normalized = pd.DataFrame(self.scaler.transform(data_imputed), columns=data.columns)

        # Get cluster labels for the data
        cluster_labels = self.kmeans_model.predict(data_normalized)

        # Calculate silhouette score
        silhouette_avg = silhouette_score(data_normalized, cluster_labels)

        return silhouette_avg

    def get_feature_means(self, data):
        if self.kmeans_model is None:
            raise ValueError("fit method must be called before calling get_feature_means")

        # Handle missing values by imputing with mean
        data_imputed = pd.DataFrame(self.imputer.transform(data), columns=data.columns)

        # Normalize the data
        data_normalized = pd.DataFrame(self.scaler.transform(data_imputed), columns=data.columns)

        # Get cluster labels for the data
        cluster_labels = self.kmeans_model.predict(data_normalized)

        # Create a DataFrame to store the feature means for each cluster
        feature_means_df = pd.DataFrame(columns=data.columns)

        # Iterate over each cluster
        for cluster_label in np.unique(cluster_labels):
            # Subset the data points belonging to the current cluster
            cluster_data = data_imputed[cluster_labels == cluster_label]

            # Compute the mean of each feature for the current cluster
            cluster_feature_means = cluster_data.mean(axis=0)

            # Add the computed feature means to the DataFrame
            feature_means_df.loc[cluster_label] = cluster_feature_means

        return feature_means_df.sort_values(by=data.columns[0])
