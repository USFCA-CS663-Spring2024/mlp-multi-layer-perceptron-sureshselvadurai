import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

class HierarchicalClustering:
    def __init__(self, num_clusters=5):
        self.num_clusters = num_clusters
        self.hierarchical_model = None
        self.imputer = SimpleImputer(strategy='mean')
        self.scaler = StandardScaler()

    def fit(self, data):
        # Handle missing values by imputing with mean
        data_imputed = pd.DataFrame(self.imputer.fit_transform(data), columns=data.columns)

        # Normalize the data
        data_normalized = pd.DataFrame(self.scaler.fit_transform(data_imputed), columns=data.columns)

        # Initialize hierarchical clustering model
        self.hierarchical_model = AgglomerativeClustering(n_clusters=self.num_clusters)

        # Fit the model to the data
        cluster_labels = self.hierarchical_model.fit_predict(data_normalized)

        return cluster_labels

    def show_dendrogram(self, data):
        # Handle missing values by imputing with mean
        data_imputed = pd.DataFrame(self.imputer.fit_transform(data), columns=data.columns)

        # Normalize the data
        data_normalized = pd.DataFrame(self.scaler.fit_transform(data_imputed), columns=data.columns)

        # Create linkage matrix
        linkage_matrix = sch.linkage(data_normalized, method='ward')

        # Plot dendrogram
        plt.figure(figsize=(10, 7))
        dendrogram = sch.dendrogram(linkage_matrix)
        plt.title('Dendrogram')
        plt.xlabel('Samples')
        plt.ylabel('Distance')
        plt.show()

    def predict_clusters(self, data):
        if self.hierarchical_model is None:
            raise ValueError("fit method must be called before calling predict_clusters")

        # Handle missing values by imputing with mean
        data_imputed = pd.DataFrame(self.imputer.transform(data), columns=data.columns)

        # Normalize the data
        data_normalized = pd.DataFrame(self.scaler.transform(data_imputed), columns=data.columns)

        # Get cluster labels for the data
        cluster_labels = self.hierarchical_model.fit_predict(data_normalized)

        return cluster_labels

    def get_clustering_metrics(self, data):
        if self.hierarchical_model is None:
            raise ValueError("fit method must be called before calling get_clustering_metrics")

        # Handle missing values by imputing with mean
        data_imputed = pd.DataFrame(self.imputer.transform(data), columns=data.columns)

        # Normalize the data
        data_normalized = pd.DataFrame(self.scaler.transform(data_imputed), columns=data.columns)

        # Get cluster labels for the data
        cluster_labels = self.hierarchical_model.fit_predict(data_normalized)

        # Calculate silhouette score
        silhouette_avg = silhouette_score(data_normalized, cluster_labels)

        return silhouette_avg

    def get_feature_means(self, data):
        if self.hierarchical_model is None:
            raise ValueError("fit method must be called before calling get_feature_means")

        # Handle missing values by imputing with mean
        data_imputed = pd.DataFrame(self.imputer.transform(data), columns=data.columns)

        # Normalize the data
        data_normalized = pd.DataFrame(self.scaler.transform(data_imputed), columns=data.columns)

        # Get cluster labels for the data
        cluster_labels = self.hierarchical_model.fit_predict(data_normalized)

        # Create a DataFrame to store the feature means for each cl
