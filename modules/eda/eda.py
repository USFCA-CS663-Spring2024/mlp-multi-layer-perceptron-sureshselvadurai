# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools


class EDAAnalyzer:

    def __init__(self, data):
            if isinstance(data, np.ndarray):
                if data.shape[1] == 2:
                    self.dataframe = pd.DataFrame(data, columns=['x', 'y'])
                else:
                    columns = ['x'] + [f'y{i}' for i in range(1, data.shape[1])]
                    self.dataframe = pd.DataFrame(data, columns=columns)
            elif isinstance(data, pd.DataFrame):
                self.dataframe = data
            else:
                raise ValueError("Input data must be a pandas DataFrame or a numpy.ndarray.")

    def display_head(self, n=5):
        return self.dataframe.head(n)

    def display_shape(self):
        df_rows, df_cols = self.dataframe.shape
        return "Shape of data: rows: " + str(df_rows) + " cols: " + str(df_cols)

    def display_column_info(self):
        # A replacement for df.info but does the same functionality
        column_names = self.dataframe.columns.tolist()
        df_rows = len(self.dataframe)
        column_info_list = []

        for col in column_names:
            non_null_count = self.dataframe[col].count()
            percent_non_null = (non_null_count / df_rows) * 100
            data_type = self.dataframe[col].dtype

            column_info_list.append({
                'Column Name': col,
                'Non-null Count': non_null_count,
                'Percent Non-null': percent_non_null,
                'Data Type': data_type
            })

        column_info = pd.DataFrame(column_info_list)
        column_info = column_info.sort_values(by='Percent Non-null', ascending=False)
        column_info = column_info.reset_index(drop=True)

        return column_info

    def describe(self):
        return self.dataframe.describe().T

    def plot_histograms(self, ignore_columns=[]):
        columns_to_plot = [col for col in self.dataframe.columns if col not in ignore_columns]
        len_columns = len(columns_to_plot)

        num_rows = (len_columns + 3) // 4
        fig, axs = plt.subplots(nrows=num_rows, ncols=4, figsize=(20, 4 * num_rows))
        axs = axs.flatten()

        for i, col in enumerate(columns_to_plot):
            sns.histplot(self.dataframe[col], ax=axs[i], kde=True)

        for j in range(i + 1, len(axs)):
            axs[j].axis('off')
        plt.subplots_adjust(hspace=0.5)

        plt.show()

    def get_skew_kurt(self, ignore_columns=[]):
        skew_kurt_data = {'Column Name': [], 'Skewness': [], 'Kurtosis': [], 'Category': []}
        columns_to_analyze = [col for col in self.dataframe.columns if col not in ignore_columns]
        skew_kurt_data = []

        for col in columns_to_analyze:
            skewness = abs(self.dataframe[col].skew())
            kurtosis = abs(self.dataframe[col].kurt())

            # Ref : https://www.statology.org/how-to-report-skewness-kurtosis/
            category = "Power Data" if skewness > 1 and kurtosis > 3 else \
                       "Normal Data" if abs(skewness) <= 1 and abs(kurtosis) <= 3 else \
                       "Inbetween"
            skew_row_data = {
                'Column Name': col,
                'Skewness': skewness,
                'Kurtosis': kurtosis,
                'Category': category
            }

            skew_kurt_data.append(skew_row_data)
        return pd.DataFrame(skew_kurt_data).sort_values(by='Category')

    def get_scatter_plot(self, y_cols=None, add_trend=False):
        numeric_columns = self.dataframe.select_dtypes(include=['number']).columns

        if y_cols is None:
            y_cols = numeric_columns

        combinations = list(itertools.product(numeric_columns, y_cols))

        combinations = [(x_col, y_col) for x_col, y_col in combinations if x_col != y_col]

        if not combinations:
            print("No valid combinations found to plot.")
            return

        num_plots = len(combinations)
        num_cols = 3
        num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate the number of rows required
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 4 * num_rows))  # Adjust figsize for readability
        axs = axs.flatten()

        for i, (x_col, y_col) in enumerate(combinations):
            if not add_trend:
                sns.scatterplot(data=self.dataframe, x=x_col, y=y_col, ax=axs[i])
            else:
                sns.regplot(data=self.dataframe, x=x_col, y=y_col, ax=axs[i], scatter=False, color='red')

        for j in range(i + 1, len(axs)):
            axs[j].axis('off')

        plt.subplots_adjust(hspace=0.5)
        plt.show()

    def get_pair_plot(self):
        return sns.pairplot(self.dataframe)

    def get_box_plot(self):
        numeric_columns = self.dataframe.select_dtypes(include=['number']).columns

        num_plots = len(numeric_columns)
        num_cols = 3
        num_rows = (num_plots + 1) // 2
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 14))
        axs = axs.flatten()

        for i, col in enumerate(numeric_columns):
            sns.boxplot(data=self.dataframe, y=self.dataframe[col], ax=axs[i])
            axs[i].set_xlabel(col)

        for j in range(i + 1, len(axs)):
            axs[j].axis('off')

        plt.subplots_adjust(hspace=0.5)
        plt.show()

    def get_heat_map(self):
        non_numeric_columns = self.dataframe.select_dtypes(exclude=['number']).columns
        correlation_matrix = self.dataframe.drop(non_numeric_columns, axis=1).corr().abs()
        sns.heatmap(correlation_matrix, annot=True)
        plt.show()

    def get_correlation_pairs(self):

        # Reference 2
        threshold_high=0.7
        threshold_low=0.3

        numeric_columns = self.dataframe.select_dtypes(include=['number']).columns
        correlation_matrix = self.dataframe[numeric_columns].corr().abs()

        correlation_pairs_df = []

        for i,col1 in enumerate(numeric_columns):
            for j,col2 in enumerate(numeric_columns):
                if col1 != col2:
                    if i < j:

                        correlation_coefficient = correlation_matrix.loc[col1, col2]

                        if correlation_coefficient > threshold_high:
                            correlation_category = 'High'
                        elif correlation_coefficient <= threshold_low:
                            correlation_category = 'Low'
                        else:
                            correlation_category = 'Moderate'

                        correlation_pairs_df.append({
                            'Column 1': col1,
                            'Column 2': col2,
                            'Correlation Coefficient': correlation_coefficient,
                            'Correlation Category': correlation_category
                        })

        return pd.DataFrame(correlation_pairs_df)

    def get_top_correlated_pairs(self, n=10):
        top_correlated_pairs = self.get_correlation_pairs().sort_values(by='Correlation Coefficient', ascending=False).head(n)
        return top_correlated_pairs

    def get_top_correlated_pairs_by_column(self,column_name,n=10):
        top_correlated_pairs = self.get_correlation_pairs()
        top_correlated_pairs = top_correlated_pairs[(top_correlated_pairs["Column 1"]==column_name) | (top_correlated_pairs["Column 2"]==column_name)]
        return top_correlated_pairs.sort_values(by='Correlation Coefficient', ascending=False).head(n)

    def get_missing_rows(self):
        return self.dataframe[self.dataframe.isnull().any(axis=1)]

    def get_unique_value_counts(self):
        object_columns = self.dataframe.select_dtypes(include=['object']).columns
        unique_counts = []

        for col in object_columns:
            unique_count = self.dataframe[col].nunique()
            unique_counts.append({'Column Name': col, 'Unique Count': unique_count})

        return pd.DataFrame(unique_counts)

    def get_date_range(self, column_name, data_format=None):
        if column_name not in self.dataframe.columns:
            raise ValueError("Column '{}' not found in the DataFrame.".format(column_name))

        column_data = self.dataframe[column_name]

        if data_format is not None:
            column_data = pd.to_datetime(column_data, format=data_format, errors='coerce')
            if column_data.isnull().any():
                raise ValueError(
                    "Invalid date format '{}' or missing values in column '{}'.".format(data_format, column_name))

        min_value = column_data.min()
        max_value = column_data.max()

        return min_value, max_value

    def get_unique_values_counts_for_column(self, column_name):
        if column_name not in self.dataframe.columns:
            raise ValueError("Column '{}' not found in the DataFrame.".format(column_name))

        unique_values_counts = self.dataframe[column_name].value_counts().reset_index()
        unique_values_counts.columns = [column_name, 'Count']

        return unique_values_counts

    def get_dataframe(self):
        return self.dataframe

    def get_non_numeric_columns(self):
        non_numeric_columns = self.dataframe.select_dtypes(exclude=['number']).columns.tolist()
        return non_numeric_columns




