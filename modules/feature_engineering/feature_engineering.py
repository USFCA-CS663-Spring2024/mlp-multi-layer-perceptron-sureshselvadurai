from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import requests


class FeatureEngineer:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def add_one_hot_encoded_column(self, column_name, print_out=False):
        encoder = OneHotEncoder()
        column_data = self.dataframe[[column_name]]
        encoded_data = encoder.fit_transform(column_data)
        column_names = encoder.get_feature_names_out([column_name])
        encoded_df = pd.DataFrame(encoded_data.toarray(), columns=column_names)
        self.dataframe = pd.concat([self.dataframe, encoded_df], axis=1)
        if print_out:
            return self.dataframe.head(1)

    def add_day_of_week(self, date_column_name, date_format="%d/%m/%Y"):
        if date_column_name not in self.dataframe.columns:
            raise ValueError("Column '{}' not found in the DataFrame.".format(date_column_name))

        new_column_name = date_column_name + '_Day_of_Week'

        self.dataframe[date_column_name] = pd.to_datetime(self.dataframe[date_column_name], format=date_format)
        self.dataframe[new_column_name] = self.dataframe[date_column_name].dt.day_name()

        return self.dataframe[[new_column_name, date_column_name]]

    def add_day_of_month(self, date_column_name, date_format="%d/%m/%Y"):
        if date_column_name not in self.dataframe.columns:
            raise ValueError("Column '{}' not found in the DataFrame.".format(date_column_name))

        new_column_name = date_column_name + '_Day_of_Month'

        self.dataframe[date_column_name] = pd.to_datetime(self.dataframe[date_column_name], format=date_format)
        self.dataframe[new_column_name] = self.dataframe[date_column_name].dt.day

        return self.dataframe[[new_column_name, date_column_name]]

    def add_month_of_year(self, date_column_name, date_format="%d/%m/%Y"):
        if date_column_name not in self.dataframe.columns:
            raise ValueError("Column '{}' not found in the DataFrame.".format(date_column_name))

        new_column_name = date_column_name + '_Month_of_Year'

        self.dataframe[date_column_name] = pd.to_datetime(self.dataframe[date_column_name], format=date_format)
        self.dataframe[new_column_name] = self.dataframe[date_column_name].dt.month

        return self.dataframe[[new_column_name, date_column_name]]

    def add_date_differences(self, data_list, days_to_evaluate, date_column_name,
                             date_format="%d/%m/%Y"):
        if date_column_name not in self.dataframe.columns:
            raise ValueError("Column '{}' not found in the DataFrame.".format(date_column_name))

        for date_str in data_list:
            before_column_name = f"days_before_{date_str}"
            after_column_name = f"days_after_{date_str}"
            self.dataframe[before_column_name] = np.nan
            self.dataframe[after_column_name] = np.nan
            reference_date = datetime.datetime.strptime(date_str, date_format)

            for index, row in self.dataframe.iterrows():
                if pd.isnull(row[date_column_name]):
                    continue
                row_date = row[date_column_name]
                difference = (row_date - reference_date).days
                if abs(difference) <= days_to_evaluate:
                    if difference < 0:
                        self.dataframe.at[index, before_column_name] = abs(difference)
                    else:
                        self.dataframe.at[index, after_column_name] = difference

        return

    def add_oil_prices(self, oil_df, date_column_name, oil_date_format="%b %Y"):
        # Convert 'Month' column to datetime and format it
        oil_df['Month'] = pd.to_datetime(oil_df['Month'], format=oil_date_format).dt.strftime('%b %Y')

        # Merge the dataframes
        merged_df = pd.merge(self.dataframe, oil_df, how='left',
                             left_on=self.dataframe[date_column_name].dt.strftime('%b %Y'), right_on='Month')

        # Convert 'Oil_Price' and 'Oil_Price_Change' columns to numeric
        merged_df['Oil_Price'] = pd.to_numeric(merged_df['Price'], errors='coerce')
        merged_df['Oil_Price_Change'] = pd.to_numeric(merged_df['Change'], errors='coerce')

        # Return the dataframe with relevant columns
        return merged_df[['Oil_Price', 'Oil_Price_Change', date_column_name]]

    def make_time_bins(self, column_name, num_bins, time_format="%H:%M"):
        def extract_time(time_str):
            # Split the string by space and take the first part
            return time_str.split()[0]

        bin_size = 24 / num_bins
        bins = [i * bin_size for i in range(num_bins + 1)]
        labels = [f'bin_{i+1}' for i in range(num_bins)]

        # Apply extract_time function to extract only the time part
        self.dataframe[f'{column_name}_Bin'] = pd.cut(pd.to_datetime(self.dataframe[column_name].apply(extract_time), format=time_format).dt.hour, bins=bins, labels=labels)
        return self.dataframe[f'{column_name}_Bin']

    def get_time_difference(self, column1, column2):
        def get_minutes(time_str):
            parts = time_str.split()
            time_part = parts[0]  # Consider only the time part
            hours, minutes = map(int, time_part.split(':'))
            return hours * 60 + minutes

        # self.dataframe[column1] = pd.to_datetime(self.dataframe[column1], errors='coerce')
        # self.dataframe[column2] = pd.to_datetime(self.dataframe[column2], errors='coerce')

        differences = []
        for index, row in self.dataframe.iterrows():
            time1 = get_minutes(str(row[column1]))
            time2 = get_minutes(str(row[column2]))
            # Check if the second time is on the next day
            if len(str(row[column2]).split()) > 1:
                time2 += 24 * 60  # Add 24 hours in minutes
            difference = time2 - time1
            differences.append(difference)
        self.dataframe['Time_Difference_Minutes'] = differences
        return self.dataframe['Time_Difference_Minutes']
    def get_dataframe(self):
        return self.dataframe

    def get_unique_values_counts_for_column(self, column_name):
        if column_name not in self.dataframe.columns:
            raise ValueError("Column '{}' not found in the DataFrame.".format(column_name))

        unique_values_counts = self.dataframe[column_name].value_counts().reset_index()
        unique_values_counts.columns = [column_name, 'Count']

        return unique_values_counts

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

    def split_data(self, features, target, test_size=0.2, random_state=None):
        """
        Split the data into train and test sets.

        Parameters:
        - features (list or DataFrame): List of feature column names or DataFrame containing features.
        - target (str): Name of the target column.
        - test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
        - random_state (int or RandomState, optional): Controls the randomness of the training and testing indices. Defaults to None.

        Returns:
        - train_x (DataFrame): Training features.
        - test_x (DataFrame): Testing features.
        - train_y (Series): Training target.
        - test_y (Series): Testing target.
        """
        # If features is a list of column names, select those columns from the dataframe
        if isinstance(features, list):
            features_data = self.dataframe[features]
        else:
            features_data = features

        # Split data into features (X) and target (y)
        X = features_data
        y = self.dataframe[target]

        # Split data into train and test sets
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=test_size, random_state=random_state)

        return train_x, test_x, train_y, test_y

    def fill_missing_values(self, feature_list, strategy='median'):
        """
        Fill missing values in the specified list of features in self.dataframe.

        Parameters:
        - feature_list (list): List of feature column names to fill missing values.
        - strategy (str, optional): Strategy to use for filling missing values. Default is 'median'.

        """
        # Ensure all specified features are present in the DataFrame
        missing_features = set(feature_list) - set(self.dataframe.columns)
        if missing_features:
            raise ValueError("Columns {} not found in the DataFrame.".format(missing_features))

        # Fill missing values
        if strategy == 'median':
            self.dataframe[feature_list] = self.dataframe[feature_list].fillna(self.dataframe[feature_list].median())
        elif strategy == 'mean':
            self.dataframe[feature_list] = self.dataframe[feature_list].fillna(self.dataframe[feature_list].mean())
        else:
            raise ValueError("Invalid strategy. Please use 'median' or 'mean'.")

        return self.dataframe[feature_list]

    def convert_to_integer(self, feature_list):
        """
        Convert the specified list of features to integers in self.dataframe.

        Parameters:
        - feature_list (list): List of feature column names to convert to integers.

        """
        # Ensure all specified features are present in the DataFrame
        missing_features = set(feature_list) - set(self.dataframe.columns)
        if missing_features:
            raise ValueError("Columns {} not found in the DataFrame.".format(missing_features))

        # Convert features to integers
        self.dataframe[feature_list] = self.dataframe[feature_list].astype(int)

    def normalize_data(self, feature_list):
        """
        Normalize the specified list of features in self.dataframe.

        Parameters:
        - feature_list (list): List of feature column names to normalize.

        """
        # Ensure all specified features are present in the DataFrame
        missing_features = set(feature_list) - set(self.dataframe.columns)
        if missing_features:
            raise ValueError("Columns {} not found in the DataFrame.".format(missing_features))

        # Normalize the features
        scaler = StandardScaler()
        self.dataframe[feature_list] = scaler.fit_transform(self.dataframe[feature_list])
        return self.dataframe[feature_list]

    def split_data_clean(self, features, target, test_size=0.2, random_state=None):
        """
        Split the data into train and test sets after removing outliers.

        Parameters:
        - features (list or DataFrame): List of feature column names or DataFrame containing features.
        - target (str): Name of the target column.
        - test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
        - random_state (int or RandomState, optional): Controls the randomness of the training and testing indices. Defaults to None.

        Returns:
        - train_x (DataFrame): Training features.
        - test_x (DataFrame): Testing features.
        - train_y (Series): Training target.
        - test_y (Series): Testing target.
        """

        # Remove outliers from features
        cleaned_data = self.remove_outliers(self.dataframe[features+[target]])

        # Split data into features (X) and target (y)
        X = cleaned_data[features]
        y = cleaned_data[[target]]

        # Split data into train and test sets
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=test_size, random_state=random_state)

        return train_x, test_x, train_y, test_y

    def remove_outliers(self, data):
        """
        Remove outlier rows from each feature in the given DataFrame.

        Parameters:
        - data (DataFrame): DataFrame containing features.

        Returns:
        - cleaned_data (DataFrame): DataFrame with outliers removed.
        """
        # Calculate the z-score for each data point in each column
        z_scores = ((data - data.mean()) / data.std()).abs()

        # Define a threshold for outliers (e.g., z-score > 3)
        threshold = 1

        # Identify outlier rows for each feature
        outlier_mask = (z_scores > threshold).any(axis=1)

        # Print the number of rows removed for each feature
        num_removed = outlier_mask.sum()
        print(f"Number of rows removed: {num_removed}")

        # Remove outlier rows
        cleaned_data = data[~outlier_mask]

        return cleaned_data

    def add_sinusoidal_day_of_week(self, date_column_name):
        new_column_name = date_column_name + '_sin_day_of_week'
        if date_column_name not in self.dataframe.columns:
            raise ValueError("Column '{}' not found in the DataFrame.".format(date_column_name))

        self.dataframe[date_column_name] = pd.to_datetime(self.dataframe[date_column_name])
        day_of_week_sin = np.sin(2 * np.pi * self.dataframe[date_column_name].dt.dayofweek / 7)
        self.dataframe[new_column_name] = day_of_week_sin

        return self.dataframe[[new_column_name, date_column_name]]

    def add_sinusoidal_month_of_year(self, date_column_name):
        new_column_name = date_column_name + '_sin_month_of_year'
        if date_column_name not in self.dataframe.columns:
            raise ValueError("Column '{}' not found in the DataFrame.".format(date_column_name))

        self.dataframe[date_column_name] = pd.to_datetime(self.dataframe[date_column_name])
        month_of_year_sin = np.sin(2 * np.pi * self.dataframe[date_column_name].dt.month / 12)
        self.dataframe[new_column_name] = month_of_year_sin

        return self.dataframe[[new_column_name, date_column_name]]

    def add_sinusoidal_day_of_month(self, date_column_name):
        new_column_name = date_column_name + '_sin_day_of_month'
        if date_column_name not in self.dataframe.columns:
            raise ValueError("Column '{}' not found in the DataFrame.".format(date_column_name))

        self.dataframe[date_column_name] = pd.to_datetime(self.dataframe[date_column_name])
        day_of_month_sin = np.sin(2 * np.pi * self.dataframe[date_column_name].dt.day / 31)
        self.dataframe[new_column_name] = day_of_month_sin

        return self.dataframe[[new_column_name, date_column_name]]

    def merge_data(self, other_df, common_column_name):
        """
        Merge the current dataframe with another dataframe based on a common column.

        Parameters:
        - other_df (DataFrame): The dataframe to merge with.
        - common_column_name (str): The name of the common column to merge on.

        Returns:
        - merged_df (DataFrame): The merged dataframe.
        """
        if common_column_name not in self.dataframe.columns or common_column_name not in other_df.columns:
            raise ValueError(f"Column '{common_column_name}' not found in both dataframes.")

        # Perform the merge
        merged_df = pd.merge(self.dataframe, other_df, how='left', on=common_column_name)

        # Rename newly added columns with common_column_name as prefix
        for column in other_df.columns:
            if column != common_column_name:
                merged_df.rename(columns={column: f"{common_column_name}_{column}"}, inplace=True)

        return merged_df

    def get_usd_inr_value(self, date_column_name):
        if date_column_name not in self.dataframe.columns:
            raise ValueError("Column '{}' not found in the DataFrame.".format(date_column_name))

        usd_inr_values = []

        for date in self.dataframe[date_column_name]:
            # Assuming you have an API endpoint to fetch historical exchange rates
            url = f"https://api.exchangerate-api.com/v4/latest/USD"  # Example API endpoint
            response = requests.get(url)

            if response.status_code == 200:
                exchange_rates = response.json()["rates"]
                if "INR" in exchange_rates:
                    usd_inr_values.append(exchange_rates["INR"])
                else:
                    usd_inr_values.append(np.nan)
            else:
                usd_inr_values.append(np.nan)

        self.dataframe['USD_INR_Value'] = usd_inr_values

        return self.dataframe[['USD_INR_Value', date_column_name]]

    def add_sinusoidal_time(self, time_column_name):
        if time_column_name not in self.dataframe.columns:
            raise ValueError("Column '{}' not found in the DataFrame.".format(time_column_name))

        # Extract hour and minute components from the time column
        hour = self.dataframe[time_column_name].apply(lambda x: pd.to_datetime(x).hour)
        minute = self.dataframe[time_column_name].apply(lambda x: pd.to_datetime(x).minute)

        # Calculate sinusoidal transformation for hour and minute
        hour_sin = np.sin(2 * np.pi * hour / 24)
        minute_sin = np.sin(2 * np.pi * minute / 60)

        # Combine hour and minute sinusoidal transformations
        time_sin = hour_sin + minute_sin

        # Add new column with sinusoidal transformation
        new_column_name = time_column_name + "_sin_time"
        self.dataframe[new_column_name] = time_sin

        return self.dataframe[[new_column_name, time_column_name]]

    def fill_missing_values_with_zero(self, feature_list):
        """
        Fill missing values in the specified list of features with zeros.

        Parameters:
        - feature_list (list): List of feature column names to fill missing values with zeros.
        """
        # Ensure all specified features are present in the DataFrame
        missing_features = set(feature_list) - set(self.dataframe.columns)
        if missing_features:
            raise ValueError("Columns {} not found in the DataFrame.".format(missing_features))

        # Fill missing values with zero
        self.dataframe[feature_list] = self.dataframe[feature_list].fillna(0)

        return self.dataframe[feature_list]

    def add_constant_column(self, column_name, constant_value):
        """
        Add a column with a constant value to the DataFrame.

        Parameters:
        - column_name (str): Name of the new column.
        - constant_value: Constant value to assign to every row in the new column.
        """
        self.dataframe[column_name] = constant_value
        return self.dataframe[column_name]


    def split_data_test_train_val(self, features, target, test_size=0.2, val_size=0.2, random_state=None):
        """
        Split the data into train, validation, and test sets.

        Parameters:
        - features (list or DataFrame): List of feature column names or DataFrame containing features.
        - target (str): Name of the target column.
        - test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
        - val_size (float, optional): The proportion of the dataset to include in the validation split. Defaults to 0.2.
        - random_state (int or RandomState, optional): Controls the randomness of the training and testing indices. Defaults to None.

        Returns:
        - train_x (DataFrame): Training features.
        - val_x (DataFrame): Validation features.
        - test_x (DataFrame): Testing features.
        - train_y (Series): Training target.
        - val_y (Series): Validation target.
        - test_y (Series): Testing target.
        """
        # If features is a list of column names, select those columns from the dataframe
        if isinstance(features, list):
            features_data = self.dataframe[features]
        else:
            features_data = features

        # Split data into features (X) and target (y)
        X = features_data
        y = self.dataframe[target]

        # Split data into train and test sets
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Further split train set into train and validation sets
        train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=val_size, random_state=random_state)

        return train_x, val_x, test_x, train_y, val_y, test_y

    def normalize_and_convert_to_numpy(self, train_x, test_x, train_y, test_y):
        # Normalize features
        train_x_normalized = (train_x - train_x.mean()) / train_x.std()
        test_x_normalized = (test_x - train_x.mean()) / train_x.std()

        # Convert data to numpy arrays
        train_X = train_x_normalized.values
        test_X = test_x_normalized.values
        train_Y = train_y.values.reshape(-1, 1)
        test_Y = test_y.values.reshape(-1, 1)

        return train_X, test_X, train_Y, test_Y

    def train_test_normalize_X_split(self,X,y, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled,X_test_scaled,y_train,y_test

    def get_features_targets(self,features,target):
        X = self.dataframe[features].values
        y = self.dataframe[target].values.reshape(-1, 1)

        return X,y