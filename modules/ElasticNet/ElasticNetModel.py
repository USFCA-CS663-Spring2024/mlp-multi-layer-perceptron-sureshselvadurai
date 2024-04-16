import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import numpy as np
import pickle


class ElasticNetModel:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.best_max_iter = None
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.best_model = None
        self.best_alpha = None
        self.best_l1_ratio = None
        self.best_mse = None
        self.test_predictions = None
        self.test_rse = None

    @classmethod
    def load_from_file(cls, X_train=None, y_train=None, X_test=None, y_test=None):
        """
        Instantiate an ElasticNetModel object by loading a saved model from a file.

        Parameters:
        - filepath : str
            The file path including the filename from which the model will be loaded.
        - X_train, y_train, X_test, y_test : array-like, optional
            Training and testing data (features and target) to be passed to the instantiated object.

        Returns:
        - obj : ElasticNetModel
            An instance of the ElasticNetModel class with the saved model loaded.
        """
        filepath = 'elastic_net_model.pkl'
        with open(filepath, 'rb') as f:
            loaded_model = pickle.load(f)

        # If training and testing data are provided, pass them to the instantiated object
        if X_train is not None and y_train is not None and X_test is not None and y_test is not None:
            obj = cls(X_train, y_train, X_test, y_test)
        else:
            # If training and testing data are not provided, instantiate an object without them
            obj = cls(None, None, None, None)

        # Assign the loaded model to the instantiated object's attributes
        obj.best_model = loaded_model
        # obj.best_alpha = loaded_model.alpha
        # obj.best_l1_ratio = loaded_model.l1_ratio
        # obj.best_max_iter = loaded_model.max_iter

        print(f"ElasticNetModel loaded successfully from: {filepath}")

        return obj

    def hyperparameter_tuning(self, alphas, l1_ratios, max_iter=None, tol=None):
        """
        Perform hyperparameter tuning using GridSearchCV only on the training data.

        Parameters:
        - alphas : array-like
            List of alpha values to try.
        - l1_ratios : array-like
            List of l1_ratio values to try.
        - max_iter : int, optional (default=1000)
            Maximum number of iterations for the solver.
        - tol : float, optional (default=1e-4)
            Tolerance for the optimization solver.

        This method updates the best_model, best_alpha, best_l1_ratio,
        and best_mse attributes.
        """
        if tol is None:
            tol = [1e-4]
        if max_iter is None:
            max_iter = [1000]
        param_grid = {
            'alpha': alphas,
            'l1_ratio': l1_ratios,
            'max_iter': max_iter,
            'tol': tol
        }
        grid_search = GridSearchCV(ElasticNet(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(self.X_train, self.y_train)

        self.best_model = grid_search.best_estimator_
        self.best_alpha = grid_search.best_params_['alpha']
        self.best_l1_ratio = grid_search.best_params_['l1_ratio']
        self.best_mse = -grid_search.best_score_
        self.best_max_iter = grid_search.best_params_['max_iter']

        # Print MSE for each combination of hyperparameters
        results = grid_search.cv_results_
        for mean_score, params in zip(results['mean_test_score'], results['params']):
            print(
                f"Alpha: {params['alpha']}, L1 Ratio: {params['l1_ratio']}, Max Iter: {params['max_iter']}, Mean Squared Error: {round(-mean_score, 2)}")

    def print_best_hyperparameters(self):
        """
        Print the best hyperparameters and the corresponding mean squared error.
        """
        print(f"Best Alpha: {self.best_alpha}")
        print(f"Best L1 Ratio: {self.best_l1_ratio}")
        print(f"Best Max Iter: {self.best_max_iter}")
        print(f"Best Mean Squared Error: {round(self.best_mse, 2)}")

    def fit(self, l1_ratio, alpha, max_iter=1000, tol=1e-4):
        """
        Fit the ElasticNet model with specified hyperparameters.

        Parameters:
        - l1_ratio : float
            The ElasticNet mixing parameter.
        - alpha : float
            The regularization parameter.
        - max_iter : int, optional (default=1000)
            Maximum number of iterations for the solver.
        - tol : float, optional (default=1e-4)
            Tolerance for the optimization solver.
        """
        self.best_model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, tol=tol, random_state=42)
        self.best_model.fit(self.X_train, self.y_train)

    def get_feature_importance(self):
        feature_names = [f'feature_{i}' for i in range(self.X_train.shape[1])]
        feature_importance = dict(zip(feature_names, self.best_model.coef_))
        sorted_feature_importance = dict(
            sorted(feature_importance.items(), key=lambda item: abs(item[1]), reverse=True))

        # Convert to DataFrame
        feature_importance_df = pd.DataFrame.from_dict(sorted_feature_importance, orient='index',
                                                       columns=['Importance'])

        return feature_importance_df

    def predict_test_data(self):
        """
        Predict the target variable for the test data and calculate the Root Squared Error (RSE).

        Returns:
        - rse: float
            Root Squared Error for the test data.
        - predictions: array-like
            Predicted target variable values for the test data.
        """
        if self.best_model is None:
            raise ValueError("You need to fit the model before making predictions.")

        self.test_predictions = self.best_model.predict(self.X_test)
        self.test_rse = mean_squared_error(self.y_test, self.test_predictions)
        self.test_rse = np.sqrt(self.test_rse)
        return self.test_rse, self.test_predictions

    def predict_data(self, X, Y=None):
        """
        Predict the target variable for the given data.

        Parameters:
        - X : array-like
            Input features for which predictions are to be made.

        Returns:
        - predictions: array-like
            Predicted target variable values for the given data.
        """
        rmse = None
        if self.best_model is None:
            raise ValueError("You need to fit the model before making predictions.")

        test_predictions = self.best_model.predict(X)

        if Y is not None:
            rse = mean_squared_error(Y, test_predictions)
            rmse = np.sqrt(rse)

        return rmse, test_predictions

    def save_model(self):
        """
        Save the trained model to a file using pickle.

        Parameters:
        - model : object
            The trained model object to be saved.
        - filepath : str
            The file path including the filename where the model will be saved.
        """
        filepath = 'elastic_net_model.pkl'
        with open(filepath, 'wb') as f:
            pickle.dump(self.best_model, f)
        print(f"Model saved successfully at: {filepath}")

    def load_model(self):
        """
        Load a trained model from a file using pickle.

        Parameters:
        - filepath : str
            The file path including the filename from which the model will be loaded.

        Returns:
        - model : object
            The trained model object loaded from the file.
        """
        filepath = 'elastic_net_model.pkl'
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded successfully from: {filepath}")
        return model
