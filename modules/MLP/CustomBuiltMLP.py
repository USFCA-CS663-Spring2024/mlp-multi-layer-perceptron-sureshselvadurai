import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class CustomBuiltMLP:
    def __init__(self, hidden_layer_tuple, activation_tuple):
        self.hidden_layers = len(hidden_layer_tuple) - 1
        self.weights = []
        self.biases = []
        self.activations = []

        # Initialize weights and biases randomly
        for i in range(self.hidden_layers):
            weight_matrix = np.random.randn(hidden_layer_tuple[i], hidden_layer_tuple[i+1])
            self.weights.append(weight_matrix)
            bias_vector = np.zeros((1, hidden_layer_tuple[i+1]))
            self.biases.append(bias_vector)
            self.activations.append(activation_tuple[i])

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        layer_output = X
        for i in range(self.hidden_layers):
            layer_input = np.dot(layer_output, self.weights[i]) + self.biases[i]
            if self.activations[i] == 'relu':
                layer_output = self.relu(layer_input)
            elif self.activations[i] == 'sigmoid':
                layer_output = self.sigmoid(layer_input)
            elif self.activations[i] == 'none':
                layer_output = layer_input
        return layer_output

    def mse_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def mse_loss_derivative(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / len(y_true)

    def backprop(self, X, y, learning_rate=0.01):
        layer_output = X
        outputs = [X]
        layer_inputs = []

        # Forward pass
        for i in range(self.hidden_layers):
            layer_input = np.dot(layer_output, self.weights[i]) + self.biases[i]
            if self.activations[i] == 'relu':
                layer_output = self.relu(layer_input)
            elif self.activations[i] == 'sigmoid':
                layer_output = self.sigmoid(layer_input)
            elif self.activations[i] == 'none':
                layer_output = layer_input
            outputs.append(layer_output)
            layer_inputs.append(layer_input)

        # Backward pass
        error = self.mse_loss_derivative(y, layer_output)
        for i in range(self.hidden_layers - 1, -1, -1):
            if self.activations[i] == 'relu':
                error *= (outputs[i+1] > 0)
            elif self.activations[i] == 'sigmoid':
                error *= (outputs[i+1] * (1 - outputs[i+1]))
            weight_derivative = np.dot(outputs[i].T, error)
            self.weights[i] -= learning_rate * weight_derivative
            self.biases[i] -= learning_rate * np.sum(error, axis=0, keepdims=True)
            error = np.dot(error, self.weights[i].T)

    def train(self, X, y, epochs=1000, learning_rate=0.01):
        for epoch in range(epochs):
            self.backprop(X, y, learning_rate)

    def predict(self, X):
        return self.forward(X)

def main():
    # Load data
    df = pd.read_csv('/Users/sureshrajaselvadurai/PycharmProjects/mlp-multi-layer-perceptron-sureshselvadurai/winequality-red.csv')

    # Define features and target
    features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                'pH', 'sulphates', 'alcohol']
    target = "quality"

    # Split data into features and target
    X = df[features].values
    y = df[target].values.reshape(-1, 1)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create and train MLP model
    mlp = CustomBuiltMLP(hidden_layer_tuple=(11, 5, 1), activation_tuple=('relu', 'sigmoid'))
    mlp.train(X_train_scaled, y_train, epochs=1000, learning_rate=0.00001)

    # Evaluate model
    train_loss = mlp.mse_loss(y_train, mlp.predict(X_train_scaled))
    test_loss = mlp.mse_loss(y_test, mlp.predict(X_test_scaled))
    print("Train MSE Loss:", train_loss)
    print("Test MSE Loss:", test_loss)

if __name__ == "__main__":
    main()
