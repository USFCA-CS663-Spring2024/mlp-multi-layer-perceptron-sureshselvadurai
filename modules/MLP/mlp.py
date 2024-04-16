import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, n_inputs, hidden_size, n_outputs, num_layers=1, activation=nn.ReLU(), learning_rate=0.0001):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.activation = activation
        self.learning_rate = learning_rate

        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(n_inputs, hidden_size))
        for _ in range(num_layers - 1):
            self.fc_layers.append(nn.Linear(hidden_size, hidden_size))
        self.fc_out = nn.Linear(hidden_size, n_outputs)

    def forward(self, x):
        for layer in self.fc_layers:
            x = self.activation(layer(x))
        x = self.fc_out(x)
        return x

    def reset_parameters(self):
        for layer in self.fc_layers:
            layer.reset_parameters()
        self.fc_out.reset_parameters()
