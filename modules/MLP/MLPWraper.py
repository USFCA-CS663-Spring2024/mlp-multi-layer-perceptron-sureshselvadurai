import torch
import pandas as pd


class MLPWrapper:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def train_validate_curve(self, train_loader, val_loader, num_epochs, print_every=1):
        train_losses = []
        val_losses = []
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            if print_every > 0 and (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}")
        return train_losses, val_losses

    def train(self, train_loader, num_epochs):
        self.model.train()
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
        print("Final Training Loss:", train_loss)

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        for inputs, targets in train_loader:
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        return total_loss / len(val_loader)


    def hyperparameter_tuning(self, train_loader, val_loader, num_epochs_values, learning_rates, hidden_sizes, num_layers_values,print_every=1):
        best_hyperparameters = {}
        best_loss = float('inf')
        loss_data = []

        for num_epochs in num_epochs_values:
            for learning_rate in learning_rates:
                for hidden_size in hidden_sizes:
                    for num_layers in num_layers_values:
                        self.model.reset_parameters()  # Reset model parameters for each hyperparameter combination
                        self.set_hyperparameters(num_epochs, learning_rate, hidden_size, num_layers)
                        train_losses, val_losses = self.train_validate_curve(train_loader, val_loader, num_epochs,print_every)
                        final_val_loss = val_losses[-1]
                        loss_data.append({'num_epochs': num_epochs, 'learning_rate': learning_rate,
                                          'hidden_size': hidden_size, 'num_layers': num_layers, 'val_loss': final_val_loss})
                        if final_val_loss < best_loss:
                            best_loss = final_val_loss
                            best_hyperparameters = {'num_epochs': num_epochs, 'learning_rate': learning_rate,
                                                    'hidden_size': hidden_size, 'num_layers': num_layers}

        loss_df = pd.DataFrame(loss_data)
        return best_hyperparameters, best_loss, loss_df

    def set_hyperparameters(self, num_epochs, learning_rate, hidden_size, num_layers):
        # Set hyperparameters in the model
        self.model.num_epochs = num_epochs
        self.model.learning_rate = learning_rate
        self.model.hidden_size = hidden_size
        self.model.num_layers = num_layers
