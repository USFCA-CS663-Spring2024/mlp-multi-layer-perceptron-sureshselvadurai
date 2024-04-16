from torch.utils.data import Dataset, DataLoader
import torch
class WineQualityDataset(Dataset):
    def __init__(self, X, y):
        self.x = torch.tensor(X.to_numpy()).float()
        self.y = torch.tensor(y.to_numpy()).float()

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)
