import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from medmnist import OrganAMNIST
import copy
from sklearn.model_selection import train_test_split



class OrganAMNISTModel(nn.Module):
    def __init__(self):
        super(OrganAMNISTModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 3 * 3, 128),  # Changed from 64*4*4 to 64*3*3
            nn.ReLU(),
            nn.Linear(128, 11)
        )

    def forward(self, x):
        features = self.features(x)
        features = features.view(-1, 64 * 3 * 3)  # Ensure correct reshaping
        out = self.classifier(features)
        return features, out

def setup_fedmix():
    # Configuration dictionary
    conf = {
        "num_rounds": 100,
        "local_epochs": 5,
        "batch_size": 32,
        "lr": 0.001,
        "momentum": 0.9,
        "weight_decay": 1e-5,
        "fedmix_lambda": 0.2,
        "mean_batch_size": 32,
        "which_dataset": "OrganAMNIST",
        "num_classes": {"OrganAMNIST": 11},
        "num_clients": 5,
        "data_split_ratio": 0.8
    }
    
    # Load OrganAMNIST dataset
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((28,28)),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    
    # Load the dataset
    train_dataset = OrganAMNIST(split='train', transform=data_transform, download=True)
    test_dataset = OrganAMNIST(split='test', transform=data_transform, download=True)
    
    # Convert dataset to DataFrame format
    def dataset_to_df(dataset):
        images = []
        labels = []
        for image, label in dataset:
            images.append(image.numpy())
            labels.append(label)
        
        return pd.DataFrame({
            'image': images,
            'label': labels
        })
    
    train_df = dataset_to_df(train_dataset)
    
    # Split data for different clients
    client_dfs = []
    samples_per_client = len(train_df) // conf["num_clients"]
    
    for i in range(conf["num_clients"]):
        start_idx = i * samples_per_client
        end_idx = (i + 1) * samples_per_client if i < conf["num_clients"] - 1 else len(train_df)
        client_df = train_df.iloc[start_idx:end_idx].copy()
        client_dfs.append(client_df)
    #print(client_dfs)
    
    
    model = OrganAMNISTModel()
    if torch.cuda.is_available():
        model = model.cuda()
    
    
    from client import Client  # Import your Client class
    clients = [Client(conf, copy.deepcopy(model), client_df) for client_df in client_dfs]
    
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=conf["batch_size"],
        shuffle=False
    )
    
    return conf, model, clients, test_loader

if __name__ == "__main__":
    conf, model, clients, test_loader = setup_fedmix()
    print("Setup completed successfully!")
    print(f"Number of clients: {len(clients)}")
    print(f"Model architecture:", model)
