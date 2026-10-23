import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt

from data import create_data_loaders
from model import EuroSATModel


def train_model():
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("The model is using CUDA")
    else:
        device = torch.device("cpu")
        print("The model is using CPU")

    train_loader, val_loader, test_loader = create_data_loaders()

    model = EuroSATModel()
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=0.001)

    epochs=10
    model.train()
  

    print("Training model...")
    for i in range(epochs):
        total_loss=0.0
        for image, label in train_loader:
            image=image.to(device)
            label=label.to(device)
            optimizer.zero_grad()
            output=model(image)
            loss= loss_fn(output, label)
            loss.backward()
            optimizer.step()
            total_loss+=loss
        print(f"Epoch {i+1}/{epochs}, Loss: {total_loss.item()/len(train_loader):.4f}")

if __name__ == "__main__":
    train_model()


