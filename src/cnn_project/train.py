import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt



def train_model(model, train_loader, device, epochs=10, lr=0.01, eta_min=0.0002):

    loss_fn = nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=lr)

    model.train()
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=3, min_lr =eta_min)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.65)


    print("Training model...")
    for i in range(epochs):
        total_loss=0.0
        for image, label in train_loader:
            image=image.to(device, non_blocking=True)
            label=label.to(device, non_blocking=True)
            optimizer.zero_grad()
            output=model(image)
            loss= loss_fn(output, label)
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
        print(f"Epoch {i+1}/{epochs}, " f"Loss: {total_loss / len(train_loader):.4f}, " f"LR: {optimizer.param_groups[0]['lr']:.10f}")
        scheduler.step()

if __name__ == "__main__":
    train_model(None, None, None)


