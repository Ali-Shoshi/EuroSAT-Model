import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path



def train_model(model, train_loader, device, epochs=10, lr=0.01, eta_min=0.0002,
                gamma=0.80, val_loader=None, history=None, checkpoint_path=None):

    if checkpoint_path is not None and val_loader is None:
        raise ValueError("val_loader is required to save the best validation checkpoint")
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_accuracy = float("-inf")
    best_epoch = None

    loss_fn = nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=lr)

    if history is None:
        history = {key: [] for key in ("epoch", "train_loss", "train_accuracy",
                                      "val_loss", "val_accuracy", "learning_rate")}
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=3, min_lr =eta_min)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=gamma)


    print("Training model...")
    for i in range(epochs):
        model.train()
        total_loss=0.0
        train_correct = 0
        train_total = 0
        for image, label in train_loader:
            image=image.to(device, non_blocking=True)
            label=label.to(device, non_blocking=True)
            optimizer.zero_grad()
            output=model(image)
            loss= loss_fn(output, label)
            loss.backward()
            optimizer.step()
            batch_size = label.size(0)
            total_loss += loss.item() * batch_size
            train_correct += (output.argmax(dim=1) == label).sum().item()
            train_total += batch_size

        val_loss = None
        val_accuracy = None
        if val_loader is not None:
            model.eval()
            val_loss_total = 0.0
            val_correct = 0
            val_total = 0
            with torch.inference_mode():
                for image, label in val_loader:
                    image = image.to(device, non_blocking=True)
                    label = label.to(device, non_blocking=True)
                    output = model(image)
                    batch_size = label.size(0)
                    val_loss_total += loss_fn(output, label).item() * batch_size
                    val_correct += (output.argmax(dim=1) == label).sum().item()
                    val_total += batch_size
            val_loss = val_loss_total / val_total
            val_accuracy = val_correct / val_total

        epoch = len(history["epoch"]) + 1
        learning_rate = optimizer.param_groups[0]["lr"]
        history["epoch"].append(epoch)
        history["train_loss"].append(total_loss / train_total)
        history["train_accuracy"].append(train_correct / train_total)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["learning_rate"].append(learning_rate)
        message = (f"Epoch {epoch}, Loss: {history['train_loss'][-1]:.4f}, "
                   f"Accuracy: {history['train_accuracy'][-1]:.4f}, LR: {learning_rate:.10f}")
        if val_loader is not None:
            message += f", Val loss: {val_loss:.4f}, Val accuracy: {val_accuracy:.4f}"
        print(message)
        if checkpoint_path is not None and val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)
        scheduler.step()

    if checkpoint_path is not None:
        print(f"Best checkpoint: epoch {best_epoch}, validation accuracy "
              f"{best_val_accuracy:.4%}, saved to {checkpoint_path}")
    return history

if __name__ == "__main__":
    train_model(None, None, None)


