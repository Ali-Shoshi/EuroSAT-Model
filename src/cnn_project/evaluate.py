import torch
import torch.nn as nn

def evaluate_model(model, train_loader, val_loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
   
    train_total =0 
    val_total=0
    train_total_loss = 0.0
    val_total_loss = 0.0
    train_correct = 0
    val_correct = 0

    with torch.no_grad():
        for image, label in train_loader:
            image = image.to(device)
            label = label.to(device)
            output = model(image)

            train_loss = loss_fn(output, label)
            batch_size = image.size(0)
            train_total_loss += train_loss.item()

            train_correct +=(output.argmax(dim=1) == label).sum().item()
            train_total +=batch_size
        for image, label in val_loader:
            image = image.to(device)
            label = label.to(device)
            output = model(image)

            val_loss = loss_fn(output, label)
            batch_size = image.size(0)
            val_total_loss += val_loss.item()

            val_correct +=(output.argmax(dim=1) == label).sum().item()
            val_total +=batch_size

    return train_correct,val_correct, train_total, val_total

