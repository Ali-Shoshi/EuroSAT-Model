import torch
import torch.nn as nn

def evaluate_model(model, dataset_loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
   
    total =0     
    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for image, label in dataset_loader:
            image = image.to(device)
            label = label.to(device)
            output = model(image)

            loss = loss_fn(output, label)
            batch_size = image.size(0)
            total_loss += loss.item()

            correct +=(output.argmax(dim=1) == label).sum().item()
            total +=batch_size

    return correct, total

