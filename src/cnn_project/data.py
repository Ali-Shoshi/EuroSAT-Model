import torch
from torchvision.datasets import EuroSAT
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torch.utils.data import RandomSampler
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
dataset = EuroSAT(root=project_root / "data", download=True, transform=ToTensor())

print("Classes of the dataset are :")
for classname, label in enumerate(dataset.classes):
    print(f"{classname}: {label}")

image, label =dataset[0]
print("Number of images: ", len(dataset))
print("The shape of an image : ", image.shape)

loader = DataLoader(dataset, batch_size=256, num_workers=0, shuffle=True)
print("Number of Images in a batch: ",loader.batch_size)
print("Number of Batches: ",len(loader))
print("Number of Workers: ",loader.num_workers)
print("Does it use shuffle: ",isinstance(loader.sampler, RandomSampler))


train_size= int(0.8 * len(dataset))
val_size = int(0.1* len(dataset))
test_size = int(0.1* len(dataset))

train_data, val_data, test_data= random_split(
    dataset, 
    [train_size,val_size,test_size],
    generator=torch.Generator().manual_seed(42)
    )


train_loader=DataLoader(train_data, batch_size=256, shuffle= True)
val_loader=DataLoader(val_data, batch_size=256, shuffle = False)
test_loader= DataLoader(test_data, batch_size=256, shuffle= False)

print("Number of images in train data: ", len(train_data))
print("Numebrer of images in validation data: ", len(val_data))
print("Number of images in test data: ", len(test_data))

print("Total:", len(train_data) + len(val_data) + len(test_data))


image, label = next(iter(train_loader))
print(image.shape)
print(label.shape)
print("Labels are :",label.min().item()," - ", label.max().item())
