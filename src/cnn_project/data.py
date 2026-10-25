import torch
from torchvision import transforms
from torchvision.datasets import EuroSAT
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torch.utils.data import RandomSampler
from torch.utils.data import Dataset
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
dataset = EuroSAT(root=project_root / "data", download=True, transform=ToTensor())

class TransformDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return self.transform(image), label
    
def create_data_loaders(batch_size=1024 ,num_workser= 8, generator=torch.Generator().manual_seed(42)):

    train_transform= transforms.Compose([
        transforms.RandomRotation(degrees=15),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness= 0.1, saturation= 0.1, contrast= 0.1),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform= transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_size= int(0.8 * len(dataset))
    val_size = int(0.1* len(dataset))
    test_size = int(0.1* len(dataset))

    train_data, val_data, test_data= random_split(
        dataset, 
        [train_size,val_size,test_size],
        generator=torch.Generator().manual_seed(42)
        )

    train_data = TransformDataset(train_data, train_transform)
    val_data = TransformDataset(val_data, val_test_transform)
    test_data = TransformDataset(test_data, val_test_transform)

    train_loader=DataLoader(train_data, batch_size=batch_size, num_workers=num_workser, persistent_workers=True, pin_memory=True, prefetch_factor=2, shuffle= True, generator=generator)
    val_loader=DataLoader(val_data, batch_size=batch_size, num_workers=num_workser,  persistent_workers=True, pin_memory=True, prefetch_factor=2 , shuffle = False, generator=generator)
    test_loader= DataLoader(test_data, batch_size=batch_size, num_workers=num_workser, persistent_workers=True, pin_memory=True, prefetch_factor=2, shuffle= False, generator=generator)

    return train_loader, val_loader, test_loader


#   Stuff just for testing ----------------------------------------------------------

if __name__ == "__main__":

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


    train_loader, val_loader, test_laoder = create_data_loaders()
    print("Number of images in train data: ", len(train_data))
    print("Numebrer of images in validation data: ", len(val_data))
    print("Number of images in test data: ", len(test_data))

    print("Total:", len(train_data) + len(val_data) + len(test_data))


    image, label = next(iter(train_loader))
    print(image.shape)
    print(label.shape)
    print("Labels are :",label.min().item()," - ", label.max().item())

    print("Transforms:", dataset.transform)

    image, label = dataset[0]

    print("Type:", image.dtype)
    print("Shape:", image.shape)
    print("Minimum:", image.min().item())
    print("Maximum:", image.max().item())
    print("Channel means:", image.mean(dim=(1, 2)))
    print("Channel standard deviations:", image.std(dim=(1, 2)))
