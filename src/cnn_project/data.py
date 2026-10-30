import torch
from torchvision.transforms import v2 as transforms
from torchvision.datasets import EuroSAT
from torchvision.transforms.functional import pil_to_tensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torch.utils.data import RandomSampler
from torch.utils.data import Dataset
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]


class CachedEuroSAT(Dataset):
    """Store decoded RGB images as uint8 tensors; augment them on access."""

    def __init__(self, root):
        root = Path(root)
        cache_path = root / "processed" / "eurosat_rgb_uint8.pt"

        if cache_path.exists():
            cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        else:
            source = EuroSAT(root=root, download=True, transform=None)
            first_image, first_label = source[0]
            first_tensor = pil_to_tensor(first_image)
            images = torch.empty((len(source), *first_tensor.shape), dtype=torch.uint8)
            labels = torch.empty(len(source), dtype=torch.long)
            images[0] = first_tensor
            labels[0] = first_label

            for index in range(1, len(source)):
                image, label = source[index]
                images[index] = pil_to_tensor(image)
                labels[index] = label

            cache = {"images": images, "labels": labels, "classes": list(source.classes)}
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = cache_path.with_suffix(".tmp")
            torch.save(cache, temporary_path)
            temporary_path.replace(cache_path)
            print(f"Created decoded image cache: {cache_path}")

        self.images = cache["images"]
        self.labels = cache["labels"]
        self.classes = cache["classes"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index].float() / 255.0
        return image, int(self.labels[index])



class TransformDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return self.transform(image), label
    
def create_data_loaders(batch_size=1024 ,num_workers= 12 , generator=torch.Generator().manual_seed(42)):
    dataset = CachedEuroSAT(root=project_root / "data")

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

    worker_options = {"num_workers": num_workers, "pin_memory": True}
    if num_workers > 0:
        worker_options.update(persistent_workers=True, prefetch_factor=3)

    train_loader=DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=generator, **worker_options)
    val_loader=DataLoader(val_data, batch_size=batch_size, shuffle=False, generator=generator, **worker_options)
    test_loader=DataLoader(test_data, batch_size=batch_size, shuffle=False, generator=generator, **worker_options)

    return train_loader, val_loader, test_loader


#   Stuff just for testing ----------------------------------------------------------

if __name__ == "__main__":

    dataset = CachedEuroSAT(root=project_root / "data")
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

    print("Cached image dtype:", dataset.images.dtype)

    image, label = dataset[0]

    print("Type:", image.dtype)
    print("Shape:", image.shape)
    print("Minimum:", image.min().item())
    print("Maximum:", image.max().item())
    print("Channel means:", image.mean(dim=(1, 2)))
    print("Channel standard deviations:", image.std(dim=(1, 2)))
