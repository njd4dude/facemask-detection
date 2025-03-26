import os
from torch.utils.data import random_split, ConcatDataset, DataLoader
from torchvision import transforms
from custom_dataset import CustomImageDataset

def prepare_datasets(dataset_base_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    categories = {"with_mask": 0, "without_mask": 1, "mask_weared_incorrect": 2}
    datasets = []

    for category, label in categories.items():
        image_dir = f"{dataset_base_path}/{category}"
        num_images = len(os.listdir(image_dir))
        labels = [label] * num_images
        datasets.append(CustomImageDataset(image_dir=image_dir, labels=labels, transform=transform))

    full_dataset = ConcatDataset(datasets)

    train_size = int(0.8 * len(full_dataset))
    validation_size = len(full_dataset) - train_size
    return random_split(full_dataset, [train_size, validation_size])

def create_dataloaders(train_dataset, validation_dataset, batch_size=32, num_workers=4):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, prefetch_factor=16)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, prefetch_factor=16)
    return train_dataloader, validation_dataloader
