import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np

def get_dataloaders(data_dir, batch_size=32, val_split=0.2, augment=False, image_size=(224, 224), whole_train = False):
    """
    Loads the iNaturalist dataset, applies transformations, and splits the training set while maintaining class balance.
    Parameters:
        data_dir (str): Path to dataset containing 'train' and 'test' folders.
        batch_size (int): Batch size for DataLoader.
        val_split (float): Fraction of training data to use for validation.
        augment (bool): Whether to apply data augmentation to training data only.
        image_size (tuple): Target image size for resizing.
    Returns:
        train_loader, val_loader, test_loader, tiny_loader
    """

    # Define transforms
    basic_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4712, 0.4600, 0.3896], std=[0.1935, 0.1877, 0.1843])
    ])

    augmented_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4712, 0.4600, 0.3896], std=[0.1935, 0.1877, 0.1843])
    ]) if augment else basic_transform

    # Load full training set with basic transform just to get labels
    base_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=basic_transform)

    # Stratified split
    train_indices, val_indices = stratified_split(base_dataset, val_split)

    # Separate datasets using different transforms
    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=augmented_transform)
    if whole_train:
        test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "val"), transform=basic_transform)
        g = torch.Generator()
        g.manual_seed(42)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
        test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, generator=g)
        return train_loader , test_loader

        
    val_dataset   = datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=basic_transform)

    train_subset = Subset(train_dataset, train_indices)
    val_subset   = Subset(val_dataset, val_indices)
    tiny_train   = Subset(train_subset, indices=range(512))
    
    test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "val"), transform=basic_transform)

    g = torch.Generator()
    g.manual_seed(42)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, generator=g)
    val_loader   = DataLoader(val_subset, batch_size=batch_size, shuffle=False, generator=g)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, generator=g)
    tiny_loader  = DataLoader(tiny_train, batch_size=16, shuffle=True, generator=g)

    return train_loader, val_loader, test_loader, tiny_loader

def stratified_split(dataset, val_split):
    """
    Splits dataset into training and validation sets with class balance.
    """
    labels = np.array(dataset.targets)
    train_indices, val_indices = train_test_split(
        np.arange(len(labels)), test_size=val_split, stratify=labels, random_state=42
    )
    return train_indices, val_indices
