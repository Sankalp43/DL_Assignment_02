import os
import torch
import random
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from tqdm.notebook import tqdm
from utils import set_seed, train_model, validate
from dataloader import get_dataloaders
from model import FlexibleCNN
import matplotlib.pyplot as plt
import argparse
import logging

# ---------------- Logging Setup ---------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

# ---------------- Device Setup ---------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# ---------------- Data Directory ---------------- #
data_dir = "inaturalist_12K"  # Replace with your dataset path
logging.info(f"Data directory set to: {data_dir}")

# ---------------- Load Pretrained ResNet50 ---------------- #
model = models.resnet50(pretrained=True)
logging.info("Loaded pretrained ResNet50 model.")

# Freeze all layers to retain learned features
for param in model.parameters():
    param.requires_grad = False
logging.info("Froze all layers of the ResNet50 model.")

# Replace the final fully connected layer with one for 10 classes
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)
logging.info("Replaced final FC layer with new layer for 10 output classes.")

# ---------------- Optional: W&B Setup Placeholder ---------------- #
# The following W&B setup is currently commented out, but can be used for experiment tracking.
# Uncomment and configure as needed.

# run = wandb.init(config=None, project="DA6401_assignment2")
# run.mark_preempting()
# with wandb.init(config=config):
#     config = wandb.config
# wandb.run.name = "Pretrained_Model,lr=0.00001,with augmentation"
# wandb.run.save()

# ---------------- Load Dataloaders ---------------- #
train_loader, val_loader, test_loader, tiny_loader = get_dataloaders(data_dir, augment=True)
logging.info("Data loaders initialized with data augmentation.")

# ---------------- Set Random Seed ---------------- #
set_seed(42)
logging.info("Random seed set to 42.")

# ---------------- Train Model ---------------- #
logging.info("Starting training...")
train_model(model, train_loader, val_loader, device, learning_rate=0.00001, epochs=20)
logging.info("Training complete.")

# ---------------- Evaluate on Test Set ---------------- #
criterion = torch.nn.CrossEntropyLoss().to(device)
logging.info("Evaluating model on test set...")
test_loss, test_acc = validate(model, test_loader, criterion, device)
logging.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# run.finish()  # Uncomment if using W&B run context
