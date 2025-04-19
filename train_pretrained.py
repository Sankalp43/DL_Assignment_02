import os
import torch
import random
import numpy as np
from torchvision import datasets, transforms , models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from tqdm.notebook import tqdm
from utils import set_seed, train_model , validate
from dataloader import get_dataloaders
from model import FlexibleCNN
import matplotlib.pyplot as plt
import argparse
import wandb
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

data_dir = "inaturalist_12K"  # Replace with your dataset path

model = models.resnet50(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace the final layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)  

# run = wandb.init(config =None,
                    # project="DA6401_assignment2",)
    # run.mark_preempting()

    # with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
# config = wandb.config

# wandb.run.name="Pretrained_Model,lr=0.00001,with augmentation"
# wandb.run.save()
train_loader, val_loader, test_loader, tiny_loader = get_dataloaders(data_dir, augment=True)
set_seed(42)
train_model(model, train_loader, val_loader, device,learning_rate=0.00001, epochs=20 )

# run.finish()

criterion = torch.nn.CrossEntropyLoss().to(device)
test_loss, test_acc = validate(model , test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")