import os
import torch
import random
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from tqdm.notebook import tqdm
from utils import set_seed
from dataloader import get_dataloaders
from model import FlexibleCNN
import matplotlib.pyplot as plt
import argparse
import wandb
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

data_dir = "inaturalist_12K"  # Replace with your dataset path

parser = argparse.ArgumentParser(description='Neural Network Training Configuration')


parser.add_argument('-wp', '--wandb_project', type=str, default='DA6401_assignment2',
                    help='Project name used to track experiments in the Weights & Biases dashboard.')

parser.add_argument('-we', '--wandb_entity', type=str, default='myname',
                    help='Wandb Entity used to track experiments in the Weights & Biases dashboard.')

parser.add_argument('-sid', '--wandb_sweepid', type=str, default=None,
                    help='Wandb Sweep ID to log sweep runs in the Weights & Biases dashboard.')

parser.add_argument('-e', '--epochs', type=int, default=20,
                    help='Number of epochs to train the neural network.')

parser.add_argument('-b', '--batch_size', type=int, default=64,
                    help='Batch size used to train the neural network.')

parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001,
                    help='Learning rate used to optimize model parameters.')

parser.add_argument('-nf', '--num_filters', type=str, default='[32,64,128,256,512]',
                    help='List of filters for convolutional layers. Provide as a stringified list.')

parser.add_argument('-ks', '--kernel_size', type=int, default=3,
                    help='Kernel size for convolutional layers.')

parser.add_argument('-d', '--dropout', type=float, default=0.2,
                    help='Dropout rate used in the network.')

parser.add_argument('-bn', '--batch_norm', type=bool, default=True,
                    help='Whether to use batch normalization.')

parser.add_argument('-ca', '--conv_activation', type=str, default='SiLU',
                    choices=['ReLU', 'GELU', 'SiLU', 'Mish'],
                    help='Activation function used in convolutional layers.')

parser.add_argument('-aug', '--augment', type=bool, default=True,
                    help='Whether to apply data augmentation.')


config = parser.parse_args()

run = wandb.init(config = config,
                    project="DA6401_assignment2",
                    resume = "allow")
   
config = wandb.config
wandb.run.name=f"run-{wandb.run.id},num_filters-{config.num_filters},kernel_size-{config.kernel_size},dropout-{config.dropout},batch_norm-{config.batch_norm},conv_activation-{config.conv_activation},augment-{config.augment}"
# wandb.run.name=f"run on Test Data-{wandb.run.id},num_filters-{config.num_filters}"
wandb.run.save()
train_loader, val_loader, test_loader, tiny_loader = get_dataloaders(data_dir, augment=config.augment)
# loader = build_dataset(config.batch_size)
set_seed(42)
print("Train Loader Length: ", config.num_filters)
print("Train Loader Length: ", type(config.num_filters))

nn = FlexibleCNN(num_filters=config.num_filters , kernel_size = config.kernel_size, dropout = config.dropout ,batch_norm = config.batch_norm,
                conv_activation = config.conv_activation, )

nn.train_model(train_loader= train_loader, val_loader=val_loader, device= device,learning_rate=config.learning_rate, epochs=config.epochs,wandb = run)
run.finish()

