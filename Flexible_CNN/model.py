import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import logging

# ---------------- Logging Setup ---------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

class FlexibleCNN(nn.Module):
    """
    A flexible convolutional neural network with configurable layers, activations,
    pooling, and dropout options.
    """
    def __init__(self, 
                 input_shape=(3, 224, 224),
                 num_classes=10,
                 num_filters=[64, 64, 64, 64, 64],
                 kernel_size=5,
                 dropout=0.3,
                 batch_norm=True,
                 conv_activation='ReLU',
                 pooling='max',
                 pooling_kernel_size=2,
                 pooling_stride=2,
                 output_activation='softmax',
                 dense_neurons=512):
        """
        Initialize the FlexibleCNN model architecture.

        Args:
            input_shape (tuple): Shape of the input image.
            num_classes (int): Number of output classes.
            num_filters (list): Number of filters in each convolutional layer.
            kernel_size (int): Kernel size for convolutions.
            dropout (float): Dropout rate before the output layer.
            batch_norm (bool): Whether to use batch normalization.
            conv_activation (str): Activation function for conv layers.
            pooling (str): Type of pooling to apply.
            pooling_kernel_size (int): Pooling kernel size.
            pooling_stride (int): Pooling stride.
            output_activation (str): Final layer activation (softmax/none).
            dense_neurons (int): Number of neurons in dense layer.
        """
        super(FlexibleCNN, self).__init__()

        self.conv_layers = nn.ModuleList()
        in_channels = input_shape[0]

        logging.info("Building convolutional layers...")
        for filters in num_filters:
            self.conv_layers.append(nn.Conv2d(in_channels, filters, kernel_size=kernel_size, padding=1))
            if batch_norm:
                self.conv_layers.append(nn.BatchNorm2d(filters))
            self.conv_layers.append(getattr(nn, conv_activation)())
            if pooling == 'max':
                self.conv_layers.append(nn.MaxPool2d(kernel_size=pooling_kernel_size, stride=pooling_stride))
            in_channels = filters

        self.flatten_size = self._get_flatten_size(input_shape)
        logging.info(f"Flatten size after conv layers: {self.flatten_size}")

        self.dense_layer = nn.Linear(self.flatten_size, dense_neurons)
        self.dense_activation = getattr(nn, conv_activation)()
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(dense_neurons, num_classes)
        self.output_activation = nn.Softmax(dim=1) if output_activation == 'softmax' else nn.Identity()

    def _get_flatten_size(self, input_shape):
        """
        Compute the flattened feature size after all conv layers.
        """
        x = torch.rand(1, *input_shape)
        with torch.no_grad():
            for layer in self.conv_layers:
                x = layer(x)
        return x.view(1, -1).size(1)

    def forward(self, x):
        """
        Forward pass through the model.
        """
        for layer in self.conv_layers:
            x = layer(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dense_layer(x)
        # x = self.dense_activation(x)  # Optional
        x = self.dropout(x)
        x = self.output_layer(x)
        return x

    def compute_total_params_and_ops(self, input_shape):
        """
        Calculate total number of parameters and approximate operations (FLOPs).
        """
        total_params = 0
        total_ops = 0
        h, w = input_shape[1], input_shape[2]
        in_channels = input_shape[0]

        for layer in self.conv_layers:
            if isinstance(layer, nn.Conv2d):
                k = layer.kernel_size[0]
                out_h, out_w = h // 2, w // 2
                total_ops += out_h * out_w * layer.out_channels * in_channels * k * k
                total_params += (in_channels * k * k + 1) * layer.out_channels
                h, w, in_channels = out_h, out_w, layer.out_channels

        total_params += (self.flatten_size + 1) * self.dense_layer.out_features
        total_ops += self.flatten_size * self.dense_layer.out_features

        total_params += (self.dense_layer.out_features + 1) * self.output_layer.out_features
        total_ops += self.dense_layer.out_features * self.output_layer.out_features

        return total_params, total_ops

    def train_model(self, train_loader, val_loader, device, learning_rate=0.1, epochs=5, wandb=None):
        """
        Train the model using the given training and validation data loaders.

        Args:
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader): Validation data loader.
            device (torch.device): Device to train on (CPU or CUDA).
            learning_rate (float): Learning rate for optimizer.
            epochs (int): Number of training epochs.
            wandb (wandb.Run): Optional Weights & Biases logger.
        """
        self.to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-3)

        logging.info("Starting training...")
        for epoch in tqdm(range(epochs), desc="Training Progress", position=0):
            self.train()
            running_loss, correct, total = 0.0, 0, 0

            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", position=1):
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                optimizer.zero_grad()
                outputs = self(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            train_loss = running_loss / total
            train_acc = correct / total

            val_loss, val_acc = self.validate(val_loader, criterion, device)

            logging.info(f"Epoch {epoch+1}/{epochs} - Train Acc: {train_acc:.4f}, "
                         f"Val Acc: {val_acc:.4f}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if wandb:
                wandb.log({
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc
                })

    def validate(self, val_loader, criterion, device):
        """
        Evaluate model performance on the validation set.

        Args:
            val_loader (DataLoader): Validation data loader.
            criterion (Loss): Loss function.
            device (torch.device): Device to validate on.

        Returns:
            Tuple[float, float]: Validation loss and accuracy.
        """
        self.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = self(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return val_loss / total, correct / total
