# DA6401 : Deep Learning (Assignment:02)

## Neural Network Approaches for Image Classification: Report

This report details two distinct approaches to image classification on the iNaturalist dataset: a custom Flexible CNN architecture built from scratch and a transfer learning approach using a pre-trained ResNet50 model.

## Flexible CNN: Custom Architecture Development

### Architecture Overview

The Flexible CNN implements a fully customizable convolutional neural network architecture with the following configurable components:

- **Convolutional Layers**: Configurable number of layers with variable filter counts
- **Kernel Size**: Adjustable convolution kernel dimensions
- **Activation Functions**: Multiple options including ReLU, GELU, SiLU, and Mish
- **Regularization**: Optional batch normalization and configurable dropout rates
- **Pooling**: Max pooling with adjustable parameters
- **Dense Layer**: Configurable number of neurons in the fully connected layer

The architecture is defined in `model.py` and follows a standard CNN pattern with stacked convolutional blocks, flattening, and fully connected layers before classification.

### Implementation Details

The `FlexibleCNN` class provides extensive configuration options:

```python
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
```

The model dynamically builds convolutional layers based on the `num_filters` parameter, allowing for experimentation with different network depths and widths. The class also includes built-in methods for training and validation, along with performance analysis capabilities.

### Hyperparameter Optimization

Hyperparameter tuning is managed through Weights \& Biases sweeps configured in `config.yaml`. The sweep explores:

- **Filter Configurations**: Various patterns including symmetrical, inverted, bottleneck, and uniform structures
- **Kernel Sizes**: 3×3, 5×5, and 7×7
- **Regularization**: Different dropout rates (0, 0.2, 0.3) and batch normalization options
- **Activation Functions**: ReLU, GELU, SiLU, Mish
- **Learning Rates**: 1e-5, 1e-4, 1e-3

The optimization process uses random search with validation accuracy as the target metric to maximize.

### Training Methodology

Training is coordinated in `train.py` with command-line arguments for hyperparameter configuration. Key training components include:

- **Loss Function**: Cross-Entropy Loss
- **Optimizer**: Adam with configurable learning rate and weight decay (1e-3)
- **Progress Tracking**: tqdm progress bars and logging
- **Experiment Monitoring**: Integration with Weights \& Biases


## Pre-Trained Model: Transfer Learning Approach

### Architecture Overview

The pre-trained approach leverages a ResNet50 model with weights pre-trained on ImageNet:

```python
model = models.resnet50(pretrained=True)
```

ResNet50 is a deep residual network with 50 layers that addresses the vanishing gradient problem through skip connections. The architecture includes:

- Initial 7×7 convolution and max pooling
- Four residual blocks with bottleneck architecture
- Global average pooling
- Fully connected output layer


### Transfer Learning Implementation

The implementation follows a standard transfer learning pattern:

1. **Load Pre-Trained Model**: Initialize ResNet50 with pre-trained weights
2. **Freeze Base Layers**: Disable gradient updates for all layers except the final classifier

```python
for param in model.parameters():
    param.requires_grad = False
```

3. **Replace Classifier**: Adapt the final fully connected layer for the target task (10 classes)

```python
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)
```


This approach leverages the feature extraction capabilities of a model trained on millions of images while adapting only the final classification layer to the specific task.

### Training Methodology

The training process for the pre-trained model uses a conservative approach:

- **Learning Rate**: Very small (1e-5) to avoid disrupting pre-trained features
- **Epochs**: Fixed at 20 epochs
- **Optimizer**: Adam (same as Flexible CNN)
- **Data Augmentation**: Enabled by default

Unlike the Flexible CNN approach, the pre-trained implementation uses fixed hyperparameters without extensive tuning.

## Data Processing

Both approaches share the same data pipeline implemented in `dataloader.py`:

- **Preprocessing**: Resizing to 224×224 and normalization with dataset-specific statistics:

```python
transforms.Normalize(mean=[0.4712, 0.4600, 0.3896], std=[0.1935, 0.1877, 0.1843])
```

- **Data Augmentation**: Optional transformations including horizontal flips and rotations
- **Dataset Splitting**: Stratified train/validation splits to maintain class balance


## Training and Evaluation Instructions

### Flexible CNN

#### Training Instructions

1. **Setup Environment**
    - Ensure PyTorch and dependencies are installed
    - Clone the repository and navigate to the project directory
2. **Prepare Dataset**
    - Place the iNaturalist dataset in the specified directory
    - Ensure the folder structure follows: `[data_dir]/train/[class_folders]` and `[data_dir]/val/[class_folders]`
3. **Configure Hyperparameters**
    - Edit command line arguments or use default values

```bash
python train.py --batch_size 64 --learning_rate 0.0001 --num_filters "[32,64,128,256,512]" --kernel_size 3 --dropout 0.2 --batch_norm True --conv_activation "SiLU" --augment True --epochs 20
```

4. **Run Training (Single Configuration)**

```bash
python train.py 
```

5. **Run Hyperparameter Sweep**

```bash
wandb sweep config.yaml
wandb agent [sweep_id]
```

6. **Monitor Training Progress**
    - Observe the terminal output showing epoch progress with tqdm bars
    - Track metrics through Weights \& Biases dashboard (if configured)
    - Training and validation accuracy/loss are logged for each epoch

#### Evaluation Instructions

1. **Evaluate on Test Set**
    - After training completion, the model is automatically evaluated on the test set
    - Test accuracy and loss are reported in the terminal
2. **Custom Evaluation**

```python
# Load trained model
model = FlexibleCNN(
    num_filters=[32, 64, 128, 256, 512],
    kernel_size=3,
    dropout=0.2,
    batch_norm=True,
    conv_activation='SiLU',
)
model.load_state_dict(torch.load('path_to_saved_model.pth'))

# Evaluate
from utils import validate
criterion = torch.nn.CrossEntropyLoss().to(device)
test_loss, test_acc = validate(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
```

3. **Analyzing Hyperparameter Sweep Results**
    - Access the W\&B dashboard to compare different model configurations
    - Sort by validation accuracy to identify optimal hyperparameters
    - Examine learning curves to detect overfitting or poor convergence

### Pre-Trained ResNet50

#### Training Instructions

1. **Setup Environment**
    - Same environment setup as Flexible CNN
    - Ensure torchvision is installed for accessing pre-trained models
2. **Prepare Dataset**
    - Same dataset preparation as Flexible CNN
3. **Run Training**

```bash
python train_pretrained.py
```

The pre-trained implementation uses fixed hyperparameters:
    - Learning rate: 0.00001
    - Epochs: 20
    - Batch size: Default from dataloader (32)
    - Data augmentation: Enabled
4. **Monitor Training Progress**
    - Training progress is displayed in the console with per-epoch metrics

#### Evaluation Instructions

1. **Automated Evaluation**
    - After training completion, the script automatically evaluates on the test set

```python
test_loss, test_acc = validate(model, test_loader, criterion, device)
logging.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
```

2. **Manual Evaluation**

```python
# Load saved model
model = models.resnet50(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)
model.load_state_dict(torch.load('path_to_saved_model.pth'))

# Evaluate
model.eval()
from utils import validate
criterion = torch.nn.CrossEntropyLoss().to(device)
test_loss, test_acc = validate(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
```

3. **Model Inspection**
    - To inspect the model architecture:

```python
print(model)
```

    - To count model parameters:

```python
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params}, Trainable parameters: {trainable_params}")
```


## Comparison and Key Differences

| Aspect | Flexible CNN | Pre-Trained ResNet50 |
| :-- | :-- | :-- |
| **Model Source** | Built from scratch | Transfer learning from ImageNet |
| **Parameter Count** | Variable (configurable) | Fixed (~25 million, mostly frozen) |
| **Trainable Parameters** | All parameters | Only final layer (~2K parameters) |
| **Training Time** | Longer | Significantly shorter |
| **Hyperparameter Focus** | Architecture design | Fine-tuning learning rate |
| **Implementation Complexity** | Higher (custom architecture) | Lower (existing architecture) |

## Conclusion

The two approaches represent fundamentally different philosophy in neural network development:

1. **Flexible CNN** offers complete control over network architecture with educational value in understanding CNN components, but requires extensive hyperparameter tuning and longer training time.
2. **Pre-Trained ResNet50** leverages transfer learning for rapid deployment with potentially higher performance on limited data, but offers less flexibility in architecture design.

The choice between approaches depends on the specific requirements:

- For understanding CNN design principles: Flexible CNN
- For quick deployment with strong performance: Pre-trained approach
- For optimal performance on niche datasets: A hybrid approach with partial fine-tuning


