# LeNet-5 Implementation from Scratch

This repository contains a PyTorch implementation of the LeNet-5 Convolutional Neural Network architecture, originally proposed by Yann LeCun et al. in their 1998 paper "Gradient-Based Learning Applied to Document Recognition."

## Architecture

The implemented LeNet-5 architecture follows the original design:

- **Input Layer**: 32x32 grayscale images
- **Convolutional Layer 1**: 6 filters (5x5), stride 1
- **Average Pooling Layer 1**: 2x2, stride 2
- **Convolutional Layer 2**: 16 filters (5x5), stride 1
- **Average Pooling Layer 2**: 2x2, stride 2
- **Fully Connected Layer 1**: 120 neurons
- **Fully Connected Layer 2**: 84 neurons
- **Output Layer**: 10 neurons (for digits 0-9)

## Requirements

- Python 3.7+
- PyTorch 2.0.0+
- torchvision 0.15.0+
- numpy 1.21.0+
- matplotlib 3.4.0+

## Installation

1. Clone the repository:
```bash
git clone https://github.com/imanoop7/LeNet-From-Scratch
cd LeNet-From-Scratch
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

```python
from lenet import LeNet5
import torch

# Initialize the model
model = LeNet5()

# Create a sample input (batch_size, channels, height, width)
input_tensor = torch.randn(1, 1, 32, 32)

# Get predictions
output = model(input_tensor)
```

## Model Details

The implementation includes:
- Convolutional layers with no padding
- Average pooling layers
- ReLU activation functions
- Fully connected layers
- The model expects input images of size 32x32 pixels (grayscale)
