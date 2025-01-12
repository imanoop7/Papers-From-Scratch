# ResNet Implementation from Scratch using Pure Python

This repository contains a pure Python implementation of the ResNet (Residual Neural Network) architecture. The implementation focuses on understanding the core concepts of ResNet without relying on deep learning frameworks.

## Implementation Details

The implementation includes:

- Basic building blocks of ResNet:
  - Convolutional layers
  - Batch normalization
  - ReLU activation
  - Residual connections

- Network architecture:
  - Initial convolutional layer
  - Four residual layer groups
  - Global average pooling
  - Final fully connected layer

## Project Structure

```
.
├── resnet.py       # Core ResNet implementation
└── train.py        # Training utilities and data loading
```

## How It Works

1. **Residual Blocks**: The core innovation of ResNet is its residual learning framework. Instead of learning direct mappings, the network learns residual functions with reference to the layer inputs.

2. **Skip Connections**: These connections allow the network to bypass one or more layers, helping to solve the vanishing gradient problem in deep networks.

3. **Batch Normalization**: Each convolutional layer is followed by batch normalization to stabilize the learning process.

## Features

- Pure Python implementation
- No external deep learning frameworks required
- Educational resource for understanding ResNet architecture
- Modular and extensible code structure

## Requirements

- Python 3.6+
- NumPy

## Usage

1. Clone the repository:
```bash
git clone https://github.com/imanoop7/Resnet-from-Scratch.git
cd Resnet-from-Scratch
```

2. Run the training script:
```bash
python train.py
```

## References

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) - Original ResNet paper
- [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) - ResNet architecture improvements

## License

MIT License