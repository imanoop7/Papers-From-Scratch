# Autoencoders From Scratch

A pure Python implementation of an autoencoder neural network without using any deep learning frameworks. This implementation demonstrates the core concepts of autoencoders using only NumPy for numerical computations.

## Overview

This project implements an autoencoder that can:
- Compress high-dimensional data into a lower-dimensional latent space
- Reconstruct the original data from the compressed representation
- Learn meaningful data representations through unsupervised learning

## Requirements

- Python 3.6+
- NumPy
- Matplotlib

Install dependencies:
```bash
pip install numpy matplotlib
```

## Usage

Run the example:
```bash
python autoencoders.py
```

The script will:
1. Generate synthetic 2D data
2. Train an autoencoder to compress it to 1D
3. Reconstruct the data back to 2D
4. Visualize the results:
   - Original data distribution
   - Encoded (compressed) representation
   - Reconstructed data

## Implementation Details

The autoencoder is implemented from scratch with:
- Dense (fully connected) layers
- ReLU and Sigmoid activation functions
- Mean Squared Error loss
- Mini-batch gradient descent optimization

Architecture:
- Encoder: Input (2D) → Dense+ReLU → Dense+ReLU → Latent Space (1D)
- Decoder: Latent Space (1D) → Dense+ReLU → Dense+Sigmoid → Output (2D)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
