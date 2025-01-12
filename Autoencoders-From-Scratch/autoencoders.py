import numpy as np
import matplotlib.pyplot as plt

# Layer implementations
class Dense:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))
        self.input = None
        self.output = None
        
    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(input_data, self.weights) + self.bias
        return self.output
    
    def backward(self, grad_output, learning_rate=0.01):
        grad_input = np.dot(grad_output, self.weights.T)
        grad_weights = np.dot(self.input.T, grad_output)
        grad_bias = np.sum(grad_output, axis=0, keepdims=True)
        
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias
        
        return grad_input

# Activation functions
class ReLU:
    def __init__(self):
        self.input = None
        
    def forward(self, input_data):
        self.input = input_data
        return np.maximum(0, input_data)
    
    def backward(self, grad_output, learning_rate=None):
        return grad_output * (self.input > 0)

class Sigmoid:
    def __init__(self):
        self.output = None
        
    def forward(self, input_data):
        self.output = 1 / (1 + np.exp(-input_data))
        return self.output
    
    def backward(self, grad_output, learning_rate=None):
        return grad_output * self.output * (1 - self.output)

# Loss function
class MSE:
    def __init__(self):
        self.input = None
        self.target = None
    
    def forward(self, input_data, target):
        self.input = input_data
        self.target = target
        return np.mean(np.square(input_data - target))
    
    def backward(self):
        return 2 * (self.input - self.target) / self.input.shape[0]

# Autoencoder implementation
class Autoencoder:
    def __init__(self, input_size, encoding_size):
        self.encoder_layers = [
            Dense(input_size, encoding_size * 2),
            ReLU(),
            Dense(encoding_size * 2, encoding_size),
            ReLU()
        ]
        
        self.decoder_layers = [
            Dense(encoding_size, encoding_size * 2),
            ReLU(),
            Dense(encoding_size * 2, input_size),
            Sigmoid()
        ]
        
        self.loss_function = MSE()
    
    def forward(self, x):
        # Encoder
        for layer in self.encoder_layers:
            x = layer.forward(x)
        
        # Decoder
        for layer in self.decoder_layers:
            x = layer.forward(x)
            
        return x
    
    def backward(self, grad_output, learning_rate):
        # Backward through decoder
        for layer in reversed(self.decoder_layers):
            grad_output = layer.backward(grad_output, learning_rate)
            
        # Backward through encoder
        for layer in reversed(self.encoder_layers):
            grad_output = layer.backward(grad_output, learning_rate)
    
    def train_step(self, x, learning_rate):
        # Forward pass
        output = self.forward(x)
        
        # Compute loss
        loss = self.loss_function.forward(output, x)
        
        # Backward pass
        grad_output = self.loss_function.backward()
        self.backward(grad_output, learning_rate)
        
        return loss
    
    def encode(self, x):
        """Encode the input data to the latent space"""
        encoded = x
        for layer in self.encoder_layers:
            encoded = layer.forward(encoded)
        return encoded

# Data generation
def generate_data(n_samples=1000):
    # Generate random points in a 2D space
    x = np.random.randn(n_samples, 2)
    # Add some structure (e.g., circular pattern)
    r = np.sqrt(x[:, 0]**2 + x[:, 1]**2)
    x = x[r < 2]
    return x

def main():
    # Generate data
    data = generate_data()
    
    # Create autoencoder (2D input -> 1D encoding -> 2D output)
    autoencoder = Autoencoder(input_size=2, encoding_size=1)
    
    # Training parameters
    epochs = 1000
    batch_size = 32
    learning_rate = 0.01
    
    # Training loop
    losses = []
    for epoch in range(epochs):
        epoch_losses = []
        
        # Train on mini-batches
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            loss = autoencoder.train_step(batch, learning_rate)
            epoch_losses.append(loss)
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
    
    # Plot training curve
    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()
    
    # Visualize results
    encoded_data = []
    decoded_data = []
    for x in data:
        x = x.reshape(1, -1)
        encoded = autoencoder.encode(x)
        decoded = autoencoder.forward(x)
        
        encoded_data.append(encoded.flatten())
        decoded_data.append(decoded.flatten())
    
    encoded_data = np.array(encoded_data)
    decoded_data = np.array(decoded_data)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.scatter(data[:, 0], data[:, 1], alpha=0.5)
    plt.title('Original Data')
    plt.grid(True)
    
    plt.subplot(132)
    plt.scatter(encoded_data, np.zeros_like(encoded_data), alpha=0.5)
    plt.title('Encoded Data (1D)')
    plt.grid(True)
    
    plt.subplot(133)
    plt.scatter(decoded_data[:, 0], decoded_data[:, 1], alpha=0.5)
    plt.title('Reconstructed Data')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()