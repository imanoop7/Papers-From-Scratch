import numpy as np

class LeNet5Scratch:
    def __init__(self):
        # Initialize weights and biases for each layer using He initialization
        # Conv1: 6 filters of size 5x5
        self.conv1_weights = np.random.randn(6, 1, 5, 5) * np.sqrt(2. / (5*5))
        self.conv1_bias = np.zeros(6)
        
        # Conv2: 16 filters of size 5x5x6
        self.conv2_weights = np.random.randn(16, 6, 5, 5) * np.sqrt(2. / (5*5*6))
        self.conv2_bias = np.zeros(16)
        
        # Fully connected layers
        self.fc1_weights = np.random.randn(120, 400) * np.sqrt(2. / 400)
        self.fc1_bias = np.zeros(120)
        
        self.fc2_weights = np.random.randn(84, 120) * np.sqrt(2. / 120)
        self.fc2_bias = np.zeros(84)
        
        self.fc3_weights = np.random.randn(10, 84) * np.sqrt(2. / 84)
        self.fc3_bias = np.zeros(10)

    def conv2d(self, x, weights, bias, stride=1):
        """Perform 2D convolution"""
        n_filters, d_filter, h_filter, w_filter = weights.shape
        n_x, d_x, h_x, w_x = x.shape
        h_out = (h_x - h_filter) // stride + 1
        w_out = (w_x - w_filter) // stride + 1
        
        output = np.zeros((n_x, n_filters, h_out, w_out))
        
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * stride
                h_end = h_start + h_filter
                w_start = j * stride
                w_end = w_start + w_filter
                
                output[:, :, i, j] = np.sum(
                    x[:, np.newaxis, :, h_start:h_end, w_start:w_end] *
                    weights[np.newaxis, :, :, :, :],
                    axis=(2, 3, 4)
                ) + bias[np.newaxis, :]
                
        return output

    def avg_pool2d(self, x, kernel_size=2, stride=2):
        """Perform average pooling"""
        n_x, d_x, h_x, w_x = x.shape
        h_out = (h_x - kernel_size) // stride + 1
        w_out = (w_x - kernel_size) // stride + 1
        
        output = np.zeros((n_x, d_x, h_out, w_out))
        
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * stride
                h_end = h_start + kernel_size
                w_start = j * stride
                w_end = w_start + kernel_size
                output[:, :, i, j] = np.mean(x[:, :, h_start:h_end, w_start:w_end], axis=(2, 3))
        
        return output

    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)

    def softmax(self, x):
        """Softmax activation function"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, x):
        """Forward pass through the network"""
        # Ensure input is in the correct format (N, C, H, W)
        if len(x.shape) == 3:
            x = x[np.newaxis, :]
            
        # First convolutional layer + avg pool
        conv1 = self.conv2d(x, self.conv1_weights, self.conv1_bias)
        conv1_activated = self.relu(conv1)
        pool1 = self.avg_pool2d(conv1_activated)
        
        # Second convolutional layer + avg pool
        conv2 = self.conv2d(pool1, self.conv2_weights, self.conv2_bias)
        conv2_activated = self.relu(conv2)
        pool2 = self.avg_pool2d(conv2_activated)
        
        # Flatten
        flattened = pool2.reshape(pool2.shape[0], -1)
        
        # Fully connected layers
        fc1 = np.dot(flattened, self.fc1_weights.T) + self.fc1_bias
        fc1_activated = self.relu(fc1)
        
        fc2 = np.dot(fc1_activated, self.fc2_weights.T) + self.fc2_bias
        fc2_activated = self.relu(fc2)
        
        fc3 = np.dot(fc2_activated, self.fc3_weights.T) + self.fc3_bias
        output = self.softmax(fc3)
        
        return output

    def get_layer_shapes(self, x):
        """Get shapes of tensors at each layer (useful for debugging)"""
        shapes = {
            'input': x.shape
        }
        
        # Conv1
        conv1 = self.conv2d(x, self.conv1_weights, self.conv1_bias)
        shapes['conv1'] = conv1.shape
        
        conv1_activated = self.relu(conv1)
        pool1 = self.avg_pool2d(conv1_activated)
        shapes['pool1'] = pool1.shape
        
        # Conv2
        conv2 = self.conv2d(pool1, self.conv2_weights, self.conv2_bias)
        shapes['conv2'] = conv2.shape
        
        conv2_activated = self.relu(conv2)
        pool2 = self.avg_pool2d(conv2_activated)
        shapes['pool2'] = pool2.shape
        
        # Flatten
        flattened = pool2.reshape(pool2.shape[0], -1)
        shapes['flattened'] = flattened.shape
        
        # FC layers
        fc1 = np.dot(flattened, self.fc1_weights.T) + self.fc1_bias
        shapes['fc1'] = fc1.shape
        
        fc2 = np.dot(self.relu(fc1), self.fc2_weights.T) + self.fc2_bias
        shapes['fc2'] = fc2.shape
        
        fc3 = np.dot(self.relu(fc2), self.fc3_weights.T) + self.fc3_bias
        shapes['fc3'] = fc3.shape
        
        return shapes

if __name__ == "__main__":
    # Create a sample input (1 image, 1 channel, 32x32)
    sample_input = np.random.randn(1, 1, 32, 32)
    
    # Initialize the model
    model = LeNet5Scratch()
    
    # Forward pass
    output = model.forward(sample_input)
    
    print("LeNet-5 From Scratch (NumPy Implementation)")
    print("-" * 45)
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"\nOutput probabilities:\n{output[0]}")
    
    # Print shapes at each layer
    print("\nTensor shapes at each layer:")
    shapes = model.get_layer_shapes(sample_input)
    for layer, shape in shapes.items():
        print(f"{layer:10s}: {shape}")
