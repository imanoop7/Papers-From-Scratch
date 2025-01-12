import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # First Convolutional Layer
        # Input: 32x32x1, Output: 28x28x6
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        
        # Second Convolutional Layer
        # Input: 14x14x6 (after pooling), Output: 10x10x16
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # Input: 400 (16*5*5), Output: 120
        self.fc2 = nn.Linear(120, 84)          # Input: 120, Output: 84
        self.fc3 = nn.Linear(84, 10)           # Input: 84, Output: 10 (classes)

    def forward(self, x):
        # First Conv + Pool Layer
        x = self.conv1(x)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(x)

        # Second Conv + Pool Layer
        x = self.conv2(x)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(x)

        # Flatten the tensor
        x = x.view(-1, 16 * 5 * 5)

        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

if __name__ == "__main__":
    # Create a sample input tensor
    batch_size = 1
    input_tensor = torch.randn(batch_size, 1, 32, 32)
    
    # Initialize the model
    model = LeNet5()
    
    # Forward pass
    output = model(input_tensor)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print("\nModel Architecture:")
    print(model)
