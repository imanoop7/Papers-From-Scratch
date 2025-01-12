import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Combined weights for all gates
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        
    def forward(self, x, hidden=None):
        return self.lstm(x, hidden if hidden else (torch.zeros(1, self.hidden_size), 
                                                 torch.zeros(1, self.hidden_size)))

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Output layer
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # Forward propagate LSTM
        lstm_out, _ = self.lstm(x)
        # Get the output from the last time step
        out = self.fc(lstm_out[:, -1, :])
        return out

def generate_sequence(length=100):
    """Generate a simple sine wave sequence for demonstration"""
    time_steps = np.linspace(0, 10, length)
    data = np.sin(time_steps)
    data = data.reshape(-1, 1)
    return data

def prepare_data(sequence, seq_length):
    """Prepare data for training"""
    X, y = [], []
    for i in range(len(sequence) - seq_length):
        X.append(sequence[i:(i + seq_length)])
        y.append(sequence[i + seq_length])
    return torch.FloatTensor(X), torch.FloatTensor(y)

def train_model():
    # Parameters
    input_size = 1
    hidden_size = 32
    seq_length = 10
    num_epochs = 100
    learning_rate = 0.01

    # Generate data
    data = generate_sequence(200)
    X, y = prepare_data(data, seq_length)
    
    # Create model
    model = LSTM(input_size, hidden_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    return model, X, y

def plot_results(model, X, y):
    """Plot the predictions against actual values"""
    model.eval()
    with torch.no_grad():
        predictions = model(X)
        
    # Convert to numpy for plotting
    predictions = predictions.numpy()
    y = y.numpy()
    
    plt.figure(figsize=(10, 6))
    plt.plot(y, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.legend()
    plt.title('LSTM Predictions vs Actual Values')
    plt.show()

if __name__ == "__main__":
    # Train the model
    model, X, y = train_model()
    
    # Plot results
    plot_results(model, X, y)