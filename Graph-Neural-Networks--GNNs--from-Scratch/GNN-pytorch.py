import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from typing import Optional, Tuple

class GNNLayer(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int):
        super(GNNLayer, self).__init__(aggr='add')  # "add" aggregation
        self.linear = nn.Linear(in_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Transform node features
        x = self.linear(x)

        # Start propagating messages
        return self.propagate(edge_index, x=x)

    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        # x_j has shape [E, out_channels]
        return x_j

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        # aggr_out has shape [N, out_channels]
        return F.relu(aggr_out)

class GNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super(GNN, self).__init__()
        self.num_layers = num_layers
        
        # Create list of GNN layers
        self.convs = nn.ModuleList()
        self.convs.append(GNNLayer(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GNNLayer(hidden_dim, hidden_dim))
        
        self.convs.append(GNNLayer(hidden_dim, output_dim))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Initial node features: x
        # Graph connectivity: edge_index
        
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        return x

def main():
    # Create a simple graph
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0],
                             [1, 0, 2, 1, 3, 2, 0, 3]], dtype=torch.long)
    x = torch.randn(4, 3)  # 4 nodes with 3 features each
    
    # Create model
    model = GNN(input_dim=3, hidden_dim=64, output_dim=2)
    
    # Forward pass
    output = model(x, edge_index)
    print("Output shape:", output.shape)
    print("Output:\n", output)

    # Create a PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index)
    
    # Example of training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    model.train()
    for epoch in range(5):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        
        # Dummy loss - replace with your actual loss function
        loss = F.mse_loss(out, torch.randn_like(out))
        loss.backward()
        optimizer.step()
        
        print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}')

if __name__ == "__main__":
    main()