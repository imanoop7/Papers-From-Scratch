import numpy as np
from typing import List, Tuple, Dict

class Graph:
    def __init__(self, num_nodes: int):
        """Initialize a graph with a given number of nodes."""
        self.num_nodes = num_nodes
        self.edges: List[Tuple[int, int]] = []
        self.node_features = np.zeros((num_nodes, 1))  # Default 1-dimensional features
        
    def add_edge(self, source: int, target: int):
        """Add an edge between source and target nodes."""
        if source < self.num_nodes and target < self.num_nodes:
            self.edges.append((source, target))
            # Add reverse edge for undirected graph
            self.edges.append((target, source))
            
    def set_node_features(self, features: np.ndarray):
        """Set the feature matrix for all nodes."""
        assert features.shape[0] == self.num_nodes, "Feature matrix must match number of nodes"
        self.node_features = features

class GNNLayer:
    def __init__(self, input_dim: int, output_dim: int):
        """Initialize a GNN layer with given input and output dimensions."""
        # Initialize weights randomly
        self.W = np.random.randn(input_dim, output_dim) * 0.1
        self.b = np.zeros(output_dim)
        
        # For storing gradients
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        
    def forward(self, graph: Graph) -> np.ndarray:
        """Forward pass of the GNN layer."""
        # Initialize message matrix
        messages = np.zeros((graph.num_nodes, self.W.shape[1]))
        
        # Message passing phase
        for source, target in graph.edges:
            # Compute message from source to target
            message = np.dot(graph.node_features[source], self.W) + self.b
            messages[target] += message
            
        # Update phase - apply ReLU activation
        return np.maximum(0, messages)
    
    def backward(self, graph: Graph, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass of the GNN layer."""
        # Gradient through ReLU
        grad_relu = (grad_output > 0).astype(float)
        
        # Initialize gradient for input features
        grad_input = np.zeros_like(graph.node_features)
        
        # Compute gradients for weights and biases
        for source, target in graph.edges:
            # Gradient w.r.t weights
            self.dW += np.outer(graph.node_features[source], grad_relu[target])
            # Gradient w.r.t bias
            self.db += grad_relu[target]
            # Gradient w.r.t input features
            grad_input[source] += np.dot(grad_relu[target], self.W.T)
            
        return grad_input

class GNN:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """Initialize a two-layer GNN."""
        self.layer1 = GNNLayer(input_dim, hidden_dim)
        self.layer2 = GNNLayer(hidden_dim, output_dim)
        
    def forward(self, graph: Graph) -> np.ndarray:
        """Forward pass through the entire GNN."""
        # First layer
        hidden = self.layer1.forward(graph)
        
        # Create temporary graph with hidden features
        hidden_graph = Graph(graph.num_nodes)
        hidden_graph.edges = graph.edges
        hidden_graph.node_features = hidden
        
        # Second layer
        output = self.layer2.forward(hidden_graph)
        return output
    
    def backward(self, graph: Graph, grad_output: np.ndarray):
        """Backward pass through the entire GNN."""
        # Create temporary graph with hidden features
        hidden = self.layer1.forward(graph)
        hidden_graph = Graph(graph.num_nodes)
        hidden_graph.edges = graph.edges
        hidden_graph.node_features = hidden
        
        # Backward pass through second layer
        grad_hidden = self.layer2.backward(hidden_graph, grad_output)
        
        # Backward pass through first layer
        grad_input = self.layer1.backward(graph, grad_hidden)
        return grad_input

# Example usage
def main():
    # Create a simple graph with 4 nodes
    graph = Graph(4)
    
    # Add edges (creating a square graph)
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    graph.add_edge(3, 0)
    
    # Set random node features
    node_features = np.random.randn(4, 3)  # 3-dimensional features
    graph.set_node_features(node_features)
    
    # Create GNN model
    model = GNN(input_dim=3, hidden_dim=4, output_dim=2)
    
    # Forward pass
    output = model.forward(graph)
    print("Output shape:", output.shape)
    print("Output:\n", output)
    
    # Simulate backward pass with random gradients
    grad_output = np.random.randn(*output.shape)
    model.backward(graph, grad_output)

if __name__ == "__main__":
    main()