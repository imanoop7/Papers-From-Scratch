import numpy as np

class NeuralLanguageModel:
    def __init__(self, vocab_size, embedding_dim, context_size, hidden_size, learning_rate=0.01, weight_decay=0.001):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Initialize the embedding matrix
        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.01
        
        # Initialize weights for the feed-forward network
        self.W_hidden = np.random.randn(hidden_size, context_size * embedding_dim) * 0.01
        self.b_hidden = np.zeros((hidden_size, 1))
        self.W_output = np.random.randn(vocab_size, hidden_size) * 0.01
        self.b_output = np.zeros((vocab_size, 1))
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)
    
    def forward(self, context_word_indices):
        # Step 1: Look up embeddings for each word in the context
        context_embeddings = np.concatenate([self.embeddings[idx].reshape(-1, 1) for idx in context_word_indices], axis=0)
        
        # Step 2: Compute hidden layer activations
        hidden_activations = np.tanh(np.dot(self.W_hidden, context_embeddings) + self.b_hidden)
        
        # Step 3: Compute output layer activations
        output_activations = np.dot(self.W_output, hidden_activations) + self.b_output
        
        # Step 4: Apply softmax to get the probability distribution
        y_hat = self.softmax(output_activations)
        
        return y_hat, hidden_activations
    
    def compute_loss(self, y_hat, target_index):
        # Cross-entropy loss
        return -np.log(y_hat[target_index, 0])
    
    def backward(self, context_word_indices, y_hat, hidden_activations, target_index):
        # Gradient for output layer
        dL_dy = y_hat
        dL_dy[target_index] -= 1  # Subtract 1 from the true class
        
        # Gradients for weights and biases in the output layer
        dL_dW_output = np.dot(dL_dy, hidden_activations.T)
        dL_db_output = dL_dy
        
        # Gradients for hidden layer
        dL_dhidden = np.dot(self.W_output.T, dL_dy) * (1 - hidden_activations ** 2)  # Derivative of tanh
        
        # Gradients for weights and biases in the hidden layer
        context_embeddings = np.concatenate([self.embeddings[idx].reshape(-1, 1) for idx in context_word_indices], axis=0)
        dL_dW_hidden = np.dot(dL_dhidden, context_embeddings.T)
        dL_db_hidden = dL_dhidden
        
        # Update weights and biases for output layer
        self.W_output -= self.learning_rate * (dL_dW_output + self.weight_decay * self.W_output)
        self.b_output -= self.learning_rate * dL_db_output
        
        # Update weights and biases for hidden layer
        self.W_hidden -= self.learning_rate * (dL_dW_hidden + self.weight_decay * self.W_hidden)
        self.b_hidden -= self.learning_rate * dL_db_hidden
        
        # Gradients for the embeddings
        dL_dembeddings = np.dot(self.W_hidden.T, dL_dhidden).reshape(self.context_size, self.embedding_dim)
        
        # Update embeddings
        for i, idx in enumerate(context_word_indices):
            self.embeddings[idx] -= self.learning_rate * (dL_dembeddings[i, :].reshape(-1) + self.weight_decay * self.embeddings[idx])
    
    def train(self, training_data, epochs=10):
        for epoch in range(epochs):
            total_loss = 0
            for context_word_indices, target_index in training_data:
                # Forward pass
                y_hat, hidden_activations = self.forward(context_word_indices)
                
                # Compute loss
                loss = self.compute_loss(y_hat, target_index)
                total_loss += loss
                
                # Backward pass
                self.backward(context_word_indices, y_hat, hidden_activations, target_index)
            
            # Print the average loss per epoch
            print(f'Epoch {epoch + 1}, Loss: {total_loss / len(training_data):.4f}')

    def predict(self, context_word_indices):
        # Forward pass to get the probabilities for the next word
        y_hat, _ = self.forward(context_word_indices)
        
        # Choose the word with the highest probability
        predicted_word_index = np.argmax(y_hat)
        
        return predicted_word_index

# Example usage
# Assume we have a small vocabulary of size 10 for demonstration purposes
vocab_size = 10
embedding_dim = 5
context_size = 3
hidden_size = 8
learning_rate = 0.01
weight_decay = 0.001

# Create the neural language model
model = NeuralLanguageModel(vocab_size, embedding_dim, context_size, hidden_size, learning_rate, weight_decay)

# Generate some dummy training data
# Each training example is a tuple (context_word_indices, target_word_index)
# For example, ([0, 1, 2], 3) means predicting word 3 given context words [0, 1, 2]
training_data = [
    ([0, 1, 2], 3),
    ([1, 2, 3], 4),
    ([2, 3, 4], 5),
    ([3, 4, 5], 6),
    ([4, 5, 6], 7),
    ([5, 6, 7], 8) 
]

# Train the model
model.train(training_data, epochs=100)


# Test the prediction
test_context = [5, 6, 7]  # Example context
predicted_word = model.predict(test_context)

print(f"Given the context {test_context}, the predicted next word index is: {predicted_word}")
