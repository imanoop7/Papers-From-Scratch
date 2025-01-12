# RNN from Scratch Using Python

This project implements a simple Recurrent Neural Network (RNN) model from scratch in Python without using any machine learning frameworks like TensorFlow or PyTorch. The project is inspired by the "Neural Probabilistic Language Model" paper by Bengio et al. (2003) and focuses on training a model to predict the next word (or number) in a sequence.

## Features

- Implement a simple neural language model from scratch.
- Support for training a model to predict the next element in a sequence using context windows.
- Gradient-based training with backpropagation.
- Use of word embeddings to represent input sequences.
- Implements softmax activation for the output layer and cross-entropy loss for training.
- Demonstrates forward and backward propagation, training, and prediction.

## Requirements

- Python 3.x
- NumPy library

## Installation

1. Clone the repository:

    ```bash
    https://github.com/imanoop7/RNN-from-Scratch-using-Python
    ```

2. Change into the project directory:

    ```bash
    cd rnn-from-scratch
    ```

3. Ensure you have the required libraries installed:

    ```bash
    pip install numpy
    ```

## Project Structure

- `rnn.py`: The main Python script implementing the RNN model.
- `README.md`: This readme file explaining the project, usage, and details.

## Implementation Details

The project consists of a simple neural network model that learns to predict the next element in a sequence using the previous elements as context. The architecture includes:

1. **Embedding Layer:** Maps each word in the vocabulary to a dense vector of fixed size.
2. **Hidden Layer:** Uses a tanh activation function to learn representations of the input context.
3. **Output Layer:** Uses a softmax activation to generate probabilities for the next word.
4. **Loss Function:** Uses cross-entropy loss to train the model, maximizing the log-likelihood of the correct word given the context.
5. **Training Loop:** Implements backpropagation through time to update the model's weights based on the gradients.

## Running the Project

1. **Generating the Dataset:** The script can generate a simple dataset consisting of sequences of numbers, where the goal is to predict the next number given the previous ones.

2. **Training the Model:** Train the model with the generated dataset. The script uses gradient descent to minimize the loss and print the loss at each epoch.

    ```bash
    python rnn.py
    ```

3. **Predicting the Next Word:** The script can predict the next word based on a given context after training.

## Code Overview

The main components of the `rnn.py` file include:

### NeuralLanguageModel Class

This class implements the neural language model. Key methods:

- `__init__(self, vocab_size, embedding_dim, context_size, hidden_size, learning_rate=0.01, weight_decay=0.001)`: Initializes the model parameters.
- `forward(self, context_word_indices)`: Performs the forward pass to compute the output probabilities.
- `compute_loss(self, y_hat, target_index)`: Calculates the cross-entropy loss for the given prediction.
- `backward(self, context_word_indices, y_hat, hidden_activations, target_index)`: Performs the backward pass to update the model parameters.
- `train(self, training_data, epochs=10)`: Trains the model on the provided dataset.
- `predict(self, context_word_indices)`: Predicts the next word based on the input context.

### Dataset Generation

The script includes a helper function to generate a simple arithmetic sequence dataset for demonstration purposes.

### Training and Testing

The training process minimizes the cross-entropy loss using gradient descent, and the testing demonstrates the model's ability to predict the next number in a sequence.

## Example Usage

1. Run the script to train the model:

    ```bash
    python rnn.py
    ```

2. Observe the printed loss at each epoch, indicating the training progress.

3. The script will display the predicted next word based on a given context after training.

## Possible Extensions

- **Use Larger and More Complex Datasets:** Extend the model to train on real text data for language modeling tasks.
- **Add Support for GRU or LSTM Cells:** Improve the model's performance on long sequences by using more advanced RNN architectures.
- **Implement Regularization Techniques:** Add dropout or other regularization techniques to reduce overfitting.
- **Parameter Tuning:** Experiment with different hyperparameters, such as the learning rate, hidden layer size, and context window size, to improve model performance.

## References

- Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. (2003). A neural probabilistic language model. *Journal of Machine Learning Research*, 3, 1137-1155.

## License

This project is licensed under the MIT License.
