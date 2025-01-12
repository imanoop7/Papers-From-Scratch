# Gradient Descent from Scratch using NumPy for Linear Regression

This repository contains an implementation of gradient descent from scratch using only NumPy for linear regression. The goal is to understand the fundamental concepts of gradient descent and linear regression without relying on high-level libraries.

## Files

- `grad.ipynb`: Jupyter notebook containing the implementation and step-by-step explanation of gradient descent for linear regression.
- `grad.py`: Python script with the complete implementation of gradient descent for linear regression.

## Implementation Details

### Data Generation

We generate sample data for linear regression using the following equation:
\[ y = 4 + 3X + \text{noise} \]
where `X` is a random variable and `noise` is Gaussian noise.

### Functions

- `initialize_parameters()`: Initializes the weight and bias parameters randomly.
- `predict(X, weight, bias)`: Predicts the output `y` given input `X`, weight, and bias.
- `compute_cost(X, y, weight, bias)`: Computes the Mean Squared Error (MSE) cost.
- `gradient_descent(X, y, weight, bias, learning_rate, iterations)`: Performs gradient descent to optimize the weight and bias parameters.

### Training

The model is trained using gradient descent with a specified learning rate and number of iterations. The cost history is recorded to visualize the convergence of the algorithm.

### Visualization

The results are visualized using Matplotlib:
- Training data and the regression line.
- Cost history over iterations.

## Usage

To run the implementation, you can either use the Jupyter notebook `grad.ipynb` or the Python script `grad.py`.

### Using Jupyter Notebook

1. Open `grad.ipynb` in Jupyter Notebook.
2. Run the cells sequentially to see the step-by-step implementation and results.

### Using Python Script

1. Run the `grad.py` script:
    ```bash
    python grad.py
    ```
2. The script will output the final parameters and display the plots.

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- Jupyter Notebook (optional)

## License

This project is licensed under the MIT License.

## Acknowledgments

This implementation is inspired by the fundamental concepts of machine learning and gradient descent algorithms.
