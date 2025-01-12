import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Initialize parameters
def initialize_parameters():
    weight = np.random.randn(1)
    bias = np.random.randn(1)
    return weight, bias

# Prediction function
def predict(X, weight, bias):
    return X * weight + bias

# Cost function (MSE)
def compute_cost(X, y, weight, bias):
    m = len(X)
    predictions = predict(X, weight, bias)
    cost = (1/(2*m)) * np.sum((predictions - y) ** 2)
    return cost

# Gradient descent function
def gradient_descent(X, y, weight, bias, learning_rate, iterations):
    m = len(X)
    cost_history = []
    
    for i in range(iterations):
        predictions = predict(X, weight, bias)
        
        # Calculate gradients
        dw = (1/m) * np.sum(X * (predictions - y))
        db = (1/m) * np.sum(predictions - y)
        
        # Update parameters
        weight = weight - learning_rate * dw
        bias = bias - learning_rate * db
        
        # Store cost
        cost = compute_cost(X, y, weight, bias)
        cost_history.append(cost)
        
    return weight, bias, cost_history

# Train the model
learning_rate = 0.01
iterations = 1000
weight, bias = initialize_parameters()
weight, bias, cost_history = gradient_descent(X, y, weight, bias, learning_rate, iterations)

# Plot results
plt.figure(figsize=(10, 5))

# Plot training data and regression line
plt.subplot(1, 2, 1)
plt.scatter(X, y, color='blue', label='Training Data')
plt.plot(X, predict(X, weight, bias), color='red', label='Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

# Plot cost history
plt.subplot(1, 2, 2)
plt.plot(cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost History')

plt.tight_layout()
plt.show()

print(f"Final parameters: weight = {weight[0]:.2f}, bias = {bias[0]:.2f}")