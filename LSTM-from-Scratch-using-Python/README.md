# LSTM from Scratch using Python

This repository contains implementations of Long Short-Term Memory (LSTM) networks from scratch, both using pure Python/NumPy and PyTorch.

## Implementations

1. `lstm.py` - Pure Python/NumPy implementation of LSTM
   - Includes basic LSTM cell with forget gate, input gate, cell state, and output gate
   - Forward propagation implementation
   - Example usage code

2. `lstm-pytorch.py` - PyTorch implementation of LSTM
   - Modern implementation using PyTorch's neural network modules
   - Training capabilities with backpropagation
   - Example usage with synthetic data
   - Visualization of predictions

## Requirements

- Python 3.x
- NumPy (for lstm.py)
- PyTorch (for lstm-pytorch.py)
- Matplotlib (for visualization)

## Installation

```bash
pip install numpy torch matplotlib
```

## Usage

1. Run `python lstm.py` to use the pure Python/NumPy implementation of LSTM
2. Run `python lstm-pytorch.py` to use the PyTorch implementation of LSTM

## License

MIT License

## References

- [Long Short-Term Memory](https://en.wikipedia.org/wiki/Long_short-term_memory)
- [PyTorch LSTM example](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html)