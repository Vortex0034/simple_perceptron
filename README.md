# Simple Multilayer Perceptron

A basic implementation of a multilayer perceptron in Python for predicting logical function outputs.

## About

This is an educational project developed for a "Machine Learning" course. Implements a simple MLP to classify data based on the logical function:

`((A2 ∨ C2 ∨ G2) ∧ (A2 ∨ B2 ∨ H2) ∧ (F2 ∨ D2 ∨ C2 ∨ ¬E2) ∧ (K2 ∨ J2 ∨ I2))`

## Features

- Configurable number of layers
- Sigmoid activation on all layers
- Binary classification (0 or 1)
- Training visualization with matplotlib
- Optimized with NumPy for performance

## Dataset

`data.csv` contains:
- Columns 0-10: input variables
- Column 11: output value of the logical function

## Quick Start

```python
from Neuron import Perceptron

# Create model
model = Perceptron(eta=0.001, layers=[11,7,7,2], iters=5000)

# Train
model.fit(X_train, y_train) # y_train with One-Hot Encoding! You can use pd.get_dummies().

# Predict
predictions = model.predict(X_test)
```

## Project Structure

```
├── Neuron.py        # Main MLP implementation
├── example.py        # Usage example
├── data.csv         # Dataset
├── requirements.txt # Dependencies
└── README.md
```

## Requirements

- Python v3.13
- Numpy v2.3.4
- Matplotlib v3.10.7
- Pandas v2.3.3