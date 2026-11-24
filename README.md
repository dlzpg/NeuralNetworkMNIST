# NeuralNetworkMNIST

A fully-connected neural network built **from scratch with NumPy** (no TensorFlow, no PyTorch) that classifies handwritten digits from the MNIST dataset.

Forward pass, backpropagation, gradient descent, evaluation and weight saving/loading are all implemented by hand, which makes this project way to understand what deep learning frameworks do under the hood.

## Results

Tweak as u like in order to get other results.

| Dataset | Samples | Accuracy |
|---|---|---|
| Training (after 2000 iterations) | 56,000 | **92.65%** |
| Test set | 14,000 | **92.21%** |
| Full dataset (train + test) | 70,000 | **92.51%** |

Training and test accuracy are very close, so the model is not overfitting.

## How it works

### Architecture

A simple two-layer network:

```
Input (784 pixels) -> Hidden layer -> Output layer (10 classes, digits 0-9)
```

Each 28x28 image is flattened into a vector of 784 values. The parameters are:

- `W1`, `b1`: weights and biases of the hidden layer
- `W2`, `b2`: weights and biases of the output layer

The output layer gives one probability per digit, and the highest one is the prediction.

### Main functions

| Function | Description |
|---|---|
| `init_params()` | Initializes `W1, b1, W2, b2` |
| `forward_pass(...)` | Computes `Z1, A1, Z2, A2` (the network's output) |
| `backpropogation(...)` | Computes the gradients `dW1, db1, dW2, db2` |
| `update_params(...)` | Updates weights and biases using the gradients and learning rate |
| `get_loss(...)` / `get_accuracy(...)` | Metrics tracked during training |
| `gradient_descent(X, Y, alpha, iterations)` | Main training loop |
| `evaluate_test(...)` | Overall and per-digit accuracy |
| `predict_one(...)` | Predicts a single image and displays it |
| `save_weights(...)` / `load_weights()` | Save and load trained parameters as `.npz` files |

## Training

Training is done with plain **gradient descent**. On every iteration the loop:

1. Runs a **forward pass** to get predictions.
2. Runs **backpropagation** to compute the gradients.
3. **Updates** weights and biases: `param = param - alpha * gradient`.
4. Every 100 iterations, prints the loss and accuracy.

```python
W1, b1, W2, b2, losses = gradient_descent(x_train, y_train, 0.25, 2000)
```

Here `0.25` is the learning rate (`alpha`) and `2000` is the number of iterations.

![Training output](images/training.png)

The loss falls steadily from **3.5386 to 0.2560**, while accuracy rises from **9.04%** (random guessing) to **92.65%**. Most of the learning happens in the first 500 iterations, and after that the improvements are smaller.

## Evaluation

`evaluate_test` reports the overall accuracy and a per-digit breakdown:

```python
evaluate_test(x_test, y_test, W1, b1, W2, b2)
```

```
Accuracy: 92.21% (12910/14000 correct)
  Digit 0: 96.07% (1345/1400)
  Digit 1: 96.77% (1559/1611)
  ...
```

### Predicting a single image

`predict_one` picks an image by index, shows the prediction, its confidence and the true label, and plots the digit:

```python
predict_one(4567, X.T, y, W1, b1, W2, b2, True)
```

```
Index:      4567 / 70000
Prediction: 8 (confidence: 97.99%)
Actual:     8
Result:     CORRECT
```

![Prediction results](images/results.png)

## Saving and loading weights

Once trained, you don't need to train again. The parameters can be stored in a compressed NumPy `.npz` file and loaded later.

```python
save_weights(W1, b1, W2, b2)
# Enter a name for the weights file: 92,2acc
# Saved to .../MNIST_NeuralNetwork/92,2acc.npz

W1, b1, W2, b2 = load_weights()
# Available weight files:
#   1. 90,4acc.npz
#   2. 92,2acc.npz
# Enter the name of the weights file to load: 92,2acc
```

`save_weights` asks for a file name and saves the four arrays. `load_weights` lists the available files and loads the one you choose.

![Saving and loading weights](images/weights.png)

## Getting started

### Requirements

- Python 3.8+
- NumPy
- Matplotlib (to display digits)
- Jupyter Notebook

```bash
pip install numpy matplotlib notebook
```

### Run

```bash
git clone https://github.com/dlzpg/NeuralNetworkMNIST.git
cd NeuralNetworkMNIST
jupyter notebook
```

Open the notebook and run the cells in order, or load a saved weights file to skip training.
