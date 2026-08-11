import numpy as np
import pandas as pd

data = pd.read_csv("data/train.csv").values

Y_train = data[:, 0]
X_train = data[:, 1:].T / 255

def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.zeros((10, 1))
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.zeros((10, 1))

    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(0, Z)

def deriv_ReLU(Z):
    return (Z > 0)*1

def softmax(Z):
    return (np.exp(Z))/np.sum(np.exp(Z), axis=0)

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1 @ X + b1

    A1 = ReLU(Z1)

    Z2 = W2 @ A1 + b2

    A2 = softmax(Z2)

    return Z1, A1, Z2, A2

def one_hot(Y):
    one_hot_Y = np.zeros((10, Y.size))

    one_hot_Y[Y, np.arange(Y.size)] = 1

    return one_hot_Y

def backward_prop(Z1, A1, A2, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)

    # Градиенты для второго слоя
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * (dZ2 @ A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)

    # Градиенты для первого слоя
    dZ1 = (W2.T @ dZ2) * deriv_ReLU(Z1)
    dW1 = 1 / m * (dZ1 @ X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2

def update(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - (alpha * dW1)
    b1 = b1 - (alpha * db1)

    W2 = W2 - (alpha * dW2)
    b2 = b2 - (alpha * db2)

    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, axis=0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()

    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, A2, W2, X, Y)
        W1, b1, W2, b2 = update(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            predictions = get_predictions(A2)
            acc = get_accuracy(predictions, Y)
            print(f"Итерация: {i} | Точность: {acc:.4f}")

    return W1, b1, W2, b2

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)
