import numpy as np


def mse(y, preds):
    return np.mean(np.square(y - preds))


def gradient_descent(X, y, learning_rate=0.001):
    w = np.random.uniform(-1, 1, X.shape[1])
    i = 0
    while True:
        new_w = w - learning_rate * 2 * (X @ w - y) @ X
        if np.linalg.norm(new_w - w) < 1e-3:
            break
        w = new_w
        if i % 10 == 0:
            print(i, 'Step', '-', mse(y, X @ w))
        i += 1
    return w


def stochastic_gradient_descent(X, y, learning_rate=0.001, max_epoch=10):
    w = np.random.uniform(-1,1, X.shape[1])
    for epoch_count in range(1, max_epoch + 1):
        idx = np.arange(X.shape[0])
        for i in idx:
            new_w = w - learning_rate * 2 * (X[i, np.newaxis] @ w - y[i]) @ X[i, np.newaxis]
            w = new_w
        print(epoch_count, 'Epoch', '-', mse(y, X @ w))
    return w


def mini_batch_gradient_descent(X, y, learning_rate=0.001, batch_size=100, max_epoch=10):
    w = np.random.uniform(-1, 1, X.shape[1])
    for epoch_count in range(1, max_epoch + 1):
        idx = np.arange(X.shape[0])
        np.random.shuffle(idx)
        for i in range(0, X.shape[0], batch_size):
            if i + batch_size > X.shape[0]:
                batch_idx = idx[i:]
            else:
                batch_idx = idx[i:i+batch_size]
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]
            new_w = w - learning_rate * 2 * (X_batch @ w - y_batch) @ X_batch
            if np.linalg.norm(new_w - w) < 1e-8:
                break
            w = new_w
        print(epoch_count, 'Epoch', '-', mse(y, X @ w))
    return w
