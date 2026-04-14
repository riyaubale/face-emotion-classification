"""
Kernel Methods: Classification & Regression

Includes:
- Face emotion classification (Gaussian kernel)
- Cross-validation for sigma tuning
- Synthetic nonlinear classification demo
- Kernel ridge regression (function approximation)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


# =========================================
# PART 1 — FACE EMOTION KERNEL CLASSIFIER
# =========================================

def kernel_classifier(X, y, sigma, lam=0.5):
    n = X.shape[0]
    distsq = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            d = np.linalg.norm(X[i] - X[j])
            distsq[i, j] = d**2

    K = np.exp(-distsq / (2 * sigma**2))
    alpha = np.linalg.inv(K + lam * np.identity(n)) @ y
    return alpha


def predict_kernel(X_train, X_test, alpha, sigma):
    y_pred = np.zeros((X_test.shape[0], 1))
    for i in range(X_test.shape[0]):
        distances = np.linalg.norm(X_train - X_test[i], axis=1)
        K = np.exp(-distances**2 / (2 * sigma**2))
        y_pred[i] = K @ alpha
    return y_pred


def accuracy(y_true, y_pred):
    return np.mean(y_true == np.sign(y_pred))


def run_face_classification():
    dataset = loadmat('face_emotion_data.mat')
    X, y = dataset['X'], dataset['y']
    n = X.shape[0]

    X = np.hstack((np.ones((n,1)), X))

    sigma_values = np.logspace(-2, 1, 30)
    accuracies = []

    for sigma in sigma_values:
        alpha = kernel_classifier(X, y, sigma)
        y_pred = predict_kernel(X, X, alpha, sigma)
        acc = accuracy(y, y_pred)
        accuracies.append(acc)

    plt.plot(sigma_values, accuracies)
    plt.xlabel("sigma")
    plt.ylabel("accuracy")
    plt.title("Kernel Classification Accuracy vs Sigma")
    plt.show()


# =========================================
# PART 2 — CROSS VALIDATION
# =========================================

def cross_validation(X, y, sigma, k=8):
    indices = np.arange(len(y))
    np.random.shuffle(indices)

    fold_sizes = np.full(k, len(y)//k)
    fold_sizes[:len(y)%k] += 1

    current = 0
    accs = []

    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_idx = indices[start:stop]
        train_idx = np.concatenate((indices[:start], indices[stop:]))

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        alpha = kernel_classifier(X_train, y_train, sigma)
        y_pred = predict_kernel(X_train, X_test, alpha, sigma)

        accs.append(accuracy(y_test, y_pred))
        current = stop

    return np.mean(accs)


def tune_sigma():
    dataset = loadmat('face_emotion_data.mat')
    X, y = dataset['X'], dataset['y']
    n = X.shape[0]
    X = np.hstack((np.ones((n,1)), X))

    sigma_values = np.logspace(-6, 2, 30)
    results = []

    for sigma in sigma_values:
        acc = cross_validation(X, y, sigma)
        results.append(acc)

    best_sigma = sigma_values[np.argmax(results)]

    plt.plot(sigma_values, results)
    plt.xlabel("sigma")
    plt.ylabel("cross-val accuracy")
    plt.title("Sigma Tuning (Cross Validation)")
    plt.show()

    print("Best sigma:", best_sigma)


# =========================================
# PART 3 — SYNTHETIC CLASSIFICATION DEMO
# =========================================

def synthetic_demo():
    n = 500
    X = np.random.rand(n,2) - 0.5

    Y = np.sign(np.sum(X**2, axis=1) - 0.1).reshape(-1,1)

    sigma = 0.5
    lam = 0.01

    alpha = kernel_classifier(X, Y, sigma, lam)

    g = 100
    x1 = np.linspace(-.5, .5, g)
    x2 = np.linspace(-.5, .5, g)
    grid = np.zeros((g,g))

    for i, a in enumerate(x1):
        for j, b in enumerate(x2):
            grid[i,j] = np.exp(-np.linalg.norm(X - np.array([a,b]), axis=1)**2/(2*sigma**2)) @ alpha

    plt.contour(x1, x2, np.sign(grid))
    plt.title("Decision Boundary (Kernel)")
    plt.show()


# =========================================
# PART 4 — KERNEL RIDGE REGRESSION
# =========================================

def kernel_ridge_regression(x, d, sigma=0.04, lam=0.01):
    n = x.shape[0]
    distsq = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            distsq[i,j] = (x[i]-x[j])**2

    K = np.exp(-distsq/(2*sigma**2))
    alpha = np.linalg.inv(K + lam*np.identity(n)) @ d
    return alpha


def regression_demo():
    np.random.seed(1024)

    n = 50
    x = np.random.rand(n,1)
    d = x*x + 0.4*np.sin(1.5*np.pi*x) + 0.04*np.random.randn(n,1)

    alpha = kernel_ridge_regression(x, d)

    x_test = np.linspace(0,1,100)
    dtest = np.zeros((100,1))

    for i in range(100):
        dtest[i] = np.exp(-(x_test[i]-x.flatten())**2/(2*0.04**2)) @ alpha

    plt.scatter(x, d, label="data")
    plt.plot(x_test, dtest, label="fit")
    plt.legend()
    plt.title("Kernel Ridge Regression")
    plt.show()


# =========================================
# MAIN
# =========================================

if __name__ == "__main__":
    run_face_classification()
    tune_sigma()
    synthetic_demo()
    regression_demo()