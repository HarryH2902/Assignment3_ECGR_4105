
# PROBLEM 1: Univariate Linear Regression with Gradient Descent
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("HW1.csv")
X = df[['X1', 'X2', 'X3']].values
Y = df['Y'].values.reshape(-1, 1)

# Gradient Descent for Univariate Case
def gradient_descent_single(X_col, Y, learning_rate=0.05, iterations=1000):
    m = len(Y)
    theta0 = 0
    theta1 = 0
    loss_history = []

    for _ in range(iterations):
        predictions = theta0 + theta1 * X_col
        error = predictions.reshape(-1, 1) - Y
        cost = (1 / (2 * m)) * np.sum(error ** 2)
        loss_history.append(cost)

        d_theta0 = (1 / m) * np.sum(error)
        d_theta1 = (1 / m) * np.sum(error.flatten() * X_col)

        theta0 -= learning_rate * d_theta0
        theta1 -= learning_rate * d_theta1

    return theta0, theta1, loss_history

# Run for each variable
results = {}
for i, label in enumerate(['X1', 'X2', 'X3']):
    X_col = X[:, i]
    theta0, theta1, loss_history = gradient_descent_single(X_col, Y)
    results[label] = {
        'theta0': theta0,
        'theta1': theta1,
        'loss_history': loss_history,
        'final_loss': loss_history[-1],
        'X_col': X_col
    }

# Plot regression and loss
for label, res in results.items():
    plt.figure()
    plt.plot(res['X_col'], res['theta0'] + res['theta1'] * res['X_col'], label='Regression Line')
    plt.scatter(res['X_col'], Y, alpha=0.6, label='Data')
    plt.title(f'Regression for {label}')
    plt.xlabel(label)
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(res['loss_history'])
    plt.title(f'Loss vs Iterations for {label}')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
