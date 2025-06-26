
# PROBLEM 2: Multivariate Linear Regression with Gradient Descent
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("HW1.csv")
X = df[['X1', 'X2', 'X3']].values
Y = df['Y'].values.reshape(-1, 1)

def gradient_descent_multi(X, Y, learning_rate=0.05, iterations=1000):
    m, n = X.shape
    X_b = np.c_[np.ones((m, 1)), X]  # Add bias term
    theta = np.zeros((n + 1, 1))
    loss_history = []

    for _ in range(iterations):
        predictions = X_b.dot(theta)
        error = predictions - Y
        cost = (1 / (2 * m)) * np.sum(error ** 2)
        loss_history.append(cost)

        gradients = (1 / m) * X_b.T.dot(error)
        theta -= learning_rate * gradients

    return theta, loss_history

# Train multivariate model
theta_multi, loss_history_multi = gradient_descent_multi(X, Y)

# Predictions
def predict(X_input, theta):
    X_input_b = np.insert(X_input, 0, 1)
    return X_input_b @ theta

predictions = {
    "(1,1,1)": predict(np.array([1, 1, 1]), theta_multi),
    "(2,0,4)": predict(np.array([2, 0, 4]), theta_multi),
    "(3,2,1)": predict(np.array([3, 2, 1]), theta_multi),
}

# Plot loss curve
plt.figure()
plt.plot(loss_history_multi)
plt.title("Multivariate Regression: Loss vs Iterations")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# Show results
theta_df = pd.DataFrame({
    'Parameter': ['Theta 0 (bias)', 'Theta 1 (X1)', 'Theta 2 (X2)', 'Theta 3 (X3)'],
    'Value': theta_multi.flatten()
})
print(theta_df)

predictions_df = pd.DataFrame({
    'Input': list(predictions.keys()),
    'Predicted Y': [float(predictions[key]) for key in predictions]
})
print(predictions_df)
