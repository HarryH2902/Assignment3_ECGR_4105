import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
df = pd.read_csv("cancer.csv")

# Drop unnecessary columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.drop(columns=["id"], errors="ignore")

# Encode labels: M = 1, B = 0
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Features and target
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# Fill missing values
X = X.fillna(X.mean())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Try PCA with K = 1 to 30
max_k = X.shape[1]
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []

for k in range(1, max_k + 1):
    pca = PCA(n_components=k)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    model = GaussianNB()
    model.fit(X_train_pca, y_train)
    y_pred = model.predict(X_test_pca)

    accuracy_list.append(accuracy_score(y_test, y_pred))
    precision_list.append(precision_score(y_test, y_pred))
    recall_list.append(recall_score(y_test, y_pred))
    f1_list.append(f1_score(y_test, y_pred))

# Plot performance metrics vs K
plt.figure(figsize=(12, 8))
plt.plot(range(1, max_k + 1), accuracy_list, label="Accuracy")
plt.plot(range(1, max_k + 1), precision_list, label="Precision")
plt.plot(range(1, max_k + 1), recall_list, label="Recall")
plt.plot(range(1, max_k + 1), f1_list, label="F1 Score")
plt.xlabel("Number of Principal Components (K)")
plt.ylabel("Score")
plt.title("Naive Bayes with PCA - Cancer Dataset")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Best result
best_k = np.argmax(accuracy_list) + 1
print(f"\nBest K: {best_k}")
print(f"Accuracy:  {accuracy_list[best_k - 1]:.4f}")
print(f"Precision: {precision_list[best_k - 1]:.4f}")
print(f"Recall:    {recall_list[best_k - 1]:.4f}")
print(f"F1 Score:  {f1_list[best_k - 1]:.4f}")
