import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Update file path as needed: if the file is in the same folder, use "cancer.csv"
df = pd.read_csv("cancer.csv")

# Clean and preprocess the data
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.drop(columns=["id"], errors="ignore")
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Separate features and target
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# Fill missing values
X = X.fillna(X.mean())

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Try different numbers of principal components (K)
max_k = X.shape[1]
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []

for k in range(1, max_k + 1):
    pca = PCA(n_components=k)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_pca, y_train)
    y_pred = model.predict(X_test_pca)

    accuracy_list.append(accuracy_score(y_test, y_pred))
    precision_list.append(precision_score(y_test, y_pred))
    recall_list.append(recall_score(y_test, y_pred))
    f1_list.append(f1_score(y_test, y_pred))

# Plot the performance metrics vs. number of PCA components
plt.figure(figsize=(12, 8))
components = range(1, max_k + 1)
plt.plot(components, accuracy_list, label="Accuracy", marker='o')
plt.plot(components, precision_list, label="Precision", marker='o')
plt.plot(components, recall_list, label="Recall", marker='o')
plt.plot(components, f1_list, label="F1 Score", marker='o')
plt.xlabel("Number of Principal Components (K)")
plt.ylabel("Score")
plt.title("Performance Metrics vs. PCA Components (Logistic Regression)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Identify the best number of components based on accuracy
best_k = np.argmax(accuracy_list) + 1
best_results = {
    "Best K": best_k,
    "Accuracy": accuracy_list[best_k - 1],
    "Precision": precision_list[best_k - 1],
    "Recall": recall_list[best_k - 1],
    "F1 Score": f1_list[best_k - 1]
}
print("Best PCA Components Performance:")
for key, value in best_results.items():
    print(f"{key}: {value}")

# Optionally, you can display the results in a DataFrame
results_df = pd.DataFrame([best_results])
print(results_df)
