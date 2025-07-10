import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# Load dataset
df = pd.read_csv("cancer.csv")

# Drop unnecessary columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.drop(columns=["id"], errors="ignore")

# Convert labels: M = 1, B = 0
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Separate features and target
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# Handle missing values with mean imputation
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression WITHOUT penalty
model = LogisticRegression(max_iter=1000, penalty=None, solver='lbfgs')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics (no penalty)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Results without Regularization:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")

# Confusion Matrix (no penalty)
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.title("Confusion Matrix - No Penalty")
plt.show()

# Logistic Regression WITH L2 Regularization
model_l2 = LogisticRegression(max_iter=1000, penalty='l2', C=1.0, solver='lbfgs')
model_l2.fit(X_train, y_train)
y_pred_l2 = model_l2.predict(X_test)

# Metrics (L2)
acc2 = accuracy_score(y_test, y_pred_l2)
prec2 = precision_score(y_test, y_pred_l2)
rec2 = recall_score(y_test, y_pred_l2)
f1_2 = f1_score(y_test, y_pred_l2)

print("\nResults with L2 Regularization:")
print(f"Accuracy:  {acc2:.4f}")
print(f"Precision: {prec2:.4f}")
print(f"Recall:    {rec2:.4f}")
print(f"F1 Score:  {f1_2:.4f}")

# Confusion Matrix (L2)
cm2 = confusion_matrix(y_test, y_pred_l2)
ConfusionMatrixDisplay(cm2).plot()
plt.title("Confusion Matrix - L2 Penalty")
plt.show()
