"""
Linear Classifier using Least Squares

- Trains classifier on face emotion dataset
- Compares all 9 features vs top 3 features
- Performs 8-fold cross validation
"""

import numpy as np
from scipy.io import loadmat


# ----------------------------
# Load Data
# ----------------------------
data = loadmat('face_emotion_data.mat')
X = data['X']
y = data['y']


# ----------------------------
# Train Full Model (9 features)
# ----------------------------
w_full = np.linalg.pinv(X) @ y

# Feature importance
feature_weights = np.abs(w_full.flatten())
sorted_weights = np.sort(feature_weights)

print("Sorted feature magnitudes:", sorted_weights)

# Top 3 features
top3_values = sorted_weights[-3:]
top3_indices = np.where(np.isin(feature_weights, top3_values))[0]

print("Top 3 feature indices:", top3_indices)


# ----------------------------
# Train Reduced Model (3 features)
# ----------------------------
X_reduced = X[:, top3_indices]
w_reduced = np.linalg.pinv(X_reduced) @ y


# ----------------------------
# Evaluate Training Error
# ----------------------------
pred_full = np.sign(X @ w_full)
error_full = np.mean(pred_full != y) * 100

print("\nIncorrectly classified using all 9 features:", error_full, "%")


pred_reduced = np.sign(X_reduced @ w_reduced)
error_reduced = np.mean(pred_reduced != y) * 100

print("Incorrectly classified using top 3 features:", error_reduced, "%")


# ----------------------------
# 8-Fold Cross Validation (All Features)
# ----------------------------
subsets_X = np.split(X, 8)
subsets_y = np.split(y, 8)

errors_full_cv = []

for i in range(8):
    X_test = subsets_X[i]
    y_test = subsets_y[i]

    X_train = np.vstack(subsets_X[:i] + subsets_X[i+1:])
    y_train = np.vstack(subsets_y[:i] + subsets_y[i+1:])

    w = np.linalg.pinv(X_train) @ y_train

    preds = np.sign(X_test @ w)
    error = np.mean(preds != y_test) * 100
    errors_full_cv.append(error)

    print(f"Fold {i+1} Error (9 features): {error:.2f} %")

print("Average CV Error (9 features):", np.mean(errors_full_cv), "%")


# ----------------------------
# 8-Fold Cross Validation (Top 3 Features)
# ----------------------------
subsets_X_reduced = np.split(X_reduced, 8)
errors_reduced_cv = []

for i in range(8):
    X_test = subsets_X_reduced[i]
    y_test = subsets_y[i]

    X_train = np.vstack(subsets_X_reduced[:i] + subsets_X_reduced[i+1:])
    y_train = np.vstack(subsets_y[:i] + subsets_y[i+1:])

    w = np.linalg.pinv(X_train) @ y_train

    preds = np.sign(X_test @ w)
    error = np.mean(preds != y_test) * 100
    errors_reduced_cv.append(error)

    print(f"Fold {i+1} Error (3 features): {error:.2f} %")

print("Average CV Error (3 features):", np.mean(errors_reduced_cv), "%")


# ----------------------------
# Final Summary
# ----------------------------
print("\n===== FINAL SUMMARY =====")
print("Training Error (9 features):", error_full, "%")
print("Training Error (3 features):", error_reduced, "%")
print("CV Error (9 features):", np.mean(errors_full_cv), "%")
print("CV Error (3 features):", np.mean(errors_reduced_cv), "%")