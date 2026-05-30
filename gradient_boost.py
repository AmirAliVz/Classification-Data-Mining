from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 14

# Load Data
df = pd.read_csv("data/medical_clean_processed.csv")

# Define Features and Target
X = df.drop('ReAdmis', axis=1)
y = df['ReAdmis']

# Part E1. Train / Validation / Test Split
# =========================================
# First split: train (70%) and temp (30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Second split: validation (15%) and test (15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# Combine features + target for saving purposes
train_df = X_train.copy()
train_df['ReAdmis'] = y_train

val_df = X_val.copy()
val_df['ReAdmis'] = y_val

test_df = X_test.copy()
test_df['ReAdmis'] = y_test

# Save clean datasets
train_df.to_csv("data/train_data.csv", index=False)
val_df.to_csv("data/validation_data.csv", index=False)
test_df.to_csv("data/test_data.csv", index=False)

# Part E2. Create an initial model
# =================================
# Train Initial Gradient Boost Model
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

# Make Predictions (on validation set)
y_pred = model.predict(X_val)
y_prob = model.predict_proba(X_val)[:, 1]

# Metrics Calculation
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

print(" ======== Initial Model Metrics: ")
print("Accuracy:", accuracy_score(y_val, y_pred))
print("Precision:", precision_score(y_val, y_pred))
print("Recall:", recall_score(y_val, y_pred))
print("F1 Score:", f1_score(y_val, y_pred))
print("AUC-ROC:", roc_auc_score(y_val, y_prob))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
print(50*"--")

# Visualization
disp = ConfusionMatrixDisplay.from_predictions(y_val, y_pred, cmap=plt.cm.Blues)
disp.ax_.set_title('Confusion Matrix on Validation Data')

for text in disp.text_.ravel():
    text.set_fontsize(25)   # make numbers bigger

disp.ax_.set_xlabel('Predicted label', fontsize=16)
disp.ax_.set_ylabel('True label', fontsize=16)
disp.ax_.set_title('Confusion Matrix on Validation Data', fontsize=18)

disp.ax_.tick_params(axis='both', labelsize=14)  # x/y tick labels

plt.show()

# Part E3. Perform hyperparameter tuning
# ========================================
# Grid Search with Cross Validation
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5],
    'min_samples_split': [2, 5]
}

grid = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='recall',
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("Best Parameters:")
for k, v in grid.best_params_.items():
    print(f"{k}: {v}")

# E4. Final Model (Test Set Evaluation)
# ======================================
# Train Optimized Model
best_model = grid.best_estimator_

# Predict on Test Data
y_test_pred = best_model.predict(X_test)
y_test_prob = best_model.predict_proba(X_test)[:, 1]

# Final Metrics
print(" ======== Optimized Model Metrics: ")
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print("Precision:", precision_score(y_test, y_test_pred))
print("Recall:", recall_score(y_test, y_test_pred))
print("F1 Score:", f1_score(y_test, y_test_pred))
print("AUC-ROC:", roc_auc_score(y_test, y_test_prob))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
print(50*"--")

# Visualization
disp = ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred, cmap=plt.cm.Blues)
disp.ax_.set_title('Confusion Matrix on Test Data')

for text in disp.text_.ravel():
    text.set_fontsize(25)   # make numbers bigger

disp.ax_.set_xlabel('Predicted label', fontsize=16)
disp.ax_.set_ylabel('True label', fontsize=16)
disp.ax_.set_title('Confusion Matrix on Test Data', fontsize=18)

disp.ax_.tick_params(axis='both', labelsize=14)  # x/y tick labels

plt.show()
