"""
Hospital Readmission Prediction — Gradient Boosting Classification

Predicts whether a hospital patient will be readmitted within 30 days
of discharge. The pipeline preprocesses a mixed-type medical dataset,
splits it into train / validation / test sets, trains a baseline
Gradient Boosting Classifier, then runs GridSearchCV to tune
hyperparameters with recall as the optimization metric.

Author: Amir Vaziri
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join("data", "medical_clean.csv")
OUTPUT_DIR = "output"
RANDOM_SEED = 42
TEMP_SPLIT_SIZE = 0.30   # fraction held out from the initial split (val + test)
VAL_RATIO = 0.50         # fraction of the temp split that becomes validation

TARGET_COL = "ReAdmis"

DROP_COLS = [
    "CaseOrder", "Customer_id", "Interaction", "UID",
    "City", "State", "County", "Zip", "Lat", "Lng",
    "Population", "TimeZone", "Job",
]

BINARY_COLS = [
    "HighBlood", "Stroke", "Overweight", "Arthritis", "Diabetes",
    "Hyperlipidemia", "BackPain", "Anxiety", "Allergic_rhinitis",
    "Reflux_esophagitis", "Asthma", "Soft_drink",
]

CATEGORICAL_COLS = [
    "Gender", "Marital", "Area", "Initial_admin",
    "Complication_risk", "Services",
]

PARAM_GRID = {
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 5],
    "min_samples_split": [2, 5],
}


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Remove administrative columns, encode the binary target and condition
    flags as integers, and one-hot encode nominal categorical features."""
    df = df.drop(columns=DROP_COLS)
    df = df.dropna()

    # Encode target: Yes → 1, No → 0
    df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})

    # Encode binary medical condition columns: Yes → 1, No → 0
    for col in BINARY_COLS:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    # One-hot encode nominal categorical columns; drop first level to avoid
    # perfect multicollinearity among dummy variables
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)

    # Cast any boolean columns produced by get_dummies to integer
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def print_metrics(label: str, y_true, y_pred, y_prob) -> None:
    """Print a labelled table of classification metrics for one evaluation."""
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")
    print(f"  Accuracy  : {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision : {precision_score(y_true, y_pred):.4f}")
    print(f"  Recall    : {recall_score(y_true, y_pred):.4f}")
    print(f"  F1 Score  : {f1_score(y_true, y_pred):.4f}")
    print(f"  AUC-ROC   : {roc_auc_score(y_true, y_prob):.4f}")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"  Confusion : TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"{'─' * 60}")


def plot_confusion_matrix(
    y_true,
    y_pred,
    title: str,
    filename: str,
    output_dir: str = OUTPUT_DIR,
) -> None:
    """Plot, style, and save a confusion matrix as a PNG to output_dir."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["No", "Yes"]
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    plt.tight_layout()

    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full classification pipeline.

    Steps:
        1. Load and preprocess the raw dataset.
        2. Split into train (70 %), validation (15 %), and test (15 %) sets.
        3. Train a baseline Gradient Boosting model and evaluate on validation.
        4. Tune hyperparameters with GridSearchCV (5-fold, scoring=recall).
        5. Evaluate the best model on the held-out test set.
    """
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 14

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load and preprocess
    # ------------------------------------------------------------------
    print(f"\nLoading: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"  Raw    — rows: {len(df):,}  columns: {df.shape[1]}")

    df_clean = preprocess(df)
    print(
        f"  Clean  — rows: {len(df_clean):,}  "
        f"features: {df_clean.shape[1] - 1}"
    )

    X = df_clean.drop(TARGET_COL, axis=1)
    y = df_clean[TARGET_COL]

    # ------------------------------------------------------------------
    # 2. Three-way split: 70 % train / 15 % validation / 15 % test
    # ------------------------------------------------------------------
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=TEMP_SPLIT_SIZE, random_state=RANDOM_SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=VAL_RATIO, random_state=RANDOM_SEED
    )
    print(
        f"\nSplit — train: {len(X_train):,}  "
        f"val: {len(X_val):,}  "
        f"test: {len(X_test):,}"
    )

    # Persist splits for reproducibility
    for subset, X_s, y_s in [
        ("train", X_train, y_train),
        ("val",   X_val,   y_val),
        ("test",  X_test,  y_test),
    ]:
        X_s.assign(**{TARGET_COL: y_s}).to_csv(
            os.path.join(OUTPUT_DIR, f"{subset}_split.csv"), index=False
        )

    # ------------------------------------------------------------------
    # 3. Baseline model — default Gradient Boosting hyperparameters
    # ------------------------------------------------------------------
    print("\nTraining baseline model (default hyperparameters)...")
    baseline = GradientBoostingClassifier(random_state=RANDOM_SEED)
    baseline.fit(X_train, y_train)

    y_val_pred = baseline.predict(X_val)
    y_val_prob = baseline.predict_proba(X_val)[:, 1]

    print_metrics(
        "Baseline Model — Validation Set",
        y_val, y_val_pred, y_val_prob,
    )
    plot_confusion_matrix(
        y_val, y_val_pred,
        title="Baseline Model — Validation Set",
        filename="ValidationConfusionMat.png",
    )

    # ------------------------------------------------------------------
    # 4. Hyperparameter tuning via GridSearchCV
    #    Recall is the scoring metric: in healthcare, false negatives
    #    (missed readmissions) carry a higher cost than false positives.
    # ------------------------------------------------------------------
    print("\nRunning GridSearchCV (5-fold, scoring=recall)...")
    grid = GridSearchCV(
        GradientBoostingClassifier(random_state=RANDOM_SEED),
        PARAM_GRID,
        cv=5,
        scoring="recall",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)

    print(f"\nBest parameters : {grid.best_params_}")
    print(f"Best CV Recall   : {grid.best_score_:.4f}")

    # ------------------------------------------------------------------
    # 5. Evaluate the tuned model on the held-out test set
    # ------------------------------------------------------------------
    best_model = grid.best_estimator_

    y_test_pred = best_model.predict(X_test)
    y_test_prob = best_model.predict_proba(X_test)[:, 1]

    print_metrics(
        "Optimized Model — Test Set",
        y_test, y_test_pred, y_test_prob,
    )
    plot_confusion_matrix(
        y_test, y_test_pred,
        title="Optimized Model — Test Set",
        filename="TestConfusionMat.png",
    )

    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
