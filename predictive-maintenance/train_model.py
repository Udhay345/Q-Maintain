"""Train XGBoost on final_phone_sensor.csv and save model artifacts."""
import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from xgboost import XGBClassifier

from scipy.spatial.transform import Rotation as R
from utils.preprocessing import clean_columns, REQUIRED_FEATURES

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_model.pkl")
PREPROCESSING_PATH = os.path.join(MODEL_DIR, "preprocessing.pkl")

CANDIDATE_CSVS = [
    os.path.join(ROOT, "..", "predictive", "final_phone_sensor.csv"),
    os.path.join(ROOT, "final_phone_sensor.csv"),
    os.path.join(ROOT, "..", "final_phone_sensor.csv"),
]


def load_dataset():
    for path in CANDIDATE_CSVS:
        path = os.path.normpath(path)
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin1")
        return df, path
    raise FileNotFoundError("final_phone_sensor.csv not found in expected locations")


def prepare_xy(df):
    df = clean_columns(df.copy())
    missing = [c for c in REQUIRED_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required features after clean_columns: {missing}")
    if "Status" not in df.columns:
        raise ValueError("Status column missing")

    X = df[REQUIRED_FEATURES].copy()
    for col in REQUIRED_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    y = df["Status"].astype(str).str.lower().str.strip().map(
        {"good": 0, "bad": 1, "critical": 2}
    )
    mask = y.notna() & X.notna().all(axis=1)
    X = X.loc[mask]
    y = y.loc[mask].astype(int)
    return X, y


def augment_rotations(X_vals, num_rotations=15, seed=42):
    """
    Applies random 3D rotations to spatial sensor vectors (Acceleration, Gyroscope,
    Linear Acceleration, Magnetic Field) to ensure orientation invariance.
    """
    augmented_X = [X_vals]
    np.random.seed(seed)
    n_samples = len(X_vals)
    for r_idx in range(num_rotations):
        rot_matrices = R.random(n_samples, random_state=seed + r_idx).as_matrix()
        X_rot = np.zeros_like(X_vals)
        for i in range(n_samples):
            r = rot_matrices[i]
            # Acceleration
            X_rot[i, 0:3] = r @ X_vals[i, 0:3]
            # Gyroscope
            X_rot[i, 3:6] = r @ X_vals[i, 3:6]
            # Linear Acceleration
            X_rot[i, 6:9] = r @ X_vals[i, 6:9]
            # Magnetic field
            X_rot[i, 9:12] = r @ X_vals[i, 9:12]
        augmented_X.append(X_rot)
    return np.vstack(augmented_X)


def train_and_save():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df, path = load_dataset()
    print(f"Loaded {path} · shape={df.shape}")
    X, y = prepare_xy(df)
    print(f"Usable samples: {len(y)} · class counts: {y.value_counts().sort_index().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    medians = X_train.median().fillna(0.0)
    X_train_imp = X_train.fillna(medians).values
    X_test_imp = X_test.fillna(medians).values

    print("Augmenting training dataset with 3D spatial rotations...")
    num_rotations = 15
    X_train_aug = augment_rotations(X_train_imp, num_rotations=num_rotations, seed=42)
    y_train_aug = np.tile(y_train.values, num_rotations + 1)
    print(f"Augmented training samples: {len(y_train_aug)}")

    model = XGBClassifier(
        n_estimators=350,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=2,
        gamma=0.05,
        reg_lambda=1.2,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_aug, y_train_aug)

    y_pred = model.predict(X_test_imp)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    print("\n--- Standard Test Evaluation ---")
    print(classification_report(y_test, y_pred, target_names=["good", "bad", "critical"]))
    print("Confusion matrix:\n", cm)
    print(f"Accuracy: {acc:.4f}")

    # Evaluate on rotated test set for orientation invariance check
    X_test_rot = augment_rotations(X_test_imp, num_rotations=5, seed=99)
    y_test_rot = np.tile(y_test.values, 6)
    y_pred_rot = model.predict(X_test_rot)
    acc_rot = accuracy_score(y_test_rot, y_pred_rot)
    print(f"Rotated Test Accuracy (Orientation Invariance Check): {acc_rot:.4f}")

    # Sanity checks on real class centroids
    for label, name in [(0, "good"), (1, "bad"), (2, "critical")]:
        sample = X[y == label].median().to_frame().T
        pred = int(model.predict(sample.fillna(medians).values)[0])
        proba = model.predict_proba(sample.fillna(medians).values)[0]
        print(f"Centroid {name} -> pred={['good','bad','critical'][pred]} probs={np.round(proba,3)}")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(
        {
            "medians": medians.to_dict(),
            "metrics": {
                "accuracy": float(acc),
                "accuracy_rotated": float(acc_rot),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "cm": cm.tolist(),
                "n_train": int(len(y_train_aug)),
                "n_test": int(len(y_test)),
                "n_total": int(len(y)),
                "source": path,
            },
            "class_centroids": {
                name: X[y == label].median().to_dict()
                for label, name in [(0, "good"), (1, "bad"), (2, "critical")]
            },
        },
        PREPROCESSING_PATH,
    )
    print(f"\nSaved orientation-invariant model -> {MODEL_PATH}")
    print(f"Saved preprocessing -> {PREPROCESSING_PATH}")
    return acc


if __name__ == "__main__":
    train_and_save()

