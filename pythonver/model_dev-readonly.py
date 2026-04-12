# ============================================================
# FINAL MODEL TESTING SUITE
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.inspection import permutation_importance, PartialDependenceDisplay


print("\n================ FINAL MODEL TESTING ================")

SCRIPT_DIR = Path(__file__).resolve().parent


def _resolve_project_root(base_dir: Path) -> Path:
    # Support both execution contexts:
    # - pythonver/*.py (dataset is in parent dir)
    # - notebook or script in project root (dataset is in same dir)
    if (base_dir / "dataset_labeled_wsn.xlsx").exists():
        return base_dir
    if (base_dir.parent / "dataset_labeled_wsn.xlsx").exists():
        return base_dir.parent
    return base_dir


PROJECT_ROOT = _resolve_project_root(SCRIPT_DIR)
DATA_PATH = PROJECT_ROOT / "dataset_labeled_wsn.xlsx"
MODEL_PATH = PROJECT_ROOT / "rf_wsn_multiclass_best_pipeline.pkl"


def _bootstrap_context_from_files():
    missing_files = [str(p) for p in [DATA_PATH, MODEL_PATH] if not p.exists()]
    if missing_files:
        raise FileNotFoundError(
            "File wajib tidak ditemukan: "
            + ", ".join(missing_files)
            + ". Pastikan jalankan script dari folder proyek."
        )

    df = pd.read_excel(DATA_PATH)
    if "label" not in df.columns:
        raise KeyError("Kolom 'label' tidak ditemukan di dataset_labeled_wsn.xlsx")

    X = df.drop(columns=["label"])
    y = df["label"]

    X_train_local, X_test_local, y_train_local, y_test_local = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    return {
        "best_rf_pipeline": joblib.load(MODEL_PATH),
        "X_test": X_test_local,
        "y_test": y_test_local,
        "label_order": sorted(y.unique()),
        "X_train": X_train_local,
        "y_train": y_train_local,
    }


required_globals = [
    "best_rf_pipeline",
    "X_test",
    "y_test",
    "label_order",
    "X_train",
    "y_train",
]
missing_globals = [name for name in required_globals if name not in globals()]
if missing_globals:
    print(
        "Konteks belum lengkap ("
        + ", ".join(missing_globals)
        + "). Mencoba bootstrap otomatis dari file dataset + model..."
    )
    globals().update(_bootstrap_context_from_files())

# ============================================================
# 1. TEST SET EVALUATION
# ============================================================

y_pred_test = best_rf_pipeline.predict(X_test)

print("\n=== CLASSIFICATION REPORT (TEST SET) ===")
print(
    classification_report(
        y_test,
        y_pred_test,
        labels=label_order,
        target_names=label_order,
        zero_division=0,
    )
)

test_accuracy = accuracy_score(y_test, y_pred_test)
print(f"\nTest Accuracy: {test_accuracy:.4f}")

report_dict = classification_report(
    y_test,
    y_pred_test,
    labels=label_order,
    target_names=label_order,
    zero_division=0,
    output_dict=True
)

f1_macro_test = report_dict["macro avg"]["f1-score"]
print(f"Test F1-macro: {f1_macro_test:.4f}")


# ============================================================
# 2. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred_test, labels=label_order)

plt.figure(figsize=(7,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_order,
    yticklabels=label_order
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Final RF")
plt.tight_layout()
plt.show()


# ============================================================
# 3. PER-CLASS RECALL (PENTING UNTUK PREVENTIVE)
# ============================================================

print("\n=== RECALL PER KELAS ===")

recalls = []
for label in label_order:
    recall_value = report_dict[label]["recall"]
    recalls.append(recall_value)
    print(f"Recall kelas {label}: {recall_value:.4f}")

plt.figure(figsize=(8,5))
plt.bar(label_order, recalls)
plt.ylim(0, 1.05)
plt.title("Recall per Kelas - Final RF")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()


# ============================================================
# 4. CONFIDENCE SCORE (PREDICTION PROBABILITY)
# ============================================================

y_proba_test = best_rf_pipeline.predict_proba(X_test)
confidence_scores = np.max(y_proba_test, axis=1)

print("\nContoh confidence score (5 data pertama):")
print(confidence_scores[:5])


# ============================================================
# 5. CROSS VALIDATION STABILITY (TRAIN SET)
# ============================================================

cv_scores = cross_val_score(
    best_rf_pipeline,
    X_train,
    y_train,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1
)

print("\n=== CROSS VALIDATION (TRAIN SET) ===")
print("Mean F1-macro:", cv_scores.mean())
print("Std  F1-macro:", cv_scores.std())


# ============================================================
# 6. OVERFITTING CHECK
# ============================================================

train_accuracy = best_rf_pipeline.score(X_train, y_train)
test_accuracy  = best_rf_pipeline.score(X_test, y_test)

print("\n=== OVERFITTING CHECK ===")
print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy : {test_accuracy:.4f}")
print(f"Gap           : {train_accuracy - test_accuracy:.4f}")


# ============================================================
# 7. PERMUTATION IMPORTANCE (GLOBAL EXPLAINABILITY)
# ============================================================

print("\n=== PERMUTATION IMPORTANCE ===")

perm_importance = permutation_importance(
    best_rf_pipeline,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

perm_df = pd.DataFrame({
    "feature": X_train.columns,
    "importance_mean": perm_importance.importances_mean
}).sort_values("importance_mean", ascending=False)

print(perm_df)

plt.figure(figsize=(8,5))
plt.barh(perm_df["feature"], perm_df["importance_mean"])
plt.gca().invert_yaxis()
plt.title("Permutation Feature Importance")
plt.tight_layout()
plt.show()


# ============================================================
# PARTIAL DEPENDENCE - MODEL TUNED (MULTICLASS FIX)
# ============================================================

print("\n=== PDP - MODEL TUNED ===")

# Ambil fitur paling penting dari permutation
top_features = perm_df["feature"].head(2).tolist()

# Pilih target class untuk PDP
pipeline_classes = list(getattr(best_rf_pipeline, "classes_", label_order))
if "energy_fault" in pipeline_classes:
    target_class_label = "energy_fault"
else:
    target_class_label = pipeline_classes[0]

print(f"Menganalisis PDP untuk kelas: {target_class_label}")

PartialDependenceDisplay.from_estimator(
    best_rf_pipeline,
    X_test,
    features=top_features,
    target=target_class_label
)

plt.tight_layout()
plt.show()

print("\n================ TESTING COMPLETE ================")

# ============================================================
# 9. FEATURE ABLATION TEST
# ============================================================

print("\n================ FEATURE ABLATION TEST ================")

from sklearn.base import clone

ablation_results = []

for feature in X_train.columns:
    print(f"\nMenghapus fitur: {feature}")

    # Drop satu fitur
    X_train_drop = X_train.drop(columns=[feature])
    X_test_drop  = X_test.drop(columns=[feature])

    # Clone model supaya tidak merusak model utama
    model_clone = clone(best_rf_pipeline)

    # Retrain tanpa fitur tersebut
    model_clone.fit(X_train_drop, y_train)

    # Evaluate
    y_pred_drop = model_clone.predict(X_test_drop)

    report_drop = classification_report(
        y_test,
        y_pred_drop,
        labels=label_order,
        target_names=label_order,
        zero_division=0,
        output_dict=True
    )

    f1_drop = report_drop["macro avg"]["f1-score"]

    ablation_results.append({
        "removed_feature": feature,
        "f1_macro_after_removal": f1_drop
    })

ablation_df = pd.DataFrame(ablation_results)

# Hitung drop performa dibanding model asli
baseline_f1 = f1_macro_test

ablation_df["f1_drop"] = baseline_f1 - ablation_df["f1_macro_after_removal"]
ablation_df = ablation_df.sort_values("f1_drop", ascending=False)

print("\n=== HASIL FEATURE ABLATION ===")
print(ablation_df)

# Visualisasi
plt.figure(figsize=(8,5))
plt.barh(ablation_df["removed_feature"], ablation_df["f1_drop"])
plt.gca().invert_yaxis()
plt.title("F1-macro Drop Setelah Feature Removal")
plt.xlabel("Penurunan F1-macro")
plt.tight_layout()
plt.show()

# ============================================================
# 10. NOISE ROBUSTNESS TEST
# ============================================================

print("\n================ NOISE ROBUSTNESS TEST ================")

noise_levels = [0.01, 0.03, 0.05, 0.1]  # 1% - 10% noise
noise_results = []

for noise in noise_levels:
    print(f"\nMenambahkan noise level: {noise}")

    X_test_noisy = X_test.copy()

    # Tambahkan Gaussian noise
    for col in X_test_noisy.columns:
        std = X_test_noisy[col].std()
        noise_vector = np.random.normal(0, noise * std, size=len(X_test_noisy))
        X_test_noisy[col] += noise_vector

    # Predict
    y_pred_noisy = best_rf_pipeline.predict(X_test_noisy)

    report_noisy = classification_report(
        y_test,
        y_pred_noisy,
        labels=label_order,
        target_names=label_order,
        zero_division=0,
        output_dict=True
    )

    f1_noisy = report_noisy["macro avg"]["f1-score"]

    noise_results.append({
        "noise_level": noise,
        "f1_macro": f1_noisy
    })

noise_df = pd.DataFrame(noise_results)

print("\n=== HASIL NOISE ROBUSTNESS ===")
print(noise_df)

# Visualisasi
plt.figure(figsize=(8,5))
plt.plot(noise_df["noise_level"], noise_df["f1_macro"], marker="o")
plt.title("F1-macro vs Noise Level")
plt.xlabel("Noise Level")
plt.ylabel("F1-macro")
plt.ylim(0,1)
plt.tight_layout()
plt.show()
