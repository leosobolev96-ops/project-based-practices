"""
Оценка пользы аугментации тренировочных данных (Задача 3).

Сравнивает Random Forest, обученный:
  - no_augmentation    -- только 960 оригинальных записей (актёры 1-16)
  - with_augmentation  -- 960 оригиналов + noise + pitch_shift + time_stretch
                          (итого 3840 примеров для обучения)

Тестируется на НЕизменённых актёрах 17-20 (обычный features.csv) --
так проверка остаётся честной, аугментации туда не попадают.

Запускать ПОСЛЕ augment_train.py.
Положи этот файл в src/emotion_calibration/train_with_augmentation.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.preprocessing import label_binarize

BASE_FEATURES_PATH = Path("data/processed/features.csv")
AUGMENTED_FEATURES_PATH = Path("data/processed/features_augmented_train.csv")
REPORT_PATH = Path("reports/augmentation_comparison.csv")

NON_FEATURE_COLUMNS = [
    "path",
    "filename",
    "emotion",
    "intensity",
    "statement",
    "repetition",
    "actor",
    "actor_gender",
    "augmentation",
]

TRAIN_ACTORS = list(range(1, 17))
TEST_ACTORS = list(range(17, 21))


def multiclass_brier_score(y_true_labels, y_proba, classes) -> float:
    y_true_bin = label_binarize(y_true_labels, classes=classes)
    return float(np.mean(np.sum((y_proba - y_true_bin) ** 2, axis=1)))


def evaluate(model, X, y_true, classes) -> dict:
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "log_loss": log_loss(y_true, y_proba, labels=classes),
        "brier_score": multiclass_brier_score(y_true, y_proba, classes),
    }


def main() -> None:
    base_df = pd.read_csv(BASE_FEATURES_PATH)
    aug_df = pd.read_csv(AUGMENTED_FEATURES_PATH)

    feature_columns = [c for c in base_df.columns if c not in NON_FEATURE_COLUMNS]
    classes = sorted(base_df["emotion"].unique())

    test_df = base_df[base_df["actor"].isin(TEST_ACTORS)]
    X_test, y_test = test_df[feature_columns], test_df["emotion"]

    train_original = base_df[base_df["actor"].isin(TRAIN_ACTORS)]
    X_train_orig, y_train_orig = train_original[feature_columns], train_original["emotion"]

    X_train_aug, y_train_aug = aug_df[feature_columns], aug_df["emotion"]

    results = []

    model_no_aug = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    model_no_aug.fit(X_train_orig, y_train_orig)
    metrics_no_aug = evaluate(model_no_aug, X_test, y_test, classes)
    results.append({"setup": "no_augmentation", "n_train": len(X_train_orig), **metrics_no_aug})
    print(f"no_augmentation ({len(X_train_orig)} записей): {metrics_no_aug}")

    model_aug = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    model_aug.fit(X_train_aug, y_train_aug)
    metrics_aug = evaluate(model_aug, X_test, y_test, classes)
    results.append({"setup": "with_augmentation", "n_train": len(X_train_aug), **metrics_aug})
    print(f"with_augmentation ({len(X_train_aug)} записей): {metrics_aug}")

    result_df = pd.DataFrame(results)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(REPORT_PATH, index=False)

    print()
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
