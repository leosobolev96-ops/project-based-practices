"""
Углублённый анализ по подгруппам (Задача 6, часть А).

Смотрим на метрики калибровки ОТДЕЛЬНО для:
  - каждого уровня intensity (normal / strong)
  - каждого варианта statement (какая фраза произносится)
  - пола актёра (male / female)

Это позволяет увидеть, не работает ли калибровка сильно по-разному для
разных подгрупп данных (что было бы незаметно, если смотреть только на
общие метрики по всему тестовому набору).

Положи этот файл в src/emotion_calibration/subgroup_analysis.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import label_binarize

DATA_PATH = Path("data/processed/features.csv")
REPORT_PATH = Path("reports/subgroup_calibration_analysis.csv")

NON_FEATURE_COLUMNS = [
    "path",
    "filename",
    "emotion",
    "intensity",
    "statement",
    "repetition",
    "actor",
    "actor_gender",
]

TRAIN_ACTORS = list(range(1, 17))
CALIBRATION_ACTORS = list(range(21, 25))
TEST_ACTORS = list(range(17, 21))

GROUP_COLUMNS = ["intensity", "statement", "actor_gender"]


def load_data():
    df = pd.read_csv(DATA_PATH)
    feature_columns = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    return df, feature_columns


def split_by_actors(df, actors):
    return df[df["actor"].isin(actors)]


def multiclass_brier_score(y_true_labels, y_proba, classes) -> float:
    y_true_bin = label_binarize(y_true_labels, classes=classes)
    return float(np.mean(np.sum((y_proba - y_true_bin) ** 2, axis=1)))


def subgroup_metrics(y_true, y_proba, classes, mask, stage: str, group_name: str, level) -> dict:
    y_true_sub = y_true[mask]
    y_proba_sub = y_proba[mask]
    y_pred_sub = np.array(classes)[y_proba_sub.argmax(axis=1)]

    present_classes = sorted(set(y_true_sub.unique()) | set(classes))
    return {
        "stage": stage,
        "group": group_name,
        "level": level,
        "n": int(mask.sum()),
        "accuracy": accuracy_score(y_true_sub, y_pred_sub),
        "log_loss": log_loss(y_true_sub, y_proba_sub, labels=classes),
        "brier_score": multiclass_brier_score(y_true_sub, y_proba_sub, classes),
    }


def main() -> None:
    df, feature_columns = load_data()

    train_df = split_by_actors(df, TRAIN_ACTORS)
    calib_df = split_by_actors(df, CALIBRATION_ACTORS)
    test_df = split_by_actors(df, TEST_ACTORS)

    X_train, y_train = train_df[feature_columns], train_df["emotion"]
    X_calib, y_calib = calib_df[feature_columns], calib_df["emotion"]
    X_test, y_test = test_df[feature_columns], test_df["emotion"]

    classes = sorted(df["emotion"].unique())

    print("Обучаю базовую модель...")
    base_model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    base_model.fit(X_train, y_train)

    print("Калибрую...")
    calibrated_model = CalibratedClassifierCV(FrozenEstimator(base_model), method="sigmoid")
    calibrated_model.fit(X_calib, y_calib)

    proba_before = base_model.predict_proba(X_test)
    proba_after = calibrated_model.predict_proba(X_test)

    rows = []
    for group_name in GROUP_COLUMNS:
        for level in sorted(test_df[group_name].astype(str).unique()):
            mask = (test_df[group_name].astype(str) == level).to_numpy()
            rows.append(subgroup_metrics(y_test, proba_before, classes, mask, "before", group_name, level))
            rows.append(subgroup_metrics(y_test, proba_after, classes, mask, "after", group_name, level))

    result_df = pd.DataFrame(rows)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(REPORT_PATH, index=False)

    print()
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
