"""
Абляционный эксперимент по новым аудиопризнакам (Задача 2).

Берём базовые 62 признака и по очереди добавляем к ним ОДНУ новую группу
признаков, чтобы увидеть эффект каждой группы отдельно:
  - baseline                -- только исходные 62 признака
  - baseline + chroma
  - baseline + contrast
  - baseline + pitch
  - baseline + rhythm
  - baseline + mfcc20
  - baseline + mfcc40
  - baseline + всё сразу

Модель для сравнения: Random Forest (модель-победитель из Задачи 1),
без калибровки -- тут нас интересует именно эффект признаков.

Запускать ПОСЛЕ extract_features_extended.py.
Положи этот файл в src/emotion_calibration/ablation_features.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.preprocessing import label_binarize

BASE_FEATURES_PATH = Path("data/processed/features.csv")
EXTENDED_FEATURES_PATH = Path("data/processed/features_extended.csv")
REPORT_PATH = Path("reports/ablation_features.csv")

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
TEST_ACTORS = list(range(17, 21))

FEATURE_GROUP_PREFIXES = {
    "chroma": "chroma_",
    "contrast": "contrast_",
    "pitch": ("pitch_mean", "pitch_std", "voiced_ratio"),
    "rhythm": ("pause_ratio", "n_pauses", "tempo"),
    "mfcc20": "mfcc20_",
    "mfcc40": "mfcc40_",
}


def columns_for_group(df: pd.DataFrame, group_name: str) -> list[str]:
    spec = FEATURE_GROUP_PREFIXES[group_name]
    if isinstance(spec, str):
        return [c for c in df.columns if c.startswith(spec)]
    return [c for c in spec if c in df.columns]


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
    ext_df = pd.read_csv(EXTENDED_FEATURES_PATH)
    df = base_df.merge(ext_df, on="path", how="inner")

    baseline_columns = [c for c in base_df.columns if c not in NON_FEATURE_COLUMNS]

    all_extra_columns = []
    for group in FEATURE_GROUP_PREFIXES:
        all_extra_columns += columns_for_group(df, group)

    configs = {
        "baseline": [],
        "baseline_plus_chroma": columns_for_group(df, "chroma"),
        "baseline_plus_contrast": columns_for_group(df, "contrast"),
        "baseline_plus_pitch": columns_for_group(df, "pitch"),
        "baseline_plus_rhythm": columns_for_group(df, "rhythm"),
        "baseline_plus_mfcc20": columns_for_group(df, "mfcc20"),
        "baseline_plus_mfcc40": columns_for_group(df, "mfcc40"),
        "baseline_plus_all": all_extra_columns,
        "baseline_plus_pitch_contrast": columns_for_group(df, "pitch") + columns_for_group(df, "contrast"),
    }

    train_df = df[df["actor"].isin(TRAIN_ACTORS)]
    test_df = df[df["actor"].isin(TEST_ACTORS)]
    classes = sorted(df["emotion"].unique())

    results = []
    for config_name, extra_columns in configs.items():
        feature_columns = baseline_columns + extra_columns
        X_train, y_train = train_df[feature_columns], train_df["emotion"]
        X_test, y_test = test_df[feature_columns], test_df["emotion"]

        model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
        model.fit(X_train, y_train)

        metrics = evaluate(model, X_test, y_test, classes)
        results.append({"feature_set": config_name, "n_features": len(feature_columns), **metrics})
        print(f"{config_name} ({len(feature_columns)} признаков): {metrics}")

    result_df = pd.DataFrame(results)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(REPORT_PATH, index=False)

    print()
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
