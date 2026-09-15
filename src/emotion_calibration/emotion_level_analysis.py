"""
Статистический анализ калибровки по эмоциям (Задача 5).

Для каждой эмоции (в режиме one-vs-rest: "эта эмоция" против всех
остальных) проверяем:
  - Brier score до и после калибровки
  - ECE (Expected Calibration Error) до и после калибровки

Чтобы не полагаться на один случайный результат, calibration-набор
20 раз пересэмплируется (bootstrap, с возвращением) и заново обучается
калибратор -- так видно, для каких эмоций калибровка помогает СТАБИЛЬНО
(в большинстве повторов), а для каких эффект случайный или отсутствует.

Положи этот файл в src/emotion_calibration/emotion_level_analysis.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator

DATA_PATH = Path("data/processed/features.csv")
REPORT_PATH = Path("reports/emotion_level_calibration_analysis.csv")

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

N_REPEATS = 20
N_BINS_ECE = 10


def load_data():
    df = pd.read_csv(DATA_PATH)
    feature_columns = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    return df, feature_columns


def split_by_actors(df, actors):
    return df[df["actor"].isin(actors)]


def one_vs_rest_brier(y_true, proba_col, class_label) -> float:
    indicator = (y_true == class_label).astype(int).to_numpy()
    return float(np.mean((proba_col - indicator) ** 2))


def ece(y_true_indicator: np.ndarray, proba_col: np.ndarray, n_bins: int = N_BINS_ECE) -> float:
    bin_edges = np.linspace(0, 1, n_bins + 1)
    n = len(proba_col)
    total = 0.0
    for i in range(n_bins):
        mask = (proba_col > bin_edges[i]) & (proba_col <= bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_confidence = proba_col[mask].mean()
        bin_accuracy = y_true_indicator[mask].mean()
        total += (mask.sum() / n) * abs(bin_confidence - bin_accuracy)
    return float(total)


def per_class_metrics(model, X_test, y_test, classes) -> dict:
    proba = model.predict_proba(X_test)
    model_classes = list(model.classes_)

    result = {}
    for class_label in classes:
        col_idx = model_classes.index(class_label)
        proba_col = proba[:, col_idx]
        indicator = (y_test == class_label).astype(int).to_numpy()

        result[class_label] = {
            "brier": one_vs_rest_brier(y_test, proba_col, class_label),
            "ece": ece(indicator, proba_col),
        }
    return result


def main() -> None:
    df, feature_columns = load_data()

    train_df = split_by_actors(df, TRAIN_ACTORS)
    calib_df = split_by_actors(df, CALIBRATION_ACTORS)
    test_df = split_by_actors(df, TEST_ACTORS)

    X_train, y_train = train_df[feature_columns], train_df["emotion"]
    X_calib_full, y_calib_full = calib_df[feature_columns], calib_df["emotion"]
    X_test, y_test = test_df[feature_columns], test_df["emotion"]

    classes = sorted(df["emotion"].unique())

    print("Обучаю базовую модель (один раз)...")
    base_model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    base_model.fit(X_train, y_train)

    before = per_class_metrics(base_model, X_test, y_test, classes)

    after_brier = {c: [] for c in classes}
    after_ece = {c: [] for c in classes}

    n_calib = len(X_calib_full)
    for seed in range(N_REPEATS):
        rng = np.random.RandomState(seed)
        idx = rng.choice(n_calib, size=n_calib, replace=True)
        X_calib_boot = X_calib_full.iloc[idx]
        y_calib_boot = y_calib_full.iloc[idx]

        calibrated = CalibratedClassifierCV(FrozenEstimator(base_model), method="sigmoid")
        calibrated.fit(X_calib_boot, y_calib_boot)

        metrics = per_class_metrics(calibrated, X_test, y_test, classes)
        for c in classes:
            after_brier[c].append(metrics[c]["brier"])
            after_ece[c].append(metrics[c]["ece"])

        if (seed + 1) % 5 == 0:
            print(f"Повтор {seed + 1}/{N_REPEATS} готов")

    rows = []
    for c in classes:
        brier_before = before[c]["brier"]
        ece_before = before[c]["ece"]

        brier_after_arr = np.array(after_brier[c])
        ece_after_arr = np.array(after_ece[c])

        brier_improve_rate = float(np.mean(brier_after_arr < brier_before))
        ece_improve_rate = float(np.mean(ece_after_arr < ece_before))

        rows.append(
            {
                "emotion": c,
                "brier_before": brier_before,
                "brier_after_mean": float(brier_after_arr.mean()),
                "brier_improvement_rate": brier_improve_rate,
                "ece_before": ece_before,
                "ece_after_mean": float(ece_after_arr.mean()),
                "ece_improvement_rate": ece_improve_rate,
            }
        )

    result_df = pd.DataFrame(rows).sort_values("brier_improvement_rate", ascending=False)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(REPORT_PATH, index=False)

    print()
    print(result_df.to_string(index=False))
    print()
    print(
        "brier_improvement_rate / ece_improvement_rate -- доля из "
        f"{N_REPEATS} повторов, где калибровка дала лучший результат, чем без неё."
    )
    print("1.0 = калибровка стабильно помогает, 0.0 = стабильно вредит, ~0.5 = эффект случайный.")


if __name__ == "__main__":
    main()
