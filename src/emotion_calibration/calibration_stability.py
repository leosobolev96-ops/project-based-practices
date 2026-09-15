"""
Устойчивость посткалибровки вероятностей (Задача 4).

Проверяем, насколько результат калибровки зависит от:
  - размера calibration-набора (100 записей против всех 240)
  - случайного выбора конкретных записей в calibration-набор (10 seed)

И дополнительно строим bootstrap-доверительные интервалы: пересэмплируем
тестовый набор (актёры 17-20) много раз, чтобы понять, в каком диапазоне
могли бы оказаться метрики, если бы тестовая выборка была чуть другой.

Базовая модель (Random Forest на актёрах 1-16) обучается один раз и не
меняется -- варьируется только сам процесс калибровки, чтобы изолированно
оценить именно её устойчивость.

Положи этот файл в src/emotion_calibration/calibration_stability.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.preprocessing import label_binarize

DATA_PATH = Path("data/processed/features.csv")
SEEDS_REPORT_PATH = Path("reports/calibration_stability_seeds.csv")
SUMMARY_REPORT_PATH = Path("reports/calibration_stability_summary.csv")
BOOTSTRAP_REPORT_PATH = Path("reports/calibration_stability_bootstrap_ci.csv")

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

CALIB_SIZES = [100, 240]
SEEDS = list(range(10))
N_BOOTSTRAP = 1000


def load_data():
    df = pd.read_csv(DATA_PATH)
    feature_columns = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    return df, feature_columns


def split_by_actors(df, actors):
    return df[df["actor"].isin(actors)]


def multiclass_brier_score(y_true_labels, y_proba, classes) -> float:
    y_true_bin = label_binarize(y_true_labels, classes=classes)
    return float(np.mean(np.sum((y_proba - y_true_bin) ** 2, axis=1)))


def point_metrics(y_true, y_proba, classes) -> dict:
    y_pred = np.array(classes)[y_proba.argmax(axis=1)]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "log_loss": log_loss(y_true, y_proba, labels=classes),
        "brier_score": multiclass_brier_score(y_true, y_proba, classes),
    }


def fit_calibrated(base_model, X_calib_full, y_calib_full, calib_size, seed):
    rng = np.random.RandomState(seed)
    if calib_size < len(X_calib_full):
        idx = rng.choice(len(X_calib_full), size=calib_size, replace=False)
    else:
        idx = np.arange(len(X_calib_full))

    X_calib_sub = X_calib_full.iloc[idx]
    y_calib_sub = y_calib_full.iloc[idx]

    calibrated = CalibratedClassifierCV(FrozenEstimator(base_model), method="sigmoid")
    calibrated.fit(X_calib_sub, y_calib_sub)
    return calibrated


def bootstrap_ci(y_test, y_proba, classes, n_bootstrap=N_BOOTSTRAP, seed=0):
    """Bootstrap-доверительный интервал (95%) для log_loss и brier_score."""
    rng = np.random.RandomState(seed)
    n = len(y_test)
    y_test_arr = np.asarray(y_test)

    boot_log_loss, boot_brier = [], []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        y_true_b = y_test_arr[idx]
        y_proba_b = y_proba[idx]
        boot_log_loss.append(log_loss(y_true_b, y_proba_b, labels=classes))
        boot_brier.append(multiclass_brier_score(y_true_b, y_proba_b, classes))

    return {
        "log_loss_mean": float(np.mean(boot_log_loss)),
        "log_loss_ci_low": float(np.percentile(boot_log_loss, 2.5)),
        "log_loss_ci_high": float(np.percentile(boot_log_loss, 97.5)),
        "brier_mean": float(np.mean(boot_brier)),
        "brier_ci_low": float(np.percentile(boot_brier, 2.5)),
        "brier_ci_high": float(np.percentile(boot_brier, 97.5)),
    }


def main() -> None:
    df, feature_columns = load_data()

    train_df = split_by_actors(df, TRAIN_ACTORS)
    calib_df = split_by_actors(df, CALIBRATION_ACTORS)
    test_df = split_by_actors(df, TEST_ACTORS)

    X_train, y_train = train_df[feature_columns], train_df["emotion"]
    X_calib_full, y_calib_full = calib_df[feature_columns], calib_df["emotion"]
    X_test, y_test = test_df[feature_columns], test_df["emotion"]

    classes = sorted(df["emotion"].unique())

    print("Обучаю базовую модель (один раз, дальше не меняется)...")
    base_model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    base_model.fit(X_train, y_train)

    # 1) Разброс по seed для каждого размера calibration-набора
    seed_rows = []
    for calib_size in CALIB_SIZES:
        for seed in SEEDS:
            calibrated = fit_calibrated(base_model, X_calib_full, y_calib_full, calib_size, seed)
            y_proba = calibrated.predict_proba(X_test)
            metrics = point_metrics(y_test, y_proba, classes)
            seed_rows.append({"calib_size": calib_size, "seed": seed, **metrics})
        print(f"Готово: calib_size={calib_size}, {len(SEEDS)} seed")

    seed_df = pd.DataFrame(seed_rows)
    SEEDS_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    seed_df.to_csv(SEEDS_REPORT_PATH, index=False)

    summary_df = (
        seed_df.groupby("calib_size")[["accuracy", "macro_f1", "log_loss", "brier_score"]]
        .agg(["mean", "std"])
    )
    summary_df.to_csv(SUMMARY_REPORT_PATH)

    print("\nРазброс метрик по 10 seed для каждого размера calibration-набора:")
    print(summary_df)

    # 2) Bootstrap-доверительные интервалы (для seed=0, каждый размер)
    bootstrap_rows = []
    for calib_size in CALIB_SIZES:
        calibrated = fit_calibrated(base_model, X_calib_full, y_calib_full, calib_size, seed=0)
        y_proba = calibrated.predict_proba(X_test)
        ci = bootstrap_ci(y_test, y_proba, classes)
        bootstrap_rows.append({"calib_size": calib_size, **ci})

    bootstrap_df = pd.DataFrame(bootstrap_rows)
    bootstrap_df.to_csv(BOOTSTRAP_REPORT_PATH, index=False)

    print("\nBootstrap 95% доверительные интервалы (по тестовому набору):")
    print(bootstrap_df.to_string(index=False))


if __name__ == "__main__":
    main()
