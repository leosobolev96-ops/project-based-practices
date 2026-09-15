"""
Устойчивость выводов к разным actor-split (Задача 6, часть Б).

Мы всё время использовали одно фиксированное разбиение актёров
(train=1-16, calib=21-24, test=17-20). Проверяем, останутся ли выводы
теми же, если разбить актёров по-другому -- несколько раз, каждый раз
без пересечений между train/calib/test (запись одного актёра целиком
находится только в одной части).

Положи этот файл в src/emotion_calibration/actor_split_robustness.py
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
REPORT_PATH = Path("reports/actor_split_robustness.csv")

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

# 24 актёра разбиты на 6 блоков по 4 -- из них собираем разные комбинации
# train (4 блока = 16 актёров) / calib (1 блок = 4 актёра) / test (1 блок = 4 актёра)
BLOCKS = [
    list(range(1, 5)),
    list(range(5, 9)),
    list(range(9, 13)),
    list(range(13, 17)),
    list(range(17, 21)),
    list(range(21, 25)),
]

# (индекс блока для test, индекс блока для calib) -- остальные 4 блока идут в train
SPLITS = {
    "split_original": (4, 5),  # test=17-20, calib=21-24 (как во всех предыдущих задачах)
    "split_swap": (5, 4),  # test=21-24, calib=17-20
    "split_shift_1": (0, 1),  # test=1-4,  calib=5-8
    "split_shift_2": (2, 3),  # test=9-12, calib=13-16
}


def load_data():
    df = pd.read_csv(DATA_PATH)
    feature_columns = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    return df, feature_columns


def split_by_actors(df, actors):
    return df[df["actor"].isin(actors)]


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
    df, feature_columns = load_data()
    classes = sorted(df["emotion"].unique())

    rows = []
    for split_name, (test_block_idx, calib_block_idx) in SPLITS.items():
        test_actors = BLOCKS[test_block_idx]
        calib_actors = BLOCKS[calib_block_idx]
        train_actors = [
            actor
            for i, block in enumerate(BLOCKS)
            if i not in (test_block_idx, calib_block_idx)
            for actor in block
        ]

        train_df = split_by_actors(df, train_actors)
        calib_df = split_by_actors(df, calib_actors)
        test_df = split_by_actors(df, test_actors)

        X_train, y_train = train_df[feature_columns], train_df["emotion"]
        X_calib, y_calib = calib_df[feature_columns], calib_df["emotion"]
        X_test, y_test = test_df[feature_columns], test_df["emotion"]

        print(f"{split_name}: train={train_actors}, calib={calib_actors}, test={test_actors}")

        base_model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
        base_model.fit(X_train, y_train)
        metrics_before = evaluate(base_model, X_test, y_test, classes)

        calibrated_model = CalibratedClassifierCV(FrozenEstimator(base_model), method="sigmoid")
        calibrated_model.fit(X_calib, y_calib)
        metrics_after = evaluate(calibrated_model, X_test, y_test, classes)

        rows.append({"split": split_name, "stage": "before", **metrics_before})
        rows.append({"split": split_name, "stage": "after", **metrics_after})

    result_df = pd.DataFrame(rows)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(REPORT_PATH, index=False)

    print()
    print(result_df.to_string(index=False))

    # короткая сводка: в скольких split-ах калибровка реально улучшила Brier Score
    pivot = result_df.pivot(index="split", columns="stage", values="brier_score")
    improved = (pivot["after"] < pivot["before"]).sum()
    print(f"\nКалибровка улучшила Brier Score в {improved} из {len(SPLITS)} split-ов.")


if __name__ == "__main__":
    main()
