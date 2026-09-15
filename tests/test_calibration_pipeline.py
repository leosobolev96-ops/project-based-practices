"""
CI-проверка пайплайна на небольшом fixture-наборе (Задача 7).

Не использует настоящий датасет RAVDESS (200+ МБ) -- вместо этого
генерирует маленький синтетический набор "признаков" прямо в памяти,
чтобы быстро проверить, что весь пайплайн (разбиение по актёрам, обучение,
калибровка, подсчёт метрик) вообще работает и не падает с ошибкой.

Положи этот файл в tests/test_calibration_pipeline.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from emotion_calibration.calibrate import evaluate, multiclass_brier_score, split_by_actors


def make_fixture_df(n_per_actor=8, n_actors=9, n_features=6, seed=1) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    emotions = ["angry", "happy", "sad", "neutral"]
    rows = []
    for actor in range(1, n_actors + 1):
        for i in range(n_per_actor):
            row = {f"feat_{j}": rng.randn() for j in range(n_features)}
            row["actor"] = actor
            row["emotion"] = emotions[i % len(emotions)]
            rows.append(row)
    return pd.DataFrame(rows)


def test_split_by_actors():
    df = make_fixture_df()
    subset = split_by_actors(df, [1, 2])
    assert set(subset["actor"].unique()) <= {1, 2}
    assert len(subset) > 0


def test_multiclass_brier_score_range():
    df = make_fixture_df()
    classes = sorted(df["emotion"].unique())
    y_true = df["emotion"]
    y_proba = np.full((len(df), len(classes)), 1 / len(classes))
    score = multiclass_brier_score(y_true, y_proba, classes)
    assert 0 <= score <= 2


def test_end_to_end_fixture_pipeline():
    """Проверяем весь пайплайн (обучение + оценка) на маленьком наборе."""
    df = make_fixture_df(n_per_actor=8, n_actors=9, n_features=6, seed=1)
    feature_columns = [c for c in df.columns if c.startswith("feat_")]

    train_df = split_by_actors(df, [1, 2, 3, 4, 5])
    test_df = split_by_actors(df, [8, 9])

    classes = sorted(df["emotion"].unique())

    model = RandomForestClassifier(n_estimators=10, random_state=0)
    model.fit(train_df[feature_columns], train_df["emotion"])

    metrics = evaluate(model, test_df[feature_columns], test_df["emotion"], classes, "test")

    assert 0 <= metrics["accuracy"] <= 1
    assert metrics["log_loss"] >= 0
    assert metrics["brier_score"] >= 0
