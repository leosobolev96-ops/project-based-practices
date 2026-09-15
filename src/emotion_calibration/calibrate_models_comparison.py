"""
Расширенное сравнение моделей и режимов калибровки.

Что делает:
- обучает 3 модели на актёрах 1-16: Random Forest, SVM, Gradient Boosting
- для каждой модели пробует 4 режима калибровки:
    - none        -- без калибровки
    - sigmoid      -- калибровка на актёрах 21-24 (Platt scaling)
    - isotonic     -- калибровка на актёрах 21-24 (isotonic regression)
    - cv_sigmoid   -- калибровка через 5-fold CV прямо на актёрах 1-16,
                      без отдельного calibration-набора
- оценивает все 12 комбинаций (3 модели x 4 режима) на актёрах 17-20
- сохраняет большую сравнительную таблицу

Положи этот файл в src/emotion_calibration/calibrate_models_comparison.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC

DATA_PATH = Path("data/processed/features.csv")
REPORT_PATH = Path("reports/calibration_models_comparison.csv")

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

MODELS = {
    "random_forest": RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight="balanced"
    ),
    "svm": SVC(kernel="rbf", probability=True, random_state=42, class_weight="balanced"),
    "gradient_boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
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


def build_calibrated(base_model, mode, X_train, y_train, X_calib, y_calib):
    """Возвращает модель, готовую к predict_proba, для заданного режима калибровки."""
    if mode == "none":
        return base_model

    if mode in ("sigmoid", "isotonic"):
        calibrated = CalibratedClassifierCV(FrozenEstimator(base_model), method=mode)
        calibrated.fit(X_calib, y_calib)
        return calibrated

    if mode == "cv_sigmoid":
        # Отдельная, "свежая" модель того же типа: калибруется через 5-fold CV
        # прямо на тренировочных данных, без отдельного calibration-набора.
        fresh_model = clone(base_model)
        calibrated = CalibratedClassifierCV(fresh_model, method="sigmoid", cv=5)
        calibrated.fit(X_train, y_train)
        return calibrated

    raise ValueError(f"Unknown calibration mode: {mode}")


def main() -> None:
    df, feature_columns = load_data()

    train_df = split_by_actors(df, TRAIN_ACTORS)
    calib_df = split_by_actors(df, CALIBRATION_ACTORS)
    test_df = split_by_actors(df, TEST_ACTORS)

    X_train, y_train = train_df[feature_columns], train_df["emotion"]
    X_calib, y_calib = calib_df[feature_columns], calib_df["emotion"]
    X_test, y_test = test_df[feature_columns], test_df["emotion"]

    classes = sorted(df["emotion"].unique())

    modes = ["none", "sigmoid", "isotonic", "cv_sigmoid"]
    results = []

    for model_name, model_template in MODELS.items():
        print(f"Обучаю базовую модель: {model_name} ...")
        base_model = clone(model_template)
        base_model.fit(X_train, y_train)

        for mode in modes:
            print(f"  -> режим калибровки: {mode}")
            fitted = build_calibrated(base_model, mode, X_train, y_train, X_calib, y_calib)
            metrics = evaluate(fitted, X_test, y_test, classes)
            results.append({"model": model_name, "calibration_mode": mode, **metrics})

    result_df = pd.DataFrame(results)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(REPORT_PATH, index=False)

    print()
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()