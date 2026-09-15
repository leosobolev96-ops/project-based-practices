"""Обучение базовых классификаторов эмоций на аудиопризнаках."""

from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = Path("data/processed/features.csv")
OUTPUT = Path("reports/model_metrics.csv")
SEED = 42


def feature_columns(data: pd.DataFrame) -> list[str]:
    """Возвращает числовые аудиопризнаки без столбцов метаданных."""
    metadata = {
        "path", "filename", "emotion", "intensity", "statement",
        "repetition", "actor", "actor_gender",
    }
    return [column for column in data.columns if column not in metadata]


def multiclass_brier_score(y_true, probabilities, classes) -> float:
    """Вычисляет многоклассовый Brier Score по one-hot-меткам."""
    class_to_index = {label: index for index, label in enumerate(classes)}
    target = pd.get_dummies(pd.Series(y_true), dtype=float).reindex(
        columns=classes, fill_value=0.0
    ).to_numpy()
    # Pandas может изменить порядок столбцов для строковых классов NumPy.
    if target.shape[1] != len(classes):
        target = pd.DataFrame(
            [[float(class_to_index[label] == index) for index in range(len(classes))]
             for label in y_true]
        ).to_numpy()
    return float(((probabilities - target) ** 2).sum(axis=1).mean())


def expected_calibration_error(y_true, probabilities, classes, bins: int = 10) -> float:
    """Возвращает ожидаемую ошибку калибровки по уверенности модели."""
    return calibration_errors(y_true, probabilities, classes, bins)["ece"]


def calibration_errors(y_true, probabilities, classes, bins: int = 10) -> dict[str, float]:
    """Вычисляет ECE, MCE и адаптивный ECE."""
    predictions = classes[probabilities.argmax(axis=1)]
    confidence = probabilities.max(axis=1)
    correct = (predictions == pd.Series(y_true).to_numpy()).astype(float)

    def summarize(groups):
        weighted_error = 0.0
        maximum_error = 0.0
        for group in groups:
            if not len(group):
                continue
            error = abs(correct[group].mean() - confidence[group].mean())
            weighted_error += len(group) * error
            maximum_error = max(maximum_error, error)
        return weighted_error / len(confidence), maximum_error

    fixed_edges = np.linspace(0.0, 1.0, bins + 1)
    fixed_groups = [
        np.flatnonzero(
            (confidence >= left)
            & (confidence <= right if right == 1.0 else confidence < right)
        )
        for left, right in zip(fixed_edges[:-1], fixed_edges[1:])
    ]
    adaptive_edges = np.quantile(confidence, np.linspace(0, 1, bins + 1))
    adaptive_groups = [
        np.flatnonzero(
            (confidence >= left)
            & (confidence <= right if index == bins - 1 else confidence < right)
        )
        for index, (left, right) in enumerate(
            zip(adaptive_edges[:-1], adaptive_edges[1:])
        )
    ]
    ece, mce = summarize(fixed_groups)
    adaptive_ece, _ = summarize(adaptive_groups)
    return {
        "ece": float(ece),
        "mce": float(mce),
        "adaptive_ece": float(adaptive_ece),
    }


def main() -> None:
    data = pd.read_csv(FEATURES)
    columns = feature_columns(data)

    train = data[data["actor"] <= 16]
    calibration = data[data["actor"] >= 21]
    test = data[data["actor"].between(17, 20)]
    x_train, y_train = train[columns], train["emotion"]
    x_calibration = calibration[columns]
    y_calibration = calibration["emotion"]

    group_cv = GroupKFold(n_splits=4)
    forest_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        {
            "n_estimators": [200, 400],
            "max_depth": [None, 10, 20],
            "min_samples_leaf": [1, 2],
        },
        scoring="f1_macro",
        cv=group_cv,
        n_jobs=-1,
    )
    forest_search.fit(x_train, y_train, groups=train["actor"])

    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
        "Random Forest": forest_search.best_estimator_,
        "Gaussian Naive Bayes": GaussianNB(),
    }

    results = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        prediction = model.predict(x_calibration)
        probabilities = model.predict_proba(x_calibration)
        results.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_calibration, prediction),
                "macro_f1": f1_score(y_calibration, prediction, average="macro"),
                "log_loss": log_loss(
                    y_calibration, probabilities, labels=model.classes_
                ),
                "brier_score": multiclass_brier_score(
                    y_calibration, probabilities, model.classes_
                ),
                "ece": expected_calibration_error(
                    y_calibration, probabilities, model.classes_
                ),
            }
        )

    metrics = pd.DataFrame(results).round(4)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT, index=False)

    print(
        f"Обучение: {len(train)}, калибровка: {len(calibration)}, "
        f"финальный тест: {len(test)} файлов"
    )
    print("Финальный тест на актёрах 17–20 не запускался")
    print(f"Лучшие параметры Random Forest: {forest_search.best_params_}")
    print(metrics.to_string(index=False))
    print(f"\nСоздан файл: {OUTPUT}")


if __name__ == "__main__":
    main()
