"""
Обучение и сохранение финальной модели для инференса (Задача 8).

Обучает Random Forest на train-актёрах, калибрует на calibration-актёрах
(так же, как в calibrate.py) и сохраняет готовую откалиброванную модель
на диск -- чтобы CLI для инференса не пересчитывал всё заново на каждый
запрос, а просто загружал уже готовую модель.

Запускать один раз (или заново -- если данные/код изменились).

Положи этот файл в src/emotion_calibration/train_final_model.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator

DATA_PATH = Path("data/processed/features.csv")
MODELS_DIR = Path("models")

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


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    feature_columns = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    classes = sorted(df["emotion"].unique())

    train_df = df[df["actor"].isin(TRAIN_ACTORS)]
    calib_df = df[df["actor"].isin(CALIBRATION_ACTORS)]

    X_train, y_train = train_df[feature_columns], train_df["emotion"]
    X_calib, y_calib = calib_df[feature_columns], calib_df["emotion"]

    print("Обучаю базовую модель...")
    base_model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    base_model.fit(X_train, y_train)

    print("Калибрую...")
    calibrated_model = CalibratedClassifierCV(FrozenEstimator(base_model), method="sigmoid")
    calibrated_model.fit(X_calib, y_calib)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated_model, MODELS_DIR / "calibrated_model.joblib")

    meta = {"feature_columns": feature_columns, "classes": classes}
    with open(MODELS_DIR / "model_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nГотово: модель сохранена в {MODELS_DIR}/calibrated_model.joblib")


if __name__ == "__main__":
    main()
