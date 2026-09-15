"""
CLI для инференса одного WAV-файла (Задача 8).

Берёт один WAV-файл, извлекает те же 62 признака, что использовались при
обучении, и выдаёт предсказанную эмоцию вместе с откалиброванными
вероятностями по каждому классу.

Перед первым запуском нужно один раз обучить и сохранить модель:
    uv run ravdess-train-final

Затем для любого WAV-файла:
    uv run ravdess-predict --wav path/to/file.wav

Положи этот файл в src/emotion_calibration/predict_single.py
"""

import argparse
import json
from pathlib import Path

import joblib
import librosa
import numpy as np

MODELS_DIR = Path("models")
SR = 16000


def extract_baseline_features(y, sr) -> dict:
    """Тот же набор из 62 признаков, что и в features.csv."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    feats = {}
    for i in range(13):
        feats[f"mfcc_{i + 1:02d}_mean"] = float(np.mean(mfcc[i]))
        feats[f"mfcc_{i + 1:02d}_std"] = float(np.std(mfcc[i]))
    for i in range(13):
        feats[f"delta_{i + 1:02d}_std"] = float(np.std(delta[i]))
    for i in range(13):
        feats[f"delta2_{i + 1:02d}_std"] = float(np.std(delta2[i]))

    rms = librosa.feature.rms(y=y)[0]
    feats["rms_mean"] = float(np.mean(rms))
    feats["rms_std"] = float(np.std(rms))

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    feats["zero_crossing_rate_mean"] = float(np.mean(zcr))
    feats["zero_crossing_rate_std"] = float(np.std(zcr))

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    feats["spectral_centroid_mean"] = float(np.mean(centroid))
    feats["spectral_centroid_std"] = float(np.std(centroid))

    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    feats["spectral_bandwidth_mean"] = float(np.mean(bandwidth))
    feats["spectral_bandwidth_std"] = float(np.std(bandwidth))

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    feats["spectral_rolloff_mean"] = float(np.mean(rolloff))
    feats["spectral_rolloff_std"] = float(np.std(rolloff))

    return feats


def main() -> None:
    parser = argparse.ArgumentParser(description="Определить эмоцию по одному WAV-файлу")
    parser.add_argument("--wav", required=True, help="Путь к WAV-файлу")
    args = parser.parse_args()

    wav_path = Path(args.wav)
    if not wav_path.exists():
        print(f"Файл не найден: {wav_path}")
        return

    model_path = MODELS_DIR / "calibrated_model.joblib"
    meta_path = MODELS_DIR / "model_meta.json"
    if not model_path.exists() or not meta_path.exists():
        print("Модель не найдена. Сначала выполни: uv run ravdess-train-final")
        return

    calibrated_model = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_columns = meta["feature_columns"]

    y, sr = librosa.load(str(wav_path), sr=SR)
    feats = extract_baseline_features(y, sr)

    X = np.array([[feats[col] for col in feature_columns]])

    proba = calibrated_model.predict_proba(X)[0]
    pred = calibrated_model.classes_[proba.argmax()]

    print(f"\nФайл: {wav_path.name}")
    print(f"Предсказанная эмоция: {pred}\n")
    print("Вероятности (откалиброванные):")
    for idx in np.argsort(proba)[::-1]:
        cls = calibrated_model.classes_[idx]
        print(f"  {cls:>10}: {proba[idx] * 100:5.1f}%")


if __name__ == "__main__":
    main()
