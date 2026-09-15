"""
Аугментация тренировочных данных (Задача 3): шум, pitch shift, time stretch.

ВАЖНО: аугментация применяется ТОЛЬКО к актёрам 1-16 (train).
Актёры 21-24 (calibration) и 17-20 (test) не трогаются вообще -- иначе
модель будет "подсматривать" на изменённые версии тех же записей, которые
потом встретит на проверке, и результат станет нечестным.

Признаки считаются в том же формате, что и в features.csv (62 базовых
признака), чтобы потом можно было напрямую сравнивать модели.

Положи этот файл в src/emotion_calibration/augment_train.py

ВНИМАНИЕ: обрабатывает 960 файлов (актёры 1-16) x (оригинал + 3 аугментации)
= 3840 извлечений признаков. Может занять 20-40+ минут.
"""

import io
import zipfile
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

ZIP_PATH = Path("data/raw/Audio_Speech_Actors_01-24.zip")
FEATURES_PATH = Path("data/processed/features.csv")
OUTPUT_PATH = Path("data/processed/features_augmented_train.csv")

SR = 16000
TRAIN_ACTORS = list(range(1, 17))

METADATA_COLUMNS = [
    "path",
    "filename",
    "emotion",
    "intensity",
    "statement",
    "repetition",
    "actor",
    "actor_gender",
]


def extract_baseline_features(y, sr) -> dict:
    """Тот же набор из 62 признаков, что и в исходном features.csv."""
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


def augment_noise(y, sr):
    noise = np.random.normal(0, 0.005, size=y.shape)
    return (y + noise).astype(np.float32)


def augment_pitch_shift(y, sr):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=2)


def augment_time_stretch(y, sr):
    return librosa.effects.time_stretch(y, rate=1.15)


AUGMENTATIONS = {
    "noise": augment_noise,
    "pitch_shift": augment_pitch_shift,
    "time_stretch": augment_time_stretch,
}


def main() -> None:
    features_df = pd.read_csv(FEATURES_PATH)
    train_meta = features_df[features_df["actor"].isin(TRAIN_ACTORS)][METADATA_COLUMNS]

    rows = []
    with zipfile.ZipFile(ZIP_PATH) as archive:
        entries = train_meta.to_dict("records")
        total = len(entries)

        for i, entry in enumerate(entries, start=1):
            if i % 50 == 0 or i == total:
                print(f"Обработано {i}/{total} файлов (train, актёры 1-16)")

            raw = archive.read(entry["path"])
            y, sr = librosa.load(io.BytesIO(raw), sr=SR)

            # оригинал (без изменений)
            row = {**entry, "augmentation": "original"}
            row.update(extract_baseline_features(y, sr))
            rows.append(row)

            # аугментированные версии
            for aug_name, aug_fn in AUGMENTATIONS.items():
                y_aug = aug_fn(y, sr)
                row = {**entry, "augmentation": aug_name}
                row.update(extract_baseline_features(y_aug, sr))
                rows.append(row)

    result_df = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nГотово: {OUTPUT_PATH} ({len(result_df)} строк)")


if __name__ == "__main__":
    main()
