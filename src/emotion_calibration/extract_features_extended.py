"""
Расширение аудиопризнаков (Задача 2): chroma, spectral contrast, pitch,
паузы/ритм, а также MFCC с 20 и 40 коэффициентами.

Считает НОВЫЕ группы признаков отдельно (не трогая уже готовый
features.csv) и сохраняет их в отдельный файл. Дальше эти новые признаки
объединяются с базовыми в ablation_features.py.

Положи этот файл в src/emotion_calibration/extract_features_extended.py

ВНИМАНИЕ: обрабатывает все 1440 файлов и считает довольно тяжёлые вещи
(pitch, темп) — может занять от 15 до 40+ минут в зависимости от компьютера.
"""

import io
import zipfile
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

ZIP_PATH = Path("data/raw/Audio_Speech_Actors_01-24.zip")
OUTPUT_PATH = Path("data/processed/features_extended.csv")

SR = 16000


def extract_chroma(y, sr) -> dict:
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    feats = {}
    for i in range(chroma.shape[0]):
        feats[f"chroma_{i + 1:02d}_mean"] = float(np.mean(chroma[i]))
        feats[f"chroma_{i + 1:02d}_std"] = float(np.std(chroma[i]))
    return feats


def extract_spectral_contrast(y, sr) -> dict:
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    feats = {}
    for i in range(contrast.shape[0]):
        feats[f"contrast_{i + 1:02d}_mean"] = float(np.mean(contrast[i]))
        feats[f"contrast_{i + 1:02d}_std"] = float(np.std(contrast[i]))
    return feats


def extract_pitch(y, sr) -> dict:
    f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
    f0_voiced = f0[f0 > 0]
    if len(f0_voiced) == 0:
        return {"pitch_mean": 0.0, "pitch_std": 0.0, "voiced_ratio": 0.0}
    return {
        "pitch_mean": float(np.mean(f0_voiced)),
        "pitch_std": float(np.std(f0_voiced)),
        "voiced_ratio": float(len(f0_voiced) / len(f0)),
    }


def extract_rhythm(y, sr) -> dict:
    intervals = librosa.effects.split(y, top_db=30)
    total_duration = len(y) / sr
    speech_duration = sum((end - start) / sr for start, end in intervals)
    pause_duration = total_duration - speech_duration
    n_pauses = max(len(intervals) - 1, 0)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return {
        "pause_ratio": float(pause_duration / total_duration) if total_duration > 0 else 0.0,
        "n_pauses": int(n_pauses),
        "tempo": float(np.atleast_1d(tempo)[0]),
    }


def extract_mfcc_n(y, sr, n_mfcc: int) -> dict:
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    feats = {}
    for i in range(n_mfcc):
        feats[f"mfcc{n_mfcc}_{i + 1:02d}_mean"] = float(np.mean(mfcc[i]))
        feats[f"mfcc{n_mfcc}_{i + 1:02d}_std"] = float(np.std(mfcc[i]))
    return feats


def main() -> None:
    rows = []

    with zipfile.ZipFile(ZIP_PATH) as archive:
        wav_names = [n for n in archive.namelist() if n.endswith(".wav")]
        total = len(wav_names)

        for i, name in enumerate(wav_names, start=1):
            if i % 50 == 0 or i == total:
                print(f"Обработано {i}/{total}")

            raw = archive.read(name)
            y, sr = librosa.load(io.BytesIO(raw), sr=SR)

            row = {"path": name}
            row.update(extract_chroma(y, sr))
            row.update(extract_spectral_contrast(y, sr))
            row.update(extract_pitch(y, sr))
            row.update(extract_rhythm(y, sr))
            row.update(extract_mfcc_n(y, sr, 20))
            row.update(extract_mfcc_n(y, sr, 40))
            rows.append(row)

    extended_df = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    extended_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nГотово: {OUTPUT_PATH} ({len(extended_df)} строк, {len(extended_df.columns)} колонок)")


if __name__ == "__main__":
    main()
