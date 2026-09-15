"""
Проверка версии и целостности датасета (Задача 7).

Считает SHA256 checksum архива RAVDESS и сравнивает с сохранённым в
config.yaml значением. Если checksum не совпадает -- значит используется
другой файл / повреждённая копия датасета, и результаты могут быть
невоспроизводимыми.

Запуск:
    uv run ravdess-verify-dataset            -- посчитать и показать checksum
    uv run ravdess-verify-dataset --save     -- посчитать и сохранить в config.yaml

Положи этот файл в src/emotion_calibration/verify_dataset.py
"""

import argparse
import hashlib
from pathlib import Path

import yaml

CONFIG_PATH = Path("config.yaml")


def compute_sha256(path: Path, chunk_size: int = 8192) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="Сохранить checksum в config.yaml")
    args = parser.parse_args()

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    zip_path = Path(config["dataset"]["zip_path"])
    if not zip_path.exists():
        print(f"Файл не найден: {zip_path}")
        return

    print("Считаю SHA256 checksum архива (может занять до минуты)...")
    checksum = compute_sha256(zip_path)
    print(f"SHA256: {checksum}")

    expected = config["dataset"].get("sha256")
    if expected:
        if expected == checksum:
            print("Checksum совпадает с config.yaml -- это тот же самый датасет.")
        else:
            print("ВНИМАНИЕ: checksum НЕ совпадает с config.yaml! Другой файл/версия датасета.")
    else:
        print("В config.yaml checksum ещё не сохранён.")

    if args.save:
        config["dataset"]["sha256"] = checksum
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
        print("Checksum сохранён в config.yaml")


if __name__ == "__main__":
    main()
