"""
Загрузка config.yaml -- единого места со всеми настройками проекта.

Использование в других скриптах:
    from emotion_calibration.config import load_config
    config = load_config()
    train_actors = config["actors"]["train"]

Положи этот файл в src/emotion_calibration/config.py
"""

from pathlib import Path

import yaml

CONFIG_PATH = Path("config.yaml")


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
