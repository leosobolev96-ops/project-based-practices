"""
Журнал экспериментов (Задача 7).

Добавляет запись о прогоне эксперимента в reports/experiment_log.csv --
дата, название скрипта, git-коммит, заметка и метрики. Так со временем
накапливается история: что менялось и как это повлияло на результат.

Использование из другого скрипта в Python:
    from emotion_calibration.log_experiment import log_experiment
    log_experiment("calibrate", {"log_loss": 1.50, "brier_score": 0.70})

Использование из терминала:
    uv run ravdess-log-experiment --script calibrate --note "первый прогон" --log_loss 1.50 --brier_score 0.70

Положи этот файл в src/emotion_calibration/log_experiment.py
"""

import argparse
import csv
import subprocess
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path("reports/experiment_log.csv")


def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def log_experiment(script_name: str, metrics: dict, note: str = "") -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": script_name,
        "git_commit": get_git_commit(),
        "note": note,
        **metrics,
    }

    file_exists = LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"Записано в {LOG_PATH}: {row}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", required=True)
    parser.add_argument("--note", default="")
    parser.add_argument("--log_loss", type=float, default=None)
    parser.add_argument("--brier_score", type=float, default=None)
    parser.add_argument("--accuracy", type=float, default=None)
    args = parser.parse_args()

    metrics = {
        k: v
        for k, v in {
            "log_loss": args.log_loss,
            "brier_score": args.brier_score,
            "accuracy": args.accuracy,
        }.items()
        if v is not None
    }
    log_experiment(args.script, metrics, args.note)


if __name__ == "__main__":
    main()
