import unittest

import numpy as np
import pandas as pd

from emotion_calibration.calibrate import (
    TemperatureScaler,
    evaluate,
    evaluate_by_emotion,
    split_actor_data,
)
from emotion_calibration.classify import (
    calibration_errors,
    expected_calibration_error,
    feature_columns,
    multiclass_brier_score,
)


class CalibrationTests(unittest.TestCase):
    def test_multiclass_brier_score_known_value(self):
        y = np.array(["a", "b"])
        probabilities = np.array([[0.8, 0.2], [0.25, 0.75]])
        self.assertAlmostEqual(multiclass_brier_score(y, probabilities, np.array(["a", "b"])), 0.1025)

    def test_perfect_predictions_have_zero_brier_and_ece(self):
        y = np.array(["a", "b"])
        p = np.array([[1.0, 0.0], [0.0, 1.0]])
        classes = np.array(["a", "b"])
        self.assertEqual(multiclass_brier_score(y, p, classes), 0.0)
        self.assertEqual(expected_calibration_error(y, p, classes), 0.0)

    def test_ece_ignores_empty_bins_without_losing_alignment(self):
        y = np.array(["a", "b"])
        probabilities = np.array([[0.55, 0.45], [0.95, 0.05]])
        classes = np.array(["a", "b"])

        errors = calibration_errors(y, probabilities, classes)

        self.assertAlmostEqual(errors["ece"], 0.7)
        self.assertAlmostEqual(expected_calibration_error(y, probabilities, classes), 0.7)

    def test_feature_columns_exclude_metadata(self):
        data = pd.DataFrame({"actor": [1], "emotion": ["calm"], "mfcc_01_mean": [0.2]})
        self.assertEqual(feature_columns(data), ["mfcc_01_mean"])

    def test_evaluate_contains_all_metrics(self):
        y = np.array(["a", "b"])
        p = np.array([[0.8, 0.2], [0.2, 0.8]])
        row = evaluate("demo", "none", y, np.array(["a", "b"]), p, np.array(["a", "b"]))
        self.assertEqual(set(row), {"model", "method", "accuracy", "macro_f1", "log_loss", "brier_score", "ece", "mce", "adaptive_ece"})

    def test_actor_split_has_no_overlap_and_correct_roles(self):
        data = pd.DataFrame({"actor": [1, 16, 17, 20, 21, 24], "emotion": ["a"] * 6})
        train, calibration, test = split_actor_data(data)
        self.assertEqual(set(train.actor), {1, 16})
        self.assertEqual(set(calibration.actor), {21, 24})
        self.assertEqual(set(test.actor), {17, 20})
        self.assertTrue(set(train.actor).isdisjoint(calibration.actor))
        self.assertTrue(set(train.actor).isdisjoint(test.actor))

    def test_temperature_scaler_returns_normalized_probabilities(self):
        probabilities = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])
        y = np.array(["a", "b", "a", "b"])
        scaler = TemperatureScaler().fit(probabilities, y, np.array(["a", "b"]))
        result = scaler.predict_proba(probabilities)
        self.assertTrue(np.allclose(result.sum(axis=1), 1.0))
        self.assertGreater(scaler.temperature_, 0.0)

    def test_brier_by_emotion_contains_every_class_and_method(self):
        y = np.array(["a", "b"])
        probabilities = np.array([[0.8, 0.2], [0.25, 0.75]])

        rows = evaluate_by_emotion(
            "demo", "none", y, probabilities, np.array(["a", "b"])
        )

        self.assertEqual({row["emotion"] for row in rows}, {"a", "b"})
        self.assertEqual({row["method"] for row in rows}, {"none"})


if __name__ == "__main__":
    unittest.main()
