from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import app  # noqa: E402


class BackendValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.client.__enter__()
        feature_names_path = PROJECT_ROOT / "model" / "serialized" / "feature_names.json"
        with feature_names_path.open("r", encoding="utf-8") as handle:
            feature_names = json.load(handle)["raw_feature_names"]
        dataset = pd.read_csv(PROJECT_ROOT / "dataset" / "dataset.csv")
        row = dataset.iloc[0].to_dict()
        cls.valid_payload = {"features": {feature: row[feature] for feature in feature_names}}

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "running")

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertIn(response.json()["status"], {"healthy", "degraded"})

    def test_models_status_endpoint(self):
        response = self.client.get("/models/status")
        self.assertEqual(response.status_code, 200)
        for target in ("CAD", "LAD", "LCX", "RCA"):
            self.assertIn(target, response.json())
            self.assertIn("loaded", response.json()[target])
            self.assertIn("threshold", response.json()[target])

    def test_metrics_endpoint_exposes_threshold_rationale(self):
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        holdout_csv = pd.read_csv(PROJECT_ROOT / "metrics" / "classification_metrics.csv").set_index("Target")
        cv_csv = pd.read_csv(PROJECT_ROOT / "metrics" / "repeated_cv_summary.csv").set_index(["Target", "Metric"])
        for target in ("CAD", "LAD", "LCX", "RCA"):
            self.assertIn(target, response.json()["perTarget"])
            target_metrics = response.json()["perTarget"][target]
            self.assertIn("threshold", target_metrics)
            self.assertEqual(target_metrics["accuracy"], holdout_csv.loc[target, "Accuracy"])
            self.assertEqual(target_metrics["rocAuc"], holdout_csv.loc[target, "ROC_AUC"])
            for metric, api_name in (("Accuracy", "accuracy"), ("ROC_AUC", "rocAuc")):
                cv_metric = target_metrics["repeatedCv"][api_name]
                source = cv_csv.loc[(target, metric)]
                self.assertEqual(cv_metric["mean"], source["Mean"])
                self.assertEqual(cv_metric["sd"], source["SD"])
                self.assertEqual(cv_metric["ci95Lower"], source["CI_95_Lower"])
                self.assertEqual(cv_metric["ci95Upper"], source["CI_95_Upper"])
            rationale = target_metrics["thresholdRationale"]
            self.assertEqual(rationale["objective"], "maximize F1")
            self.assertIn("recall_floor", rationale)
            self.assertIn("Precision", rationale["production_operating_point"])
            self.assertIn("Recall", rationale["production_operating_point"])

    def test_predict_endpoint(self):
        response = self.client.post("/predict", json=self.valid_payload)
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertTrue(body["success"])
        for target in ("CAD", "LAD", "LCX", "RCA"):
            self.assertIn(target, body["predictions"])
            prediction = body["predictions"][target]
            self.assertIn("probability", prediction)
            self.assertIn("threshold", prediction)
            self.assertIn("prediction", prediction)
            self.assertIn("risk_level", prediction)
            self.assertGreaterEqual(float(prediction["probability"]), 0.0)
            self.assertLessEqual(float(prediction["probability"]), 1.0)
            self.assertGreaterEqual(float(prediction["threshold"]), 0.0)
            self.assertLessEqual(float(prediction["threshold"]), 1.0)
            self.assertIn(int(prediction["prediction"]), {0, 1})
            self.assertIn(target, body["explanations"])

    def test_invalid_input_is_rejected(self):
        invalid_payload = {"features": {"Age": "not-a-number", "Weight": 90}}
        response = self.client.post("/predict", json=invalid_payload)
        self.assertIn(response.status_code, {400, 422})

    def test_target_columns_are_rejected(self):
        bad_payload = {"features": {**self.valid_payload["features"], "CAD": 1}}
        response = self.client.post("/predict", json=bad_payload)
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
