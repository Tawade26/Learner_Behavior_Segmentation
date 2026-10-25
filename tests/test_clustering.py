import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.run_clustering import ClusteringError, model_matrix, run


HEADERS = [
    "code_module", "code_presentation", "id_student", "context_final_result",
    "context_course_length_days", "context_course_assessment_count",
    "context_course_activity_type_count", "vle_clicks_during_course",
    "vle_active_days_during_course", "vle_activity_type_count_during_course",
    "assessment_submitted_count", "assessment_mean_score",
    "assessment_timed_count", "assessment_late_count",
]


def make_records():
    records = []
    for index in range(12):
        high = index >= 6
        records.append({
            "code_module": "AAA", "code_presentation": "2013J", "id_student": str(index),
            "context_final_result": "Pass" if index % 2 else "Fail",
            "context_course_length_days": "100", "context_course_assessment_count": "4",
            "context_course_activity_type_count": "5",
            "vle_clicks_during_course": str((800 if high else 10) + index),
            "vle_active_days_during_course": str((65 if high else 3) + index % 3),
            "vle_activity_type_count_during_course": "4" if high else "1",
            "assessment_submitted_count": "4" if high else "0",
            "assessment_mean_score": str(70 + index) if high else "",
            "assessment_timed_count": "4" if high else "0",
            "assessment_late_count": "1" if high else "0",
        })
    return records


class ClusteringTests(unittest.TestCase):
    def test_context_outcome_cannot_change_model_inputs(self):
        records = make_records()
        original, imputation = model_matrix(records)
        for record in records:
            record["context_final_result"] = "Changed"
        changed, _ = model_matrix(records)
        np.testing.assert_array_equal(original, changed)
        self.assertEqual(original.shape, (12, 6))
        self.assertEqual(imputation["score_median"], 78.5)

    def test_end_to_end_outputs_and_quality_gate(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            features = root / "student_features.csv"
            with features.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=HEADERS)
                writer.writeheader()
                writer.writerows(make_records())
            quality = root / "data_quality.json"
            quality.write_text(json.dumps({"status": "failed", "student_course_records": 12}), encoding="utf-8")
            with self.assertRaises(ClusteringError):
                run(features, root / "out", k_min=2, k_max=3, sample_size=12)
            quality.write_text(json.dumps({"status": "ok", "student_course_records": 12}), encoding="utf-8")
            assignments, evaluation, profiles, metadata = run(
                features, root / "out", k_min=2, k_max=3, sample_size=12
            )
            with assignments.open(encoding="utf-8", newline="") as handle:
                assigned = list(csv.DictReader(handle))
            with evaluation.open(encoding="utf-8", newline="") as handle:
                evaluated = list(csv.DictReader(handle))
            with profiles.open(encoding="utf-8", newline="") as handle:
                profile_rows = list(csv.DictReader(handle))
            run_info = json.loads(metadata.read_text(encoding="utf-8"))
            self.assertEqual(len(assigned), 12)
            self.assertEqual(len(evaluated), 2)
            self.assertEqual(sum(int(r["student_course_records"]) for r in profile_rows), 12)
            self.assertEqual(run_info["selected_k"], len(profile_rows))
            self.assertEqual(len(run_info["model_features"]), 6)
            repeated = run(features, root / "again", k_min=2, k_max=3, sample_size=12)
            with repeated[0].open(encoding="utf-8", newline="") as handle:
                repeated_labels = [row["cluster"] for row in csv.DictReader(handle)]
            self.assertEqual([row["cluster"] for row in assigned], repeated_labels)


if __name__ == "__main__":
    unittest.main()
