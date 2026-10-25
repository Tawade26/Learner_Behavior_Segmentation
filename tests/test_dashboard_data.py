import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from dashboard_data import ArtifactError, load_dashboard_data, summarize_selection


def write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


class DashboardDataTests(unittest.TestCase):
    def make_outputs(self, root):
        features = root / "student_features.csv"
        features.write_text("sample feature table\n", encoding="utf-8")
        digest = hashlib.sha256(features.read_bytes()).hexdigest()
        (root / "data_quality.json").write_text(
            json.dumps({"status": "ok", "student_course_records": 4}), encoding="utf-8"
        )
        (root / "clustering_run.json").write_text(
            json.dumps({"input_features_sha256": digest, "student_course_records": 4, "selected_k": 2}),
            encoding="utf-8",
        )
        records = [
            {"code_module": "AAA", "code_presentation": "2013J", "id_student": "001", "cluster": 0,
             "context_course_assessment_count": 2, "assessment_submitted_count": 2,
             "assessment_timed_count": 2, "assessment_late_count": 1,
             "assessment_mean_score": 80, "vle_clicks_during_course": 40, "vle_active_days_during_course": 9},
            {"code_module": "AAA", "code_presentation": "2013J", "id_student": "002", "cluster": 0,
             "context_course_assessment_count": 2, "assessment_submitted_count": 1,
             "assessment_timed_count": 1, "assessment_late_count": 0,
             "assessment_mean_score": 60, "vle_clicks_during_course": 20, "vle_active_days_during_course": 4},
            {"code_module": "AAA", "code_presentation": "2013J", "id_student": "003", "cluster": 1,
             "context_course_assessment_count": 2, "assessment_submitted_count": 0,
             "assessment_timed_count": 0, "assessment_late_count": 0,
             "assessment_mean_score": "", "vle_clicks_during_course": 0, "vle_active_days_during_course": 0},
            {"code_module": "BBB", "code_presentation": "2014B", "id_student": "004", "cluster": 1,
             "context_course_assessment_count": 2, "assessment_submitted_count": 0,
             "assessment_timed_count": 0, "assessment_late_count": 0,
             "assessment_mean_score": "", "vle_clicks_during_course": 2, "vle_active_days_during_course": 1},
        ]
        write_csv(root / "clustered_students.csv", list(records[0]), records)
        write_csv(root / "cluster_profiles.csv", ["cluster", "student_course_records"],
                  [{"cluster": 0, "student_course_records": 2}, {"cluster": 1, "student_course_records": 2}])
        write_csv(root / "k_evaluation.csv", ["k", "inertia", "silhouette_sample"],
                  [{"k": 2, "inertia": 10, "silhouette_sample": 0.5}])
        labels = root / "labels.json"
        labels.write_text(json.dumps({"input_features_sha256": digest, "selected_k": 2,
                                      "labels": {"0": "Active", "1": "Quiet"}}), encoding="utf-8")
        return features, labels

    def test_load_filter_summary_and_stale_hash(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            features, labels_path = self.make_outputs(root)
            students, _, _, _, labels = load_dashboard_data(root, labels_path)
            self.assertEqual(labels, {0: "Active", 1: "Quiet"})
            self.assertEqual(students.loc[0, "id_student"], "001")
            selected = students[students["code_module"] == "AAA"]
            summary = summarize_selection(selected)
            self.assertEqual(summary["records"].tolist(), [2, 1])
            self.assertAlmostEqual(summary.loc[summary["cluster"] == 0, "mean_participation"].iloc[0], 0.75)
            self.assertAlmostEqual(summary["share"].sum(), 1.0)
            features.write_text("changed data\n", encoding="utf-8")
            with self.assertRaises(ArtifactError):
                load_dashboard_data(root, labels_path)


if __name__ == "__main__":
    unittest.main()
