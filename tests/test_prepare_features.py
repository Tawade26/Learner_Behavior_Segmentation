import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_features import DataError, prepare


def write_csv(folder, name, header, records):
    with (folder / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(records)


class PrepareFeaturesTests(unittest.TestCase):
    def make_data(self, folder):
        write_csv(folder, "courses.csv", ["code_module", "code_presentation", "module_presentation_length"],
                  [["AAA", "2013J", 30]])
        write_csv(folder, "studentInfo.csv",
                  ["code_module", "code_presentation", "id_student", "final_result"],
                  [["AAA", "2013J", 1, "Pass"], ["AAA", "2013J", 2, "Withdrawn"]])
        write_csv(folder, "studentRegistration.csv",
                  ["code_module", "code_presentation", "id_student", "date_registration", "date_unregistration"],
                  [["AAA", "2013J", 1, -4, ""], ["AAA", "2013J", 2, -2, 8]])
        write_csv(folder, "vle.csv", ["id_site", "code_module", "code_presentation", "activity_type"],
                  [[10, "AAA", "2013J", "resource"]])
        write_csv(folder, "studentVle.csv",
                  ["code_module", "code_presentation", "id_student", "id_site", "date", "sum_click"],
                  [["AAA", "2013J", 1, 10, -1, 2], ["AAA", "2013J", 1, 10, 3, 4],
                   ["AAA", "2013J", 1, 10, 3, 1], ["AAA", "2013J", 1, 10, 30, 3]])
        write_csv(folder, "assessments.csv",
                  ["code_module", "code_presentation", "id_assessment", "assessment_type", "date"],
                  [["AAA", "2013J", 100, "TMA", 5], ["AAA", "2013J", 101, "Exam", ""],
                   ["AAA", "2013J", 102, "TMA", 10]])
        write_csv(folder, "studentAssessment.csv",
                  ["id_assessment", "id_student", "date_submitted", "is_banked", "score"],
                  [[100, 1, 4, 0, 80], [101, 1, 25, 0, 60], [102, 1, 12, 0, ""]])

    def test_retains_inactive_student_and_handles_unknown_timing(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            self.make_data(root)
            features_path, quality_path = prepare(root, root / "out")
            with features_path.open(encoding="utf-8", newline="") as handle:
                records = {row["id_student"]: row for row in csv.DictReader(handle)}
            quality = json.loads(quality_path.read_text(encoding="utf-8"))
            self.assertEqual(len(records), 2)
            self.assertEqual(records["1"]["vle_total_clicks"], "10")
            self.assertEqual(records["1"]["vle_active_days"], "3")
            self.assertEqual(records["1"]["vle_clicks_during_course"], "5")
            self.assertEqual(records["1"]["vle_clicks_after_end"], "3")
            self.assertEqual(records["1"]["context_course_assessment_count"], "3")
            self.assertEqual(records["1"]["context_course_activity_type_count"], "1")
            self.assertEqual(records["1"]["vle_activity_type_count_during_course"], "1")
            self.assertEqual(records["1"]["assessment_mean_score"], "70.0000")
            self.assertEqual(records["1"]["assessment_timed_count"], "2")
            self.assertEqual(records["1"]["assessment_mean_days_from_due"], "0.5000")
            self.assertEqual(records["2"]["vle_total_clicks"], "0")
            self.assertEqual(records["2"]["assessment_mean_score"], "")
            self.assertEqual(quality["assessment_submissions_without_timing"], 1)
            self.assertEqual(quality["students_without_vle_activity"], 1)
            self.assertEqual(quality["status"], "ok")

    def test_unmatched_join_blocks_feature_output(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            self.make_data(root)
            with (root / "studentVle.csv").open("a", encoding="utf-8", newline="") as handle:
                csv.writer(handle).writerow(["AAA", "2013J", 2, 999, 1, 2])
            with self.assertRaises(DataError):
                prepare(root, root / "out")
            quality = json.loads((root / "out" / "data_quality.json").read_text(encoding="utf-8"))
            self.assertEqual(quality["unmatched_rows"]["studentVle_to_vle"], 1)
            self.assertEqual(quality["status"], "failed")
            self.assertFalse((root / "out" / "student_features.csv").exists())


if __name__ == "__main__":
    unittest.main()
