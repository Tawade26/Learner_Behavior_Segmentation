"""Prepare one OULAD feature row per student, module, and presentation.

This Phase 1 pipeline uses only the Python standard library. The large
studentVle.csv file is read one row at a time rather than loaded into memory.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path


REQUIRED_COLUMNS = {
    "courses.csv": {"code_module", "code_presentation", "module_presentation_length"},
    "studentInfo.csv": {"code_module", "code_presentation", "id_student", "final_result"},
    "studentRegistration.csv": {
        "code_module", "code_presentation", "id_student",
        "date_registration", "date_unregistration",
    },
    "vle.csv": {"id_site", "code_module", "code_presentation", "activity_type"},
    "studentVle.csv": {
        "code_module", "code_presentation", "id_student", "id_site", "date", "sum_click",
    },
    "assessments.csv": {
        "code_module", "code_presentation", "id_assessment", "assessment_type", "date",
    },
    "studentAssessment.csv": {
        "id_assessment", "id_student", "date_submitted", "is_banked", "score",
    },
}


class DataError(ValueError):
    """Input data cannot be safely prepared."""


@dataclass(slots=True)
class StudentFeatures:
    final_result: str
    course_length_days: int
    course_assessment_count: int = 0
    course_activity_type_count: int = 0
    date_registration: str = ""
    date_unregistration: str = ""
    vle_total_clicks: int = 0
    vle_clicks_before_start: int = 0
    vle_clicks_during_course: int = 0
    vle_clicks_after_end: int = 0
    active_days: set[int] = field(default_factory=set)
    active_days_during_course: set[int] = field(default_factory=set)
    activity_types: set[str] = field(default_factory=set)
    activity_types_during_course: set[str] = field(default_factory=set)
    clicks_by_type: Counter[str] = field(default_factory=Counter)
    assessment_submitted_count: int = 0
    assessment_banked_count: int = 0
    assessment_scored_count: int = 0
    assessment_score_sum: float = 0.0
    assessment_timed_count: int = 0
    assessment_ontime_count: int = 0
    assessment_late_count: int = 0
    assessment_days_from_due_sum: int = 0


def rows(data_dir: Path, filename: str):
    path = data_dir / filename
    if not path.is_file():
        raise DataError(f"Required file is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS[filename] - set(reader.fieldnames or [])
        if missing:
            raise DataError(f"{filename} is missing columns: {', '.join(sorted(missing))}")
        for line_number, row in enumerate(reader, start=2):
            yield line_number, row


def integer(value: str, filename: str, line: int, column: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise DataError(f"{filename}:{line}: invalid {column}: {value!r}") from exc


def enrollment_key(row: dict[str, str]) -> tuple[str, str, str]:
    return row["code_module"], row["code_presentation"], row["id_student"]


def course_key(row: dict[str, str]) -> tuple[str, str]:
    return row["code_module"], row["code_presentation"]


def load_reference_data(data_dir: Path, quality: dict):
    counts = quality["source_rows"]
    courses = {}
    for line, row in rows(data_dir, "courses.csv"):
        counts["courses.csv"] += 1
        key = course_key(row)
        if key in courses:
            raise DataError(f"courses.csv:{line}: duplicate course {key}")
        length = integer(row["module_presentation_length"], "courses.csv", line, "module_presentation_length")
        if length <= 0:
            raise DataError(f"courses.csv:{line}: course length must be positive")
        courses[key] = length

    students = {}
    for line, row in rows(data_dir, "studentInfo.csv"):
        counts["studentInfo.csv"] += 1
        key = enrollment_key(row)
        if key in students:
            raise DataError(f"studentInfo.csv:{line}: duplicate student-course record {key}")
        if key[:2] not in courses:
            raise DataError(f"studentInfo.csv:{line}: course not found for {key}")
        students[key] = StudentFeatures(row["final_result"], courses[key[:2]])

    registration_keys = set()
    for line, row in rows(data_dir, "studentRegistration.csv"):
        counts["studentRegistration.csv"] += 1
        key = enrollment_key(row)
        if key in registration_keys:
            raise DataError(f"studentRegistration.csv:{line}: duplicate registration {key}")
        registration_keys.add(key)
        student = students.get(key)
        if student is None:
            quality["unmatched_rows"]["studentRegistration_to_studentInfo"] += 1
            continue
        student.date_registration = row["date_registration"]
        student.date_unregistration = row["date_unregistration"]
    quality["missing_registration_count"] = len(students.keys() - registration_keys)

    sites = {}
    activity_types = set()
    course_activity_types = defaultdict(set)
    for line, row in rows(data_dir, "vle.csv"):
        counts["vle.csv"] += 1
        key = (*course_key(row), row["id_site"])
        if key in sites:
            raise DataError(f"vle.csv:{line}: duplicate VLE site {key}")
        sites[key] = row["activity_type"]
        activity_types.add(row["activity_type"])
        course_activity_types[course_key(row)].add(row["activity_type"])

    assessments = {}
    course_assessment_counts = Counter()
    for line, row in rows(data_dir, "assessments.csv"):
        counts["assessments.csv"] += 1
        assessment_id = row["id_assessment"]
        if assessment_id in assessments:
            raise DataError(f"assessments.csv:{line}: duplicate assessment {assessment_id}")
        deadline = row["date"].strip()
        assessments[assessment_id] = (
            *course_key(row),
            row["assessment_type"],
            integer(deadline, "assessments.csv", line, "date") if deadline else None,
        )
        course_assessment_counts[course_key(row)] += 1
        if not deadline:
            quality["assessments_without_deadline"] += 1
    for key, student in students.items():
        student.course_assessment_count = course_assessment_counts[key[:2]]
        student.course_activity_type_count = len(course_activity_types[key[:2]])
    return students, sites, assessments, sorted(activity_types)


def aggregate_assessments(data_dir: Path, students: dict, assessments: dict, quality: dict):
    seen = set()
    for line, row in rows(data_dir, "studentAssessment.csv"):
        quality["source_rows"]["studentAssessment.csv"] += 1
        assessment_id, student_id = row["id_assessment"], row["id_student"]
        submission_key = assessment_id, student_id
        if submission_key in seen:
            raise DataError(f"studentAssessment.csv:{line}: duplicate submission {submission_key}")
        seen.add(submission_key)
        assessment = assessments.get(assessment_id)
        if assessment is None:
            quality["unmatched_rows"]["studentAssessment_to_assessments"] += 1
            continue
        student = students.get((assessment[0], assessment[1], student_id))
        if student is None:
            quality["unmatched_rows"]["studentAssessment_to_studentInfo"] += 1
            continue
        banked = integer(row["is_banked"], "studentAssessment.csv", line, "is_banked")
        if banked not in (0, 1):
            raise DataError(f"studentAssessment.csv:{line}: is_banked must be 0 or 1")
        if banked:
            student.assessment_banked_count += 1
            continue
        student.assessment_submitted_count += 1
        score = row["score"].strip()
        if score:
            try:
                student.assessment_score_sum += float(score)
            except ValueError as exc:
                raise DataError(f"studentAssessment.csv:{line}: invalid score {score!r}") from exc
            student.assessment_scored_count += 1
        else:
            quality["assessment_submissions_without_score"] += 1
        submitted = row["date_submitted"].strip()
        if assessment[3] is None or not submitted:
            quality["assessment_submissions_without_timing"] += 1
            continue
        days_from_due = integer(submitted, "studentAssessment.csv", line, "date_submitted") - assessment[3]
        student.assessment_timed_count += 1
        student.assessment_days_from_due_sum += days_from_due
        if days_from_due <= 0:
            student.assessment_ontime_count += 1
        else:
            student.assessment_late_count += 1


def aggregate_vle(data_dir: Path, students: dict, sites: dict, quality: dict):
    for line, row in rows(data_dir, "studentVle.csv"):
        quality["source_rows"]["studentVle.csv"] += 1
        clicks = integer(row["sum_click"], "studentVle.csv", line, "sum_click")
        date = integer(row["date"], "studentVle.csv", line, "date")
        if clicks < 0:
            raise DataError(f"studentVle.csv:{line}: sum_click cannot be negative")
        student = students.get(enrollment_key(row))
        if student is None:
            quality["unmatched_rows"]["studentVle_to_studentInfo"] += 1
            continue
        activity_type = sites.get((*course_key(row), row["id_site"]))
        if activity_type is None:
            quality["unmatched_rows"]["studentVle_to_vle"] += 1
            continue
        student.vle_total_clicks += clicks
        student.active_days.add(date)
        student.activity_types.add(activity_type)
        student.clicks_by_type[activity_type] += clicks
        if date < 0:
            student.vle_clicks_before_start += clicks
            quality["vle_rows_before_course_start"] += 1
        elif date < student.course_length_days:
            student.vle_clicks_during_course += clicks
            student.active_days_during_course.add(date)
            student.activity_types_during_course.add(activity_type)
        else:
            student.vle_clicks_after_end += clicks
            quality["vle_rows_after_course_end"] += 1


def number_or_blank(total: float, count: int) -> str:
    return f"{total / count:.4f}" if count else ""


def write_outputs(output_dir: Path, students: dict, activity_types: list[str], quality: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    quality["student_course_records"] = len(students)
    quality["students_without_vle_activity"] = sum(not s.active_days for s in students.values())
    quality["students_without_nonbanked_assessments"] = sum(
        not s.assessment_submitted_count for s in students.values()
    )
    quality["feature_totals"] = {
        "vle_clicks": sum(s.vle_total_clicks for s in students.values()),
        "nonbanked_assessment_submissions": sum(s.assessment_submitted_count for s in students.values()),
        "banked_assessment_submissions": sum(s.assessment_banked_count for s in students.values()),
    }
    for student in students.values():
        if student.vle_total_clicks != (
            student.vle_clicks_before_start + student.vle_clicks_during_course + student.vle_clicks_after_end
        ):
            raise DataError("VLE click totals do not reconcile across course periods")
        if student.assessment_timed_count != student.assessment_ontime_count + student.assessment_late_count:
            raise DataError("Assessment timing counts do not reconcile")
    quality["activity_type_columns"] = {name: f"vle_clicks_{name}" for name in activity_types}
    problems = sum(quality["unmatched_rows"].values()) + quality["missing_registration_count"]
    quality["status"] = "failed" if problems else "ok"
    report = output_dir / "data_quality.json"
    report.write_text(json.dumps(quality, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if problems:
        raise DataError(f"Unmatched source records found; see {report}")

    fixed_columns = [
        "code_module", "code_presentation", "id_student", "context_final_result",
        "context_course_length_days", "context_course_assessment_count",
        "context_course_activity_type_count", "context_date_registration", "context_date_unregistration",
        "vle_total_clicks", "vle_active_days", "vle_active_days_during_course",
        "vle_clicks_before_start", "vle_clicks_during_course", "vle_clicks_after_end",
        "vle_activity_type_count", "vle_activity_type_count_during_course",
        "assessment_submitted_count", "assessment_banked_count", "assessment_scored_count",
        "assessment_mean_score", "assessment_timed_count", "assessment_ontime_count",
        "assessment_late_count", "assessment_mean_days_from_due",
    ]
    path = output_dir / "student_features.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fixed_columns + [f"vle_clicks_{name}" for name in activity_types])
        writer.writeheader()
        for (module, presentation, student_id), student in sorted(students.items()):
            record = {
                "code_module": module,
                "code_presentation": presentation,
                "id_student": student_id,
                "context_final_result": student.final_result,
                "context_course_length_days": student.course_length_days,
                "context_course_assessment_count": student.course_assessment_count,
                "context_course_activity_type_count": student.course_activity_type_count,
                "context_date_registration": student.date_registration,
                "context_date_unregistration": student.date_unregistration,
                "vle_total_clicks": student.vle_total_clicks,
                "vle_active_days": len(student.active_days),
                "vle_active_days_during_course": len(student.active_days_during_course),
                "vle_clicks_before_start": student.vle_clicks_before_start,
                "vle_clicks_during_course": student.vle_clicks_during_course,
                "vle_clicks_after_end": student.vle_clicks_after_end,
                "vle_activity_type_count": len(student.activity_types),
                "vle_activity_type_count_during_course": len(student.activity_types_during_course),
                "assessment_submitted_count": student.assessment_submitted_count,
                "assessment_banked_count": student.assessment_banked_count,
                "assessment_scored_count": student.assessment_scored_count,
                "assessment_mean_score": number_or_blank(student.assessment_score_sum, student.assessment_scored_count),
                "assessment_timed_count": student.assessment_timed_count,
                "assessment_ontime_count": student.assessment_ontime_count,
                "assessment_late_count": student.assessment_late_count,
                "assessment_mean_days_from_due": number_or_blank(
                    student.assessment_days_from_due_sum, student.assessment_timed_count
                ),
            }
            record.update({f"vle_clicks_{name}": student.clicks_by_type[name] for name in activity_types})
            writer.writerow(record)
    return path, report


def prepare(data_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    quality = {
        "source_rows": {filename: 0 for filename in REQUIRED_COLUMNS},
        "unmatched_rows": defaultdict(int),
        "assessments_without_deadline": 0,
        "assessment_submissions_without_score": 0,
        "assessment_submissions_without_timing": 0,
        "vle_rows_before_course_start": 0,
        "vle_rows_after_course_end": 0,
    }
    students, sites, assessments, activity_types = load_reference_data(data_dir, quality)
    aggregate_assessments(data_dir, students, assessments, quality)
    aggregate_vle(data_dir, students, sites, quality)
    return write_outputs(output_dir, students, activity_types, quality)


def main():
    repo_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=repo_dir.parent / "dataset")
    parser.add_argument("--output-dir", type=Path, default=repo_dir / "generated")
    args = parser.parse_args()
    try:
        features, report = prepare(args.data_dir, args.output_dir)
    except DataError as exc:
        parser.exit(1, f"Data preparation failed: {exc}\n")
    print(f"Features: {features}")
    print(f"Quality report: {report}")


if __name__ == "__main__":
    main()
