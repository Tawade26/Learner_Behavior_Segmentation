"""Read and summarize the saved analysis for the local dashboard."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_FILES = (
    "student_features.csv",
    "data_quality.json",
    "clustered_students.csv",
    "cluster_profiles.csv",
    "k_evaluation.csv",
    "clustering_run.json",
)


class ArtifactError(ValueError):
    """Saved analysis outputs are missing or do not match each other."""


def artifact_signature(folder: Path) -> tuple[tuple[str, int, int], ...]:
    """Change Streamlit's cache key when any analysis output changes."""
    signature = []
    for name in REQUIRED_FILES:
        path = folder / name
        if not path.is_file():
            raise ArtifactError(f"Missing analysis output: {path}")
        stat = path.stat()
        signature.append((name, stat.st_mtime_ns, stat.st_size))
    return tuple(signature)


def load_dashboard_data(folder: Path, label_config: Path | None = None):
    artifact_signature(folder)
    quality = json.loads((folder / "data_quality.json").read_text(encoding="utf-8"))
    run = json.loads((folder / "clustering_run.json").read_text(encoding="utf-8"))
    if quality.get("status") != "ok":
        raise ArtifactError("The data quality report is not OK. Rerun data preparation.")
    feature_hash = hashlib.sha256((folder / "student_features.csv").read_bytes()).hexdigest()
    if feature_hash != run.get("input_features_sha256"):
        raise ArtifactError("The clustering results do not match the current feature table. Rerun K-Means.")

    students = pd.read_csv(
        folder / "clustered_students.csv",
        dtype={"code_module": "string", "code_presentation": "string", "id_student": "string", "cluster": int},
    )
    profiles = pd.read_csv(folder / "cluster_profiles.csv")
    evaluation = pd.read_csv(folder / "k_evaluation.csv")
    expected_count = run.get("student_course_records")
    if len(students) != expected_count or len(students) != quality.get("student_course_records"):
        raise ArtifactError("Student record counts do not match the saved run metadata.")
    if students.duplicated(["code_module", "code_presentation", "id_student"]).any():
        raise ArtifactError("Cluster assignments have duplicate student-course keys.")
    counts = students["cluster"].value_counts().to_dict()
    profile_counts = dict(zip(profiles["cluster"], profiles["student_course_records"]))
    if counts != profile_counts or len(counts) != run.get("selected_k"):
        raise ArtifactError("Cluster assignments and profiles have different group counts.")
    if run.get("selected_k") not in set(evaluation["k"]):
        raise ArtifactError("Selected K is absent from the K evaluation table.")

    labels = {int(cluster): f"Cluster {cluster}" for cluster in counts}
    if label_config and label_config.is_file():
        configured = json.loads(label_config.read_text(encoding="utf-8"))
        if (configured.get("input_features_sha256") == feature_hash
                and configured.get("selected_k") == run.get("selected_k")):
            labels.update({int(k): str(v) for k, v in configured.get("labels", {}).items() if int(k) in counts})
    return students, profiles, evaluation, run, labels


def summarize_selection(students: pd.DataFrame) -> pd.DataFrame:
    """Describe only visible records; never refit clusters after filtering."""
    if students.empty:
        return pd.DataFrame()
    values = students.copy()
    opportunities = values["context_course_assessment_count"].replace(0, np.nan)
    timed = values["assessment_timed_count"].replace(0, np.nan)
    values["participation_rate"] = values["assessment_submitted_count"] / opportunities
    values["late_rate"] = values["assessment_late_count"] / timed
    values["has_score"] = values["assessment_mean_score"].notna().astype(float)
    summary = values.groupby("cluster", sort=True).agg(
        records=("id_student", "size"),
        mean_clicks=("vle_clicks_during_course", "mean"),
        mean_active_days=("vle_active_days_during_course", "mean"),
        mean_participation=("participation_rate", "mean"),
        mean_score=("assessment_mean_score", "mean"),
        mean_late_rate=("late_rate", "mean"),
        score_coverage=("has_score", "mean"),
    ).reset_index()
    summary["share"] = summary["records"] / len(values)
    return summary
