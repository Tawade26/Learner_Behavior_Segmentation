"""Evaluate K-Means and cluster the prepared OULAD student-course records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


MODEL_FEATURES = (
    "log1p_vle_clicks_per_week",
    "vle_active_day_rate",
    "vle_activity_breadth_rate",
    "assessment_participation_rate",
    "assessment_mean_score_imputed",
    "assessment_late_rate_imputed",
)
PROFILE_FIELDS = (
    "vle_clicks_during_course",
    "vle_active_days_during_course",
    "vle_activity_type_count_during_course",
    "assessment_submitted_count",
    "assessment_mean_score",
    "assessment_late_count",
)


class ClusteringError(ValueError):
    """The prepared data or clustering configuration is invalid."""


def read_prepared_data(features_path: Path):
    quality_path = features_path.with_name("data_quality.json")
    if not quality_path.is_file():
        raise ClusteringError(f"Quality report is missing: {quality_path}")
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    if quality.get("status") != "ok":
        raise ClusteringError(f"Quality report is not OK: {quality_path}")
    with features_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "code_module", "code_presentation", "id_student",
            "context_course_length_days", "context_course_assessment_count",
            "context_course_activity_type_count", "vle_clicks_during_course",
            "vle_active_days_during_course", "vle_activity_type_count_during_course",
            "assessment_submitted_count", "assessment_mean_score",
            "assessment_timed_count", "assessment_late_count",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ClusteringError(f"Feature table is missing columns: {', '.join(sorted(missing))}")
        records = list(reader)
        fieldnames = reader.fieldnames or []
    if len(records) != quality.get("student_course_records"):
        raise ClusteringError("Feature table row count does not match the quality report")
    if len(records) < 3:
        raise ClusteringError("At least three student-course records are needed")
    keys = [(r["code_module"], r["code_presentation"], r["id_student"]) for r in records]
    if len(set(keys)) != len(keys):
        raise ClusteringError("Feature table contains duplicate student-course keys")
    return records, fieldnames, quality


def value(record: dict[str, str], name: str, row_number: int) -> float:
    try:
        result = float(record[name])
    except (TypeError, ValueError) as exc:
        raise ClusteringError(f"Feature row {row_number} has invalid {name}: {record[name]!r}") from exc
    if not np.isfinite(result):
        raise ClusteringError(f"Feature row {row_number} has nonfinite {name}")
    return result


def model_matrix(records: list[dict[str, str]]):
    numerical = []
    scores = []
    late_rates = []
    for index, record in enumerate(records, start=2):
        days = value(record, "context_course_length_days", index)
        assessment_count = value(record, "context_course_assessment_count", index)
        available_types = value(record, "context_course_activity_type_count", index)
        clicks = value(record, "vle_clicks_during_course", index)
        active_days = value(record, "vle_active_days_during_course", index)
        types_used = value(record, "vle_activity_type_count_during_course", index)
        submitted = value(record, "assessment_submitted_count", index)
        timed = value(record, "assessment_timed_count", index)
        late = value(record, "assessment_late_count", index)
        if days <= 0 or assessment_count < 0 or available_types < 0:
            raise ClusteringError(f"Feature row {index} has invalid course context")
        if min(clicks, active_days, types_used, submitted, timed, late) < 0:
            raise ClusteringError(f"Feature row {index} has a negative count")
        if active_days > days or types_used > available_types or late > timed:
            raise ClusteringError(f"Feature row {index} has inconsistent activity or timing counts")
        if assessment_count and submitted > assessment_count:
            raise ClusteringError(f"Feature row {index} has too many assessment submissions")
        if not assessment_count and submitted:
            raise ClusteringError(f"Feature row {index} has submissions but no assessments")
        numerical.append([
            np.log1p(clicks * 7.0 / days),
            active_days / days,
            types_used / available_types if available_types else 0.0,
            submitted / assessment_count if assessment_count else 0.0,
        ])
        score_text = record["assessment_mean_score"].strip()
        score = value(record, "assessment_mean_score", index) if score_text else np.nan
        if np.isfinite(score) and not 0 <= score <= 100:
            raise ClusteringError(f"Feature row {index} has score outside 0-100")
        scores.append(score)
        late_rates.append(late / timed if timed else np.nan)

    score_array = np.asarray(scores, dtype=float)
    late_array = np.asarray(late_rates, dtype=float)
    score_median = float(np.nanmedian(score_array)) if np.isfinite(score_array).any() else 0.0
    late_median = float(np.nanmedian(late_array)) if np.isfinite(late_array).any() else 0.0
    matrix = np.column_stack((
        np.asarray(numerical, dtype=float),
        np.where(np.isfinite(score_array), score_array, score_median),
        np.where(np.isfinite(late_array), late_array, late_median),
    ))
    if not np.isfinite(matrix).all():
        raise ClusteringError("Model features contain nonfinite values")
    return matrix, {"score_median": score_median, "late_rate_median": late_median}


def evaluate_k(matrix: np.ndarray, k_values: range, sample_size: int, seed: int):
    scaled = StandardScaler().fit_transform(matrix)
    sample_size = min(sample_size, len(scaled))
    rng = np.random.default_rng(seed)
    sample_indices = np.sort(rng.choice(len(scaled), size=sample_size, replace=False))
    results = []
    for k in k_values:
        if k >= len(scaled):
            raise ClusteringError(f"K={k} needs more than {k} records")
        model = KMeans(n_clusters=k, random_state=seed, n_init=20, algorithm="lloyd")
        labels = model.fit_predict(scaled)
        counts = np.bincount(labels, minlength=k)
        sample_labels = labels[sample_indices]
        silhouette = (
            float(silhouette_score(scaled[sample_indices], sample_labels))
            if 1 < len(np.unique(sample_labels)) < len(sample_indices)
            else None
        )
        results.append({
            "k": k,
            "inertia": float(model.inertia_),
            "silhouette_sample": silhouette,
            "smallest_cluster_records": int(counts.min()),
            "smallest_cluster_share": float(counts.min() / len(labels)),
            "largest_cluster_records": int(counts.max()),
        })
    return scaled, results, sample_size


def select_k(results: list[dict], requested_k: int | None) -> tuple[int, str]:
    values = {result["k"] for result in results}
    if requested_k is not None:
        if requested_k not in values:
            raise ClusteringError(f"Requested K={requested_k} is outside evaluated values {sorted(values)}")
        return requested_k, "explicit --k selection"
    eligible = [r for r in results if r["silhouette_sample"] is not None and r["smallest_cluster_share"] >= 0.03]
    if not eligible:
        eligible = [r for r in results if r["silhouette_sample"] is not None]
        selection_rule = "highest sampled silhouette; no candidate met the 3% smallest-cluster threshold"
    else:
        selection_rule = "highest sampled silhouette with at least 3% in every cluster"
    if not eligible:
        raise ClusteringError("No candidate K has a valid silhouette score")
    chosen = max(eligible, key=lambda r: (r["silhouette_sample"], -r["k"]))
    return chosen["k"], selection_rule


def write_csv(path: Path, fieldnames: list[str], records):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_profiles(path: Path, records: list[dict], labels: np.ndarray, matrix: np.ndarray):
    profiles = []
    for cluster_id in sorted(set(labels)):
        indices = np.flatnonzero(labels == cluster_id)
        profile = {
            "cluster": int(cluster_id),
            "student_course_records": int(len(indices)),
            "share_of_records": f"{len(indices) / len(labels):.4f}",
            "share_with_score": f"{sum(bool(records[i]['assessment_mean_score'].strip()) for i in indices) / len(indices):.4f}",
            "share_with_timing": f"{sum(int(records[i]['assessment_timed_count']) > 0 for i in indices) / len(indices):.4f}",
        }
        for feature_index, name in enumerate(MODEL_FEATURES):
            profile[f"mean_{name}"] = f"{matrix[indices, feature_index].mean():.4f}"
        for name in PROFILE_FIELDS:
            observed = [float(records[i][name]) for i in indices if records[i][name].strip()]
            profile[f"mean_{name}"] = f"{np.mean(observed):.4f}" if observed else ""
            profile[f"observed_{name}_count"] = len(observed)
        profiles.append(profile)
    write_csv(path, list(profiles[0]), profiles)
    return profiles


def run(features_path: Path, output_dir: Path, k_min: int = 2, k_max: int = 8,
        sample_size: int = 2000, seed: int = 42, requested_k: int | None = None):
    if k_min < 2 or k_max < k_min or sample_size < 3:
        raise ClusteringError("Require 2 <= k-min <= k-max and sample-size >= 3")
    records, input_columns, _ = read_prepared_data(features_path)
    matrix, imputation = model_matrix(records)
    scaled, evaluation, actual_sample_size = evaluate_k(matrix, range(k_min, k_max + 1), sample_size, seed)
    selected_k, selection_rule = select_k(evaluation, requested_k)
    final_model = KMeans(n_clusters=selected_k, random_state=seed, n_init=20, algorithm="lloyd")
    labels = final_model.fit_predict(scaled)
    output_dir.mkdir(parents=True, exist_ok=True)

    assignments_path = output_dir / "clustered_students.csv"
    write_csv(assignments_path, input_columns + ["cluster"],
              ({**record, "cluster": int(label)} for record, label in zip(records, labels)))
    evaluation_path = output_dir / "k_evaluation.csv"
    write_csv(evaluation_path, list(evaluation[0]), evaluation)
    profiles_path = output_dir / "cluster_profiles.csv"
    profiles = write_profiles(profiles_path, records, labels, matrix)
    metadata_path = output_dir / "clustering_run.json"
    metadata = {
        "input_features_sha256": hashlib.sha256(features_path.read_bytes()).hexdigest(),
        "student_course_records": len(records),
        "model_features": list(MODEL_FEATURES),
        "input_excluded_from_model": "Identifiers and context columns, including final_result, are excluded",
        "imputation": imputation,
        "standardization": "StandardScaler fitted to all student-course records",
        "k_candidates": list(range(k_min, k_max + 1)),
        "silhouette_sample_size": actual_sample_size,
        "random_seed": seed,
        "kmeans_n_init": 20,
        "selected_k": selected_k,
        "selection_rule": selection_rule,
        "scikit_learn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "cluster_sizes": {str(p["cluster"]): p["student_course_records"] for p in profiles},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return assignments_path, evaluation_path, profiles_path, metadata_path


def main():
    repo_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=repo_dir / "generated" / "student_features.csv")
    parser.add_argument("--output-dir", type=Path, default=repo_dir / "generated")
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=None, help="Override automatic K selection after reviewing profiles")
    args = parser.parse_args()
    try:
        outputs = run(args.features, args.output_dir, args.k_min, args.k_max,
                      args.sample_size, args.seed, args.k)
    except (ClusteringError, OSError) as exc:
        parser.exit(1, f"Clustering failed: {exc}\n")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
