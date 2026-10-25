"""Local Streamlit dashboard for the saved learner behavior analysis."""

from pathlib import Path

import streamlit as st

from dashboard_data import ArtifactError, artifact_signature, load_dashboard_data, summarize_selection


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "generated"
LABELS = ROOT / "cluster_labels.json"

st.set_page_config(page_title="Learner behavior segmentation", layout="wide")


@st.cache_data(show_spinner=False)
def cached_analysis(signature, label_signature):
    return load_dashboard_data(OUTPUTS, LABELS)


try:
    students, profiles, evaluation, run, labels = cached_analysis(
        artifact_signature(OUTPUTS), LABELS.stat().st_mtime_ns if LABELS.is_file() else None
    )
except (ArtifactError, OSError, ValueError) as exc:
    st.error(str(exc))
    st.info("From the repository folder, run `python scripts/prepare_features.py` and then `python scripts/run_clustering.py --k 3`.")
    st.stop()


st.title("Learner behavior segmentation")
st.caption("OULAD · K-Means · descriptive analytics prototype")

with st.sidebar:
    st.header("Filter records")
    module = st.selectbox("Module", ["All modules", *sorted(students["code_module"].unique())])
    module_rows = students if module == "All modules" else students[students["code_module"] == module]
    presentation = st.selectbox(
        "Presentation", ["All presentations", *sorted(module_rows["code_presentation"].unique())]
    )
    selected_clusters = st.multiselect(
        "Clusters", options=sorted(labels), default=sorted(labels),
        format_func=lambda cluster: f"{cluster}: {labels[cluster]}",
    )
    student_search = st.text_input("Student ID contains", placeholder="Optional")

filtered = module_rows
if presentation != "All presentations":
    filtered = filtered[filtered["code_presentation"] == presentation]
filtered = filtered[filtered["cluster"].isin(selected_clusters)]
if student_search.strip():
    filtered = filtered[filtered["id_student"].str.contains(student_search.strip(), regex=False, na=False)]

st.caption("Filters change the displayed records. Clusters remain those fitted to the full dataset.")
count_col, cluster_col, k_col = st.columns(3)
count_col.metric("Student-course records shown", f"{len(filtered):,}")
cluster_col.metric("Clusters shown", filtered["cluster"].nunique())
k_col.metric("Selected K", run["selected_k"])

if filtered.empty:
    st.info("No records match these filters.")
    st.stop()

summary = summarize_selection(filtered)
summary["Cluster"] = summary["cluster"].map(lambda cluster: f"{cluster}: {labels[cluster]}")
summary["Cluster ID"] = summary["cluster"].map(lambda cluster: f"Cluster {cluster}")

st.subheader("Cluster distribution")
st.bar_chart(summary, x="Cluster ID", y="records", x_label="Learner group", y_label="Student-course records", sort=False)
st.dataframe(
    summary[["Cluster", "records", "share"]].rename(
        columns={"records": "Records", "share": "Share of filtered records"}
    ).round(3),
    hide_index=True,
    width="stretch",
)

st.subheader("Behavior comparison")
metric_labels = {
    "mean_clicks": "Average in-course clicks",
    "mean_active_days": "Average active days",
    "mean_participation": "Average assessment participation rate",
    "mean_score": "Average score where recorded",
    "mean_late_rate": "Average late-submission rate where timing is known",
}
metric = st.selectbox("Compare", list(metric_labels), format_func=lambda name: metric_labels[name])
st.bar_chart(summary, x="Cluster ID", y=metric, x_label="Learner group", y_label=metric_labels[metric], sort=False)
st.dataframe(
    summary[[
        "Cluster", "records", "mean_clicks", "mean_active_days", "mean_participation",
        "mean_score", "mean_late_rate", "score_coverage",
    ]].rename(columns={
        "records": "Records", "mean_clicks": "Mean clicks", "mean_active_days": "Mean active days",
        "mean_participation": "Assessment participation", "mean_score": "Mean score",
        "mean_late_rate": "Late-submission rate", "score_coverage": "Share with score",
    }).round(2),
    hide_index=True,
    width="stretch",
)
st.caption("Mean score excludes missing scores. Late-submission rate excludes records without a dated submission.")

with st.expander("K evaluation and method"):
    st.write(
        f"K-Means was fitted to {run['student_course_records']:,} records with a fixed seed of "
        f"{run['random_seed']}. Silhouette scores use a fixed sample of "
        f"{run['silhouette_sample_size']:,} records. K={run['selected_k']} was selected "
        "after reviewing both the metrics and the behavioral profiles."
    )
    left, right = st.columns(2)
    with left:
        st.caption("Elbow method: inertia by K")
        st.line_chart(evaluation, x="k", y="inertia", x_label="K", y_label="Inertia")
    with right:
        st.caption("Silhouette score by K")
        st.line_chart(evaluation, x="k", y="silhouette_sample", x_label="K", y_label="Silhouette score")
    st.dataframe(
        evaluation[["k", "inertia", "silhouette_sample", "smallest_cluster_records"]].round(3),
        hide_index=True,
        width="stretch",
    )
    st.caption("Cluster names describe this saved run. They are not predefined learner categories or diagnoses.")

st.subheader("Learner assignments")
st.caption("Each row represents one student in one module presentation. IDs are anonymized in OULAD.")
page_size = 200
page_count = (len(filtered) + page_size - 1) // page_size
page = st.number_input("Page", min_value=1, max_value=page_count, value=1, step=1)
start = (page - 1) * page_size
st.caption(f"Showing records {start + 1:,}–{min(start + page_size, len(filtered)):,} of {len(filtered):,}.")
table = filtered.iloc[start:start + page_size][[
    "code_module", "code_presentation", "id_student", "cluster",
    "vle_clicks_during_course", "vle_active_days_during_course",
    "assessment_submitted_count", "assessment_mean_score", "assessment_late_count",
]].copy()
table["cluster"] = table["cluster"].map(lambda cluster: f"{cluster}: {labels[cluster]}")
st.dataframe(
    table.rename(columns={
        "code_module": "Module", "code_presentation": "Presentation", "id_student": "Student ID",
        "cluster": "Cluster", "vle_clicks_during_course": "In-course clicks",
        "vle_active_days_during_course": "Active days",
        "assessment_submitted_count": "Assessments submitted",
        "assessment_mean_score": "Mean score", "assessment_late_count": "Late submissions",
    }).round(2),
    hide_index=True,
    width="stretch",
    height=420,
)
