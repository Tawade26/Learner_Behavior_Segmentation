# Learner Behavior Segmentation

A local, reproducible data analytics prototype for an Elective 4 pseudo-capstone project. It uses the Open University Learning Analytics Dataset (OULAD) to describe learner behavior with K-Means clusters. The dashboard is for exploration; cluster labels are interpretations of this run, not diagnoses or predictions about individual students.

## Current progress

| Phase | Status | What is available |
| --- | --- | --- |
| 1. Data preparation | Complete | Streaming CSV preparation, one feature row per student/module/presentation, and a data quality report. |
| 2. Data mining | Complete | K-Means evaluation, a reproducible fitted run, assignments, and cluster profiles. |
| 3. Local prototype | Complete | A Streamlit dashboard for cluster distribution, behavior comparisons, K evaluation, and paged learner assignments. |

The current run covers **32,593 student-course records** and uses **K=3**. It was developed and tested locally on Windows with Python 3.14.2. The repository does not include the raw OULAD data or generated results; each collaborator runs the pipeline locally.

## Clone and run on Windows

You need Git, Python (3.14 is the version used for this prototype), and the seven OULAD CSV files. Download the dataset from [The Open University's OULAD page](https://research.stem.open.ac.uk/ouanalyse/dataset/) or use a copy supplied by the project team. Extract the CSVs and keep the `dataset` folder beside the repository as shown below. The folder names matter because this is the scripts' default layout:

```text
your-workspace/
├── dataset/
│   ├── assessments.csv
│   ├── courses.csv
│   ├── studentAssessment.csv
│   ├── studentInfo.csv
│   ├── studentRegistration.csv
│   ├── studentVle.csv
│   └── vle.csv
└── project_repo/
    ├── app.py
    ├── requirements.txt
    └── scripts/
```

From `your-workspace` in PowerShell:

```powershell
git clone https://github.com/Tawade26/Learner_Behavior_Segmentation.git project_repo
cd project_repo
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\prepare_features.py
.\.venv\Scripts\python.exe scripts\run_clustering.py --k 3
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`. Stop the server with `Ctrl+C`. **Launch with `-m streamlit run app.py`**; running `python app.py` or an editor's “Run Python File” action produces `missing ScriptRunContext` warnings and does not start the dashboard correctly.

On macOS/Linux, create the environment with `python3 -m venv .venv` and replace `.\.venv\Scripts\python.exe` in the commands with `.venv/bin/python`. The Python 3.14.2 setup above is the one tested so far.

If the dataset is elsewhere, pass `--data-dir PATH` to `prepare_features.py`. Both scripts support `--output-dir PATH`; the dashboard currently expects outputs in the repository's `generated` folder.

## What the pipeline writes

| File in `generated/` | Purpose |
| --- | --- |
| `student_features.csv` | Prepared student-course feature table. |
| `data_quality.json` | Source counts, missingness, join checks, and a pass/fail status. |
| `k_evaluation.csv` | Inertia, sampled silhouette, and cluster-size results across candidate K values. |
| `clustered_students.csv` | Prepared records with their assigned cluster IDs. |
| `cluster_profiles.csv` | Group sizes, feature means, and coverage measures. |
| `clustering_run.json` | Selected K, seed, model settings, versions, and input-file hash. |

The preparation script reads the large `studentVle.csv` file one row at a time rather than loading it all into memory. It retains every `studentInfo.csv` record, including withdrawn learners and those with no VLE or assessment records. Count features for absent activity are zero; unavailable means and timing measures are blank. Duplicate source keys, invalid numeric values, or unmatched joins stop preparation. Check `data_quality.json` if preparation fails.

The dashboard reads these saved files; changing a filter does not refit K-Means. It checks their record counts and input hash before displaying results. The current names in `cluster_labels.json` only apply to the matching feature-table hash and K. If the analysis changes, the dashboard falls back to numbered cluster names until the profiles are reviewed again.

## Data and modeling choices

The feature-table key is `(code_module, code_presentation, id_student)`. VLE clicks are joined from `studentVle.csv` to `vle.csv` by module, presentation, and site. In-course activity uses days `0` through `course_length - 1`; activity before or after the course is kept separately. Assessment submissions are joined to `assessments.csv`. Banked results are counted separately and excluded from current-presentation submission, score, and timing measures. Timing is submission date minus deadline: negative is early, zero is on time, and positive is late.

K-Means uses six behavior and assessment measures: log-transformed in-course clicks per week, in-course active-day rate, share of available activity types used, assessment participation rate, average score, and late-submission rate. Missing score and late rate are filled with their observed medians; their coverage is reported in the profiles. All six measures are standardized. Identifiers and `context_` columns are excluded from the model. In particular, `context_final_result` is an outcome and is **not** a clustering input.

The script evaluates K=2 through K=8 by default with a fixed seed and a fixed silhouette sample of at most 2,000 records. Without `--k`, it selects the highest sampled silhouette among candidates whose smallest cluster has at least 3% of records; if none qualify, it selects the highest sampled silhouette. The current prototype explicitly uses `--k 3` after reviewing the metrics and profiles. K=2 has the highest sampled silhouette (0.394), but primarily separates broad high/low activity groups. K=3 has silhouette 0.360 and about 24% lower inertia than K=2.

| Cluster | Records | Current descriptive reading |
| --- | ---: | --- |
| 0 | 8,849 | Moderate activity, lower assessment scores, and more late submissions. |
| 1 | 8,948 | Limited activity and assessment participation. |
| 2 | 14,796 | Sustained activity and high assessment participation. |

Cluster IDs are arbitrary. These descriptions are specific to the current data and feature choices. The current features do not show whether VLE activity rises near deadlines, so none of the clusters is called “deadline-driven.”

## Next improvements

1. **Validate the groups:** compare clustering across seeds and module/presentation subsets, inspect missing-score and timing coverage, and document how stable the interpretations are.
2. **Add time-based behavior:** build weekly or deadline-relative VLE measures if the project needs to study changing engagement or deadline patterns.
3. **Evaluate usefulness:** compare clusters with outcomes only after fitting, and decide with the project team which questions the segments should answer. Avoid treating cluster membership as a prediction or causal explanation.
4. **Improve data ingestion:** add a local dataset picker or validated import flow. For a future deployed version, design storage and processing for large uploads such as `studentVle.csv` before building an end-user upload screen.
5. **Refine the interface:** add clearer methodology and coverage explanations, export options, and UI polish after the analysis is validated. Deployment is not part of the current local prototype.

## Repository hygiene

`.gitignore` excludes the virtual environment, generated analysis outputs, a repo-local `dataset/` folder, caches, local secrets, logs, and operating-system/editor files. The default `../dataset` folder is outside this repository. Do not commit raw data or generated outputs; collaborators can reproduce them with the commands above.
