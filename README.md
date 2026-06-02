# Studor PathAI DS Screening Project

Submission-ready implementation for the Studor PathAI take-home assessment using the Open University Learning Analytics Dataset (OULAD).

The project is intentionally notebook-first: the notebooks show the data cleaning, feature reasoning, model selection evidence, and recommendation evaluation, while `src/` keeps the reusable pipeline code clean and reproducible.

## Project Structure

```text
studords/
├── data/ or dataset/           # OULAD CSVs (gitignored; download separately)
├── notebooks/
│   ├── 01_engagement_score.ipynb
│   ├── 02_disengagement_model.ipynb
│   └── 03_recommendation_engine.ipynb
├── src/
│   ├── data_loader.py            # paths, constants, loading, audits
│   ├── vle_utils.py              # shared VLE merge helpers
│   ├── features.py               # weekly + pre-start feature engineering
│   ├── scoring.py                # engagement score weights and validation
│   ├── modeling.py               # Week 6 risk classifier
│   ├── recommendations.py        # content-based + CF recommender
│   ├── evaluation.py             # plots + 6-page PDF summary
│   ├── report_stats.py           # derived stats for Word report
│   ├── write_word_report.py      # detailed Word report generator
│   └── run_pipeline.py           # single entry point
├── reports/
│   ├── figures/                  # pipeline figures (tracked)
│   ├── Studor_PathAI_Report.pdf  # concise submission summary
│   └── Studor_PathAI_Report.docx # detailed technical report
├── requirements.txt
└── README.md
```

## Quick Start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python src/run_pipeline.py
```

The first run takes a few minutes because `studentVle.csv` has 10M+ rows.

## Workflow

1. **Notebooks** — main evidence trail for the assignment. They show EDA, assumptions, feature justification, model selection, and evaluation.
2. **`src/run_pipeline.py`** — reproducible pipeline entry point. It regenerates outputs, figures, PDF, and Word report from the raw OULAD CSVs.
3. **`reports/`** — final submission artifacts: the 6-page PDF report, editable Word report, and figures.

## What This Builds

### Task 1 — Engagement score (0–100)

- Weekly score from ten peer-normalized components including **pre-start proactivity** (VLE activity before day 0 / negative dates).
- Weights derived from training-split AUC at Week 6, rounded to a readable scorecard.
- Validated against observed withdraw/fail rates by score band; seven student archetypes plotted across two readable trajectory charts.

### Task 2 — Week 6 disengagement model

- Binary classifier: Withdrawn/Fail vs Pass/Distinction using only Week ≤6 features.
- Model-selection evidence compared Logistic Regression, Random Forest, XGBoost, and regularized XGBoost.
- Based on that comparison table, the production pipeline trains **XGBoost only**: it had the best validation F1 (0.787) and recall (0.842), with competitive ROC-AUC (0.863) and PR-AUC (0.895).
- Converts predicted risk into operational tiers instead of calling most students “urgent.”
- Uses top 20% risk as a high-touch advisor support queue, the next 40% for light-touch behavioural nudges, and the remainder for monitoring.
- Headline metrics: ROC-AUC 0.870, high-touch precision 0.991, top-60% light-touch F1 0.793, and top-60% light-touch recall 0.848.
- Reports precision, recall, F1, ROC-AUC, tier-level observed risk, calibration, and behavioral drivers.

### Task 3 — Course recommendations

- Content-based shared feature-space recommender: Week-6 student vector vs module profile vector (cosine similarity + Wilson pass-rate prior).
- Collaborative filtering baseline over prior successful module patterns.
- Temporal holdout split: train on `2013B/2013J/2014B`, evaluate on `2014J`.
- Cold-start defaults to modules ranked by Wilson lower-bound success prior.
- Evaluated with temporal holdout hit@3 and catalog coverage.

## Dataset

Download the seven OULAD CSV files from [Kaggle](https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad) into `data/`:

```text
assessments.csv  courses.csv  studentAssessment.csv  studentInfo.csv
studentRegistration.csv  studentVle.csv  vle.csv
```

The loader also checks `dataset/` if `data/` is absent.

## Outputs

**Generated (gitignored)** — `outputs/`:

- `weekly_engagement_features.csv`, `weekly_engagement_scores.csv`, `engagement_archetypes.csv`
- `data_cleaning_overview.csv`, `data_consistency_audit.csv`, `data_missingness_audit.csv`, `leakage_audit.csv`
- `engagement_weight_rationale.csv`, `engagement_score_band_risk.csv`, `feature_rationale_week6.csv`
- `enrollment_train_test_split.csv`, `risk_metrics.json`, `risk_watchlist_metrics.json`
- `risk_threshold_analysis.csv`, `risk_intervention_tiers.csv`, `risk_feature_drivers.csv`, `risk_behavioral_feature_drivers.csv`
- `task2_feature_catalog.csv`, `week6_archetype_risk_rates.csv`, `profile_group_risk_rates.csv`
- `calibration.csv`, `recommendation_metrics.json`, `recommendation_holdout_eval.csv`
- `models/week6_risk_model.joblib`

**Submission artifacts** — `reports/`:

- `figures/*.png` — all pipeline charts
- `Studor_PathAI_Report.pdf` — 6-page executive summary
- `Studor_PathAI_Report.docx` — detailed technical report

Regenerate Word only: `.venv/bin/python src/write_word_report.py`

## Submission Checklist

- Run the three notebooks from top to bottom if you want the full analytical walkthrough.
- Run `.venv/bin/python src/run_pipeline.py` to reproduce all generated outputs and reports.
- Submit the GitHub repo without raw OULAD CSVs, `outputs/`, `.venv/`, or `task/`.
- Attach [reports/Studor_PathAI_Report.pdf](reports/Studor_PathAI_Report.pdf) as the 6-page PDF report.
- Use [reports/Studor_PathAI_Report.docx](reports/Studor_PathAI_Report.docx) if final manual editing is needed before export.

## Walkthrough Video Guide

Suggested 10-minute structure:

1. **Project framing** — PathAI turns OULAD clickstream, assessment, and profile data into engagement scores, risk tiers, and next-module recommendations.
2. **Task 1** — show the six feature buckets, Week 6 score validation, and archetype trajectory charts.
3. **Task 2** — explain leakage-safe Week 6 modelling, why XGBoost was selected, and why the output is tiered to avoid alert fatigue.
4. **Task 3** — compare collaborative filtering with the shared feature-space content model and explain the cold-start strategy.
5. **Reflection** — one key decision: turning model risk into intervention tiers; one next step: live advisor feedback and subgroup calibration.

## Latest Results

- Engagement weights: trend and punctuality receive 0% after train AUC checks; clicks, studiousness, diversity, and recency each receive 15%.
- Model choice: XGBoost selected from the model-comparison table (validation F1 0.787, recall 0.842, ROC-AUC 0.863, PR-AUC 0.895); default runs train XGBoost only to keep the pipeline faster.
- Week 6 headline: XGBoost ROC-AUC 0.870; high-touch precision 0.991; top-60% light-touch F1 0.793 and recall 0.848.
- High-touch support queue: 20.1% of students, observed withdraw/fail rate 99.1%, captures 37.7% of at-risk students.
- Light-touch nudges: next 40.1% of students, observed withdraw/fail rate 62.0%, captures another 47.1% of at-risk students.
- Monitoring: remaining 39.8% of students, observed withdraw/fail rate 20.2%, reducing advisor alert fatigue.
- Recommendations (temporal holdout `2014J`): content hit@3 0.354, CF hit@3 0.468; both achieve 7/7 coverage; cold-start AAA/GGG/EEE.

## Notes

- Negative VLE dates are **pre-start engagement**, not errors — they signal student eagerness before day 0.
- `outputs/` and raw data are gitignored; commit `src/`, `notebooks/`, `requirements.txt`, and `reports/`.
- Do not commit the original brief in `task/` or raw OULAD CSVs for public GitHub submission.
