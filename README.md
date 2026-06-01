# Studor PathAI DS Screening Project

Baseline implementation for the Studor PathAI take-home assessment using the Open University Learning Analytics Dataset (OULAD).

Evidence-led pipeline covering three product tasks: weekly engagement scoring, Week 6 disengagement alerts, and next-module recommendations.

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

1. **`src/run_pipeline.py`** — regenerates all outputs, figures, PDF, and Word report.
2. **Notebooks** — narrative walkthroughs that rebuild the key tables, show EDA, justify features, and compare models.
3. **`reports/`** — submission deliverables (figures + reports).

## What This Builds

### Task 1 — Engagement score (0–100)

- Weekly score from ten peer-normalized components including **pre-start proactivity** (VLE activity before day 0 / negative dates).
- Weights derived from training-split AUC at Week 6, rounded to a readable scorecard.
- Validated against observed withdraw/fail rates by score band; seven student archetypes plotted across two readable trajectory charts.

### Task 2 — Week 6 disengagement model

- Binary classifier: Withdrawn/Fail vs Pass/Distinction using only Week ≤6 features.
- Compares Logistic Regression, Random Forest, KNN, calibrated Linear SVC, XGBoost, and Soft Voting.
- Selects the urgent-alert classifier by cross-validated F1/PR-AUC, then keeps a separate high-recall F2 watchlist threshold.
- Uses a balanced urgent threshold for stronger precision/F1 and a broader watchlist threshold for higher recall.
- Reports precision, recall, F1, ROC-AUC, confusion matrix, calibration, and behavioral drivers.

### Task 3 — Course recommendations

- Content-based (demographics + engagement + proactivity history) vs collaborative filtering (cosine similarity).
- Cold-start defaults to modules where proactive starters succeed best.
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
- `risk_threshold_analysis.csv`, `risk_feature_drivers.csv`, `risk_behavioral_feature_drivers.csv`
- `task2_feature_catalog.csv`, `week6_archetype_risk_rates.csv`, `profile_group_risk_rates.csv`
- `calibration.csv`, `recommendation_metrics.json`, `recommendation_holdout_eval.csv`
- `models/week6_risk_model.joblib`

**Submission artifacts** — `reports/`:

- `figures/*.png` — all pipeline charts
- `Studor_PathAI_Report.pdf` — 6-page executive summary
- `Studor_PathAI_Report.docx` — detailed technical report

Regenerate Word only: `.venv/bin/python src/write_word_report.py`

## Latest Results

- Engagement weights: trend and punctuality receive 0% after train AUC checks; clicks, studiousness, diversity, and recency each receive 15%.
- Urgent alert: precision 0.762, recall 0.833, F1 0.796, ROC-AUC 0.872, alert share 57.7%.
- Watchlist: precision 0.603, recall 0.970, alert share 85.0%.
- Recommendations: content hit@3 0.111, CF hit@3 0.594; cold-start AAA/EEE/GGG.

## Notes

- Negative VLE dates are **pre-start engagement**, not errors — they signal student eagerness before day 0.
- `outputs/` and raw data are gitignored; commit `src/`, `notebooks/`, `requirements.txt`, and `reports/`.
- Do not commit the original brief in `task/` or raw OULAD CSVs for public GitHub submission.
