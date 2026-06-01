from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
LEGACY_DATA_DIR = ROOT / "dataset"
OUT_DIR = ROOT / "outputs"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
MODEL_DIR = OUT_DIR / "models"

WEEK_CUTOFF = 6
TEST_SIZE = 0.25
RANDOM_STATE = 42
URGENT_RECALL_FLOOR = 0.85
CAPACITY_QUANTILE = 0.80
DAYS_SINCE_CLICK_MISSING = 999
RISK_LABELS = {"Withdrawn", "Fail"}
SUCCESS_LABELS = {"Pass", "Distinction"}
KEYS = ["code_module", "code_presentation", "id_student"]
SCORE_COMPONENTS = {
    "p_pre_start_proactivity": "Pre-start course material engagement",
    "p_clicks": "Peer-normalized click volume",
    "p_active_days": "Active-day consistency",
    "p_studiousness": "Module-material study frequency",
    "p_activity_diversity": "Activity diversity",
    "p_recency": "Recent activity",
    "p_trend": "Three-week trend",
    "p_submission_completion": "Assessment completion",
    "p_punctuality": "Assessment punctuality",
    "p_low_weight_completion": "Low-weight assignment completion",
}


def load_pipeline_weekly() -> pd.DataFrame:
    path = OUT_DIR / "weekly_engagement_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run: .venv/bin/python src/run_pipeline.py")
    return pd.read_csv(path)


def load_pipeline_split() -> pd.DataFrame:
    path = OUT_DIR / "enrollment_train_test_split.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run: .venv/bin/python src/run_pipeline.py")
    return pd.read_csv(path)


def ensure_dirs() -> None:
    for path in [OUT_DIR, FIG_DIR, MODEL_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def resolve_data_dir() -> Path:
    if (DATA_DIR / "studentInfo.csv").exists():
        return DATA_DIR
    if (LEGACY_DATA_DIR / "studentInfo.csv").exists():
        return LEGACY_DATA_DIR
    return DATA_DIR


def load_data(data_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    base = data_dir or resolve_data_dir()
    return {
        "courses": pd.read_csv(base / "courses.csv"),
        "student_info": pd.read_csv(base / "studentInfo.csv"),
        "registration": pd.read_csv(base / "studentRegistration.csv"),
        "assessments": pd.read_csv(base / "assessments.csv"),
        "student_assessment": pd.read_csv(base / "studentAssessment.csv"),
        "vle": pd.read_csv(base / "vle.csv"),
        "student_vle": pd.read_csv(base / "studentVle.csv"),
    }


def audit_and_clean_data(data: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Audit OULAD quality while applying only conservative cleaning.

    We do not rewrite the official `final_result` labels. Inconsistencies are
    reported so the data-cleaning story is explicit and reproducible.
    """
    ensure_dirs()
    cleaned = {name: df.copy() for name, df in data.items()}

    overview_rows = []
    missing_rows = []
    for name, df in cleaned.items():
        overview_rows.append(
            {
                "table": name,
                "rows": int(len(df)),
                "columns": int(df.shape[1]),
                "duplicate_rows": int(df.duplicated().sum()),
                "missing_cells": int(df.isna().sum().sum()),
            }
        )
        for col, count in df.isna().sum().items():
            if count > 0:
                missing_rows.append(
                    {
                        "table": name,
                        "column": col,
                        "missing_count": int(count),
                        "missing_pct": float(count / len(df)),
                    }
                )

    student_info = cleaned["student_info"]
    registration = cleaned["registration"]
    assessments = cleaned["assessments"]
    student_assessment = cleaned["student_assessment"]
    vle = cleaned["vle"]
    student_vle = cleaned["student_vle"]
    courses = cleaned["courses"]

    consistency = []
    consistency.append(
        {
            "check": "studentInfo key uniqueness",
            "issue_count": int(student_info.duplicated(KEYS).sum()),
            "policy": "Use student-module-presentation as the enrolment key.",
        }
    )
    consistency.append(
        {
            "check": "studentRegistration key uniqueness",
            "issue_count": int(registration.duplicated(KEYS).sum()),
            "policy": "Use student-module-presentation as the enrolment key.",
        }
    )
    unregistered = registration[registration["date_unregistration"].notna()]
    unreg_with_result = student_info.merge(unregistered[KEYS + ["date_unregistration"]], on=KEYS, how="inner")
    consistency.append(
        {
            "check": "Unregistered enrolments not labelled Withdrawn",
            "issue_count": int((unreg_with_result["final_result"] != "Withdrawn").sum()),
            "policy": "Audit only; official final_result remains the supervised target.",
        }
    )
    consistency.append(
        {
            "check": "Withdrawn enrolments without unregistration date",
            "issue_count": int(
                student_info.merge(registration[KEYS + ["date_unregistration"]], on=KEYS, how="left")
                .query("final_result == 'Withdrawn' and date_unregistration.isna()")
                .shape[0]
            ),
            "policy": "Do not use date_unregistration as a predictive feature because it can occur after Week 6.",
        }
    )
    consistency.append(
        {
            "check": "Missing score rows",
            "issue_count": int(student_assessment["score"].isna().sum()),
            "policy": "Keep missing scores as missing; median-impute only inside model preprocessing.",
        }
    )
    consistency.append(
        {
            "check": "Invalid score rows outside 0-100",
            "issue_count": int(
                student_assessment["score"].dropna().lt(0).sum()
                + student_assessment["score"].dropna().gt(100).sum()
            ),
            "policy": "Invalid values would be excluded from score aggregates.",
        }
    )
    consistency.append(
        {
            "check": "Assessment rows with missing due date",
            "issue_count": int(assessments["date"].isna().sum()),
            "policy": "Exclude missing-date assessments from weekly due/submission features.",
        }
    )
    weight_check = assessments.groupby(["code_module", "code_presentation", "assessment_type"], as_index=False)[
        "weight"
    ].sum()
    consistency.append(
        {
            "check": "Negative VLE activity dates (pre-start engagement)",
            "issue_count": int((student_vle["date"] < 0).sum()),
            "policy": "Valid pre-presentation activity; date is days before module start. Used as proactivity signal.",
        }
    )
    vle_with_length = student_vle.merge(courses, on=["code_module", "code_presentation"], how="left")
    consistency.append(
        {
            "check": "VLE activity after module presentation length",
            "issue_count": int((vle_with_length["date"] > vle_with_length["module_presentation_length"]).sum()),
            "policy": "Exclude from weekly engagement panel.",
        }
    )
    consistency.append(
        {
            "check": "Missing VLE week_from/week_to metadata",
            "issue_count": int(vle[["week_from", "week_to"]].isna().sum().sum()),
            "policy": "Do not use week_from/week_to as model features.",
        }
    )

    leakage_rows = [
        ["Pre-start VLE activity (negative dates)", "Yes", "Yes", "Observed before day 0; reflects eagerness to begin course materials."],
        ["Week 1-6 VLE clicks", "Yes", "Yes", "Observed before the Week 6 prediction point."],
        ["Week 1-6 active days", "Yes", "Yes", "Observed before the Week 6 prediction point."],
        ["Week 1-6 activity diversity", "Yes", "Yes", "Observed before the Week 6 prediction point."],
        ["Early non-exam assessment submissions", "Yes", "Yes", "Only due/submitted work by Week 6 is used."],
        ["Student demographics/context", "Yes", "Yes", "Known at registration or course enrolment."],
        ["Engagement score", "Yes", "Yes", "Computed from Week 1-6 behavior using train-derived score weights."],
        ["Final result", "No", "Target only", "Used only to train/evaluate the supervised label."],
        ["Final exam score", "No", "No", "Not available by Week 6."],
        ["Full-semester total clicks", "No", "No", "Would leak future engagement behavior."],
        ["date_unregistration after Week 6", "No", "No", "Would leak withdrawal timing/outcome."],
    ]

    audits = {
        "overview": pd.DataFrame(overview_rows),
        "missingness": pd.DataFrame(missing_rows),
        "consistency": pd.DataFrame(consistency),
        "leakage": pd.DataFrame(leakage_rows, columns=["candidate_feature", "available_by_week6", "used", "reason"]),
        "assessment_weights": weight_check,
    }
    audits["overview"].to_csv(OUT_DIR / "data_cleaning_overview.csv", index=False)
    audits["missingness"].to_csv(OUT_DIR / "data_missingness_audit.csv", index=False)
    audits["consistency"].to_csv(OUT_DIR / "data_consistency_audit.csv", index=False)
    audits["leakage"].to_csv(OUT_DIR / "leakage_audit.csv", index=False)
    audits["assessment_weights"].to_csv(OUT_DIR / "assessment_weight_audit.csv", index=False)
    return cleaned, audits
