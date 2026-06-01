"""Derived statistics for report generation from pipeline outputs."""

from __future__ import annotations

import pandas as pd

from data_loader import KEYS, OUT_DIR, RISK_LABELS, WEEK_CUTOFF


def pre_start_validation() -> dict[str, float | int]:
    weekly = pd.read_csv(OUT_DIR / "weekly_engagement_features.csv")
    week6 = weekly[weekly["week"] == WEEK_CUTOFF].drop_duplicates(KEYS)
    week6 = week6.assign(risk=week6["final_result"].isin(RISK_LABELS).astype(int))
    has_pre = week6["pre_start_flag"] == 1
    return {
        "enrolments": int(len(week6)),
        "with_pre_start": int(has_pre.sum()),
        "with_pre_start_pct": float(has_pre.mean()),
        "risk_with_pre_start": float(week6.loc[has_pre, "risk"].mean()) if has_pre.any() else 0.0,
        "risk_without_pre_start": float(week6.loc[~has_pre, "risk"].mean()) if (~has_pre).any() else 0.0,
    }


def split_summary() -> dict[str, float | int]:
    split = pd.read_csv(OUT_DIR / "enrollment_train_test_split.csv")
    test = split[split["split"] == "test"]
    return {
        "total": int(len(split)),
        "train": int((split["split"] == "train").sum()),
        "test": int(len(test)),
        "risk_rate": float(split["risk_label"].mean()),
    }


def archetype_lines() -> list[str]:
    archetypes = pd.read_csv(OUT_DIR / "engagement_archetypes.csv")
    keys = archetypes[KEYS + ["archetype", "final_result"]].drop_duplicates(KEYS + ["archetype"])
    lines = []
    for _, row in keys.iterrows():
        lines.append(
            f"{row['archetype']}: student {int(row['id_student'])}, module {row['code_module']}, "
            f"result {row['final_result']}"
        )
    return lines


def archetype_definitions() -> pd.DataFrame:
    return pd.read_csv(OUT_DIR / "engagement_archetype_definitions.csv")


def feature_description(name: str, description: str, rationale: pd.DataFrame) -> str:
    row = rationale.loc[rationale["feature"] == name].iloc[0]
    prefix = f"{description} " if description else ""
    return (
        f"{name}: {prefix}withdraw/fail mean = {row['withdraw_fail_mean']:.2f}, "
        f"pass/distinction mean = {row['pass_distinction_mean']:.2f} "
        f"(r = {row['risk_correlation']:+.3f})."
    )
