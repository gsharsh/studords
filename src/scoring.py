from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from data_loader import KEYS, OUT_DIR, RISK_LABELS, SCORE_COMPONENTS, SUCCESS_LABELS, WEEK_CUTOFF


def rounded_weights(raw_weights: pd.Series, step: float = 0.05) -> pd.Series:
    nonzero = raw_weights[raw_weights > 0].index
    rounded = pd.Series(0.0, index=raw_weights.index)
    if len(nonzero) == 0:
        return pd.Series(1 / len(raw_weights), index=raw_weights.index).round(4)
    rounded.loc[nonzero] = (raw_weights.loc[nonzero] / step).round() * step
    rounded.loc[nonzero] = rounded.loc[nonzero].clip(lower=step)
    diff_steps = int(round((1.0 - rounded.sum()) / step))
    if diff_steps != 0:
        order = raw_weights.loc[nonzero].sort_values(ascending=diff_steps > 0).index.tolist()
        for i in range(abs(diff_steps)):
            target = order[i % len(order)]
            rounded[target] += step if diff_steps > 0 else -step
            rounded[target] = max(rounded[target], step)
    return (rounded / rounded.sum()).round(4)


def derive_engagement_weights(weekly: pd.DataFrame, split: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    train_keys = split[split["split"] == "train"][KEYS + ["risk_label"]]
    week6 = weekly[weekly["week"] == WEEK_CUTOFF].merge(train_keys, on=KEYS, how="inner")
    success = 1 - week6["risk_label"]

    rows = []
    raw = {}
    for component, label in SCORE_COMPONENTS.items():
        auc = roc_auc_score(success, week6[component])
        signal = max(auc - 0.5, 0.0)
        raw[component] = signal
        rows.append(
            {
                "component": component,
                "label": label,
                "success_auc": auc,
                "signal_over_random": signal,
                "scorecard_decision": (
                    "Included in scorecard"
                    if signal > 0
                    else "Excluded from scorecard: train AUC <= 0.50, so the signal is noisy or directionally weak."
                ),
            }
        )

    raw_series = pd.Series(raw)
    raw_weights = raw_series / raw_series.sum() if raw_series.sum() > 0 else pd.Series(1 / len(raw_series), index=raw_series.index)
    final_weights = rounded_weights(raw_weights)
    rationale = pd.DataFrame(rows)
    rationale["raw_weight"] = rationale["component"].map(raw_weights)
    rationale["score_weight"] = rationale["component"].map(final_weights)
    rationale = rationale.sort_values("score_weight", ascending=False)
    rationale.to_csv(OUT_DIR / "engagement_weight_rationale.csv", index=False)
    return final_weights.to_dict(), rationale


def apply_engagement_score(weekly: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    weekly = weekly.copy()
    score = np.zeros(len(weekly))
    for component, weight in weights.items():
        score += weekly[component].fillna(0).to_numpy() * weight
    weekly["engagement_score"] = (100 * score).round(1)
    return weekly


def validate_engagement_score(weekly: pd.DataFrame) -> pd.DataFrame:
    week6 = weekly[weekly["week"] == WEEK_CUTOFF].copy()
    week6["risk"] = week6["final_result"].isin(RISK_LABELS).astype(int)
    week6["score_band"] = pd.cut(
        week6["engagement_score"],
        bins=[0, 20, 40, 60, 80, 100],
        labels=["0-20", "20-40", "40-60", "60-80", "80-100"],
        include_lowest=True,
    )
    bands = (
        week6.groupby("score_band", observed=False)
        .agg(
            enrolments=("risk", "size"),
            observed_withdraw_fail_rate=("risk", "mean"),
            avg_score=("engagement_score", "mean"),
        )
        .reset_index()
    )
    bands.to_csv(OUT_DIR / "engagement_score_band_risk.csv", index=False)
    return bands


def select_archetypes(weekly: pd.DataFrame) -> pd.DataFrame:
    profile = weekly.groupby(KEYS).agg(
        final_result=("final_result", "first"),
        early_score=("engagement_score", lambda s: s.head(4).mean()),
        late_score=("engagement_score", lambda s: s.tail(4).mean()),
        score_sd=("engagement_score", "std"),
        mean_score=("engagement_score", "mean"),
        max_score=("engagement_score", "max"),
        min_score=("engagement_score", "min"),
        zero_click_weeks=("weekly_clicks", lambda s: int((s == 0).sum())),
        active_week_ratio=("weekly_clicks", lambda s: float((s > 0).mean())),
        avg_activity_diversity=("activity_diversity", "mean"),
        avg_material_active_days=("material_active_days", "mean"),
        avg_material_click_share=("material_click_share", "mean"),
        avg_forum_active_days=("forum_active_days", "mean"),
        assessment_submitted_ratio=("assessment_submitted_ratio", "last"),
        punctuality_ratio=("punctuality_ratio", "last"),
        low_weight_completion_ratio=("low_weight_completion_ratio", "last"),
        late_submission_count=("late_submission_count", "last"),
        avg_score_so_far=("avg_score_so_far", "last"),
        pre_start_proactivity=("pre_start_proactivity_raw", "first"),
    )
    profile["delta"] = profile["late_score"] - profile["early_score"]
    profile["score_range"] = profile["max_score"] - profile["min_score"]
    profile["is_success"] = profile["final_result"].isin(SUCCESS_LABELS)

    selected: list[tuple[str, tuple]] = []
    used_keys: set[tuple] = set()

    def choose(archetype: str, candidates: pd.DataFrame, sort_cols: list[str], ascending: list[bool]) -> None:
        nonlocal selected, used_keys
        if candidates.empty:
            candidates = profile.copy()
        candidates = candidates.loc[[idx for idx in candidates.index if idx not in used_keys]]
        if candidates.empty:
            candidates = profile.loc[[idx for idx in profile.index if idx not in used_keys]]
        if candidates.empty:
            return
        picked = candidates.sort_values(sort_cols, ascending=ascending).index[0]
        used_keys.add(picked)
        selected.append((archetype, picked))

    steady = profile[
        profile["is_success"]
        & (profile["mean_score"] > profile["mean_score"].quantile(0.70))
    ].sort_values(["score_sd", "mean_score"], ascending=[True, False])
    dropout = profile[profile["final_result"] == "Withdrawn"]
    recoverer = profile[profile["is_success"]]
    sporadic = profile[
        (profile["score_range"] >= profile["score_range"].quantile(0.85))
        & (profile["zero_click_weeks"] >= profile["zero_click_weeks"].quantile(0.60))
        & (profile["max_score"] >= profile["max_score"].quantile(0.70))
    ]
    procrastinator = profile[
        profile["is_success"]
        & (profile["late_submission_count"] > 0)
        & (profile["avg_score_so_far"] >= profile["avg_score_so_far"].quantile(0.70))
        & (profile["punctuality_ratio"] <= profile["punctuality_ratio"].quantile(0.40))
    ]
    compliance = profile[
        profile["is_success"]
        & (profile["assessment_submitted_ratio"] >= 0.90)
        & (profile["avg_activity_diversity"] <= profile["avg_activity_diversity"].quantile(0.45))
        & (profile["avg_material_active_days"] <= profile["avg_material_active_days"].quantile(0.45))
    ]
    opportunistic = profile[
        (profile["score_range"] >= profile["score_range"].quantile(0.70))
        & (profile["avg_material_click_share"] >= profile["avg_material_click_share"].quantile(0.70))
        & (profile["active_week_ratio"] <= profile["active_week_ratio"].quantile(0.70))
    ]

    choose("Steady Engager", steady, ["score_sd", "mean_score"], [True, False])
    choose("Early Dropout", dropout, ["delta", "late_score"], [True, True])
    choose("Late Recoverer", recoverer, ["delta", "late_score"], [False, False])
    choose("Sporadic / Burst Engager", sporadic, ["score_range", "zero_click_weeks"], [False, False])
    choose("Perfectionist Procrastinator", procrastinator, ["avg_score_so_far", "late_submission_count"], [False, False])
    choose("Surface Level / Compliance Engager", compliance, ["assessment_submitted_ratio", "avg_activity_diversity"], [False, True])
    choose("Opportunistic Engager", opportunistic, ["avg_material_click_share", "score_range"], [False, False])

    archetype_keys = pd.DataFrame(
        [(*key, archetype) for archetype, key in selected],
        columns=KEYS + ["archetype"],
    )
    archetypes = weekly.merge(archetype_keys, on=KEYS, how="inner")
    definitions = pd.DataFrame(
        [
            {
                "archetype": "Steady Engager",
                "description": "Maintains consistently strong engagement across the semester.",
                "feature_signature": "High mean engagement score, low score volatility, strong active-day consistency.",
            },
            {
                "archetype": "Early Dropout",
                "description": "Disengages early and does not recover.",
                "feature_signature": "Sharp negative early-to-late score delta, low late score, often withdrawn final result.",
            },
            {
                "archetype": "Late Recoverer",
                "description": "Starts weakly but improves later in the semester.",
                "feature_signature": "Large positive score delta from early weeks to late weeks.",
            },
            {
                "archetype": "Sporadic / Burst Engager",
                "description": "Works in intense, irregular bursts rather than a steady rhythm.",
                "feature_signature": "High score range, high max score, and several zero-click weeks.",
            },
            {
                "archetype": "Perfectionist Procrastinator",
                "description": "Delays participation or submissions while still producing high-quality work.",
                "feature_signature": "Successful/high-scoring enrolment with late submissions and lower punctuality.",
            },
            {
                "archetype": "Surface Level / Compliance Engager",
                "description": "Does the minimum required to pass or complete the task.",
                "feature_signature": "High assessment completion but low activity diversity and low material-study frequency.",
            },
            {
                "archetype": "Opportunistic Engager",
                "description": "Selectively engages when material appears useful or interesting.",
                "feature_signature": "High material-click share and engagement spikes, but weaker active-week consistency.",
            },
        ]
    )
    selected_profile = profile.loc[[key for _, key in selected]].reset_index()
    selected_profile = selected_profile.merge(archetype_keys, on=KEYS, how="left")
    definitions = definitions.merge(selected_profile, on="archetype", how="left")
    definitions.to_csv(OUT_DIR / "engagement_archetype_definitions.csv", index=False)
    archetypes.to_csv(OUT_DIR / "engagement_archetypes.csv", index=False)
    return archetypes
