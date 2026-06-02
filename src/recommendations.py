from __future__ import annotations

from typing import Any

import json
import numpy as np
import pandas as pd

from data_loader import KEYS, OUT_DIR, SUCCESS_LABELS, WEEK_CUTOFF


def _wilson_lower_bound(successes: float, total: float, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    p = successes / total
    denom = 1.0 + (z**2 / total)
    center = p + (z**2 / (2.0 * total))
    margin = z * np.sqrt((p * (1.0 - p) / total) + (z**2 / (4.0 * total**2)))
    return float((center - margin) / denom)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, np.maximum(norms, 1e-12))


def _build_week6_enrollment_profile(weekly: pd.DataFrame) -> pd.DataFrame:
    week6 = weekly[weekly["week"] <= WEEK_CUTOFF]
    profile = (
        week6.groupby(KEYS)
        .agg(
            week6_clicks=("weekly_clicks", "sum"),
            week6_clicks_norm=("weekly_clicks_norm", "mean"),
            week6_active_days=("active_days_last_7", "sum"),
            week6_activity_diversity=("activity_diversity", "mean"),
            week6_volatility=("week_to_week_volatility", "last"),
            week6_slope=("click_trend_3w", "last"),
            week6_homepage_share=("homepage_dependency_ratio", "mean"),
            week6_submission_rate=("assessment_submitted_ratio", "last"),
            week6_punctuality=("punctuality_ratio", "last"),
            week6_engagement_score=("engagement_score", "last"),
            week6_pre_start_flag=("pre_start_flag", "first"),
        )
        .reset_index()
    )
    profile["task1_archetype"] = np.select(
        [
            (profile["week6_engagement_score"] >= 60) & (profile["week6_slope"] >= -2),
            (profile["week6_engagement_score"] < 40) & (profile["week6_submission_rate"] < 0.6),
            (profile["week6_engagement_score"] < 50) & (profile["week6_volatility"] > profile["week6_volatility"].median()),
        ],
        ["steady_engager", "early_disengager", "sporadic_burst_engager"],
        default="mixed_engagement",
    )
    return profile


def evaluate_recommenders(data: dict[str, pd.DataFrame], weekly: pd.DataFrame) -> dict[str, Any]:
    info = data["student_info"].copy()
    week6_profile = _build_week6_enrollment_profile(weekly)
    info = info.merge(week6_profile, on=KEYS, how="left")
    info["success"] = info["final_result"].isin(SUCCESS_LABELS).astype(int)
    modules = sorted(info["code_module"].unique())

    train = info[info["code_presentation"].isin(["2013B", "2013J", "2014B"])].copy()
    holdout = info[info["code_presentation"].eq("2014J")].copy()
    seen_in_train = set(train["id_student"].unique())
    holdout = holdout[holdout["id_student"].isin(seen_in_train)].copy()

    # Build a shared student-course feature space.
    feature_cols = [
        "week6_clicks_norm",
        "week6_active_days",
        "week6_activity_diversity",
        "week6_engagement_score",
        "week6_volatility",
        "week6_slope",
        "week6_homepage_share",
        "week6_punctuality",
        "week6_submission_rate",
        "studied_credits",
        "num_of_prev_attempts",
        "highest_education",
        "imd_band",
        "age_band",
        "disability",
        "gender",
        "task1_archetype",
    ]
    feature_cols = [c for c in feature_cols if c in train.columns]
    train_matrix = train[["id_student", "code_module", "success"] + feature_cols].copy()
    holdout_matrix = holdout[["id_student", "code_module", "success"] + feature_cols].copy()

    combined = pd.concat(
        [train_matrix[feature_cols], holdout_matrix[feature_cols]],
        axis=0,
        ignore_index=True,
    )
    non_numeric_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(combined[c])]
    encoded = pd.get_dummies(combined, columns=non_numeric_cols, dummy_na=True)
    numeric_cols = encoded.select_dtypes(include=["number"]).columns.tolist()
    means = encoded.loc[: len(train_matrix) - 1, numeric_cols].mean()
    stds = encoded.loc[: len(train_matrix) - 1, numeric_cols].std().replace(0, 1.0)
    encoded[numeric_cols] = (encoded[numeric_cols] - means) / stds
    train_vectors = encoded.iloc[: len(train_matrix)].to_numpy(dtype=float)
    holdout_vectors = encoded.iloc[len(train_matrix) :].to_numpy(dtype=float)

    train_norm = _normalize_rows(train_vectors)
    holdout_norm = _normalize_rows(holdout_vectors)

    course_profiles = (
        pd.DataFrame(train_vectors, index=train_matrix.index)
        .assign(code_module=train_matrix["code_module"].values)
        .groupby("code_module")
        .mean()
    )
    course_norm = _normalize_rows(course_profiles.to_numpy(dtype=float))
    course_ids = course_profiles.index.tolist()

    module_stats = train.groupby("code_module")["success"].agg(["sum", "count"]).reindex(modules).fillna(0)
    wilson_prior = module_stats.apply(lambda r: _wilson_lower_bound(r["sum"], r["count"]), axis=1)
    if wilson_prior.max() > wilson_prior.min():
        wilson_prior = (wilson_prior - wilson_prior.min()) / (wilson_prior.max() - wilson_prior.min())
    else:
        wilson_prior = wilson_prior * 0.0

    # Keep alpha fixed and explainable for reproducibility.
    best_alpha = 0.2

    def _content_recs_for_vector(
        sid: int,
        vec: np.ndarray,
        alpha: float,
        taken_map: dict[int, set[str]],
    ) -> list[str]:
        sims = course_norm @ vec
        score_df = pd.DataFrame({"code_module": course_ids, "cosine_similarity": sims})
        score_df["wilson_prior"] = score_df["code_module"].map(wilson_prior).fillna(0)
        score_df["score"] = (1.0 - alpha) * score_df["cosine_similarity"] + alpha * score_df["wilson_prior"]
        taken = taken_map.get(sid, set())
        ranked = score_df[~score_df["code_module"].isin(taken)].sort_values("score", ascending=False)["code_module"].tolist()
        return ranked[:3]

    taken_train = train.groupby("id_student")["code_module"].apply(set).to_dict()
    user_module = train.pivot_table(index="id_student", columns="code_module", values="success", aggfunc="max").fillna(0)

    def cf_recs(row: pd.Series) -> list[str]:
        sid = row["id_student"]
        if sid not in user_module.index or user_module.loc[sid].sum() == 0:
            return []
        target = user_module.loc[sid].to_numpy()
        matrix = user_module.to_numpy()
        denom = np.linalg.norm(matrix, axis=1) * np.linalg.norm(target)
        sims = np.divide(matrix @ target, denom, out=np.zeros(len(matrix)), where=denom > 0)
        sim_series = pd.Series(sims, index=user_module.index).drop(index=sid, errors="ignore").nlargest(50)
        peer_ids = sim_series.index
        peer_pre_start = train[train["id_student"].isin(peer_ids)].drop_duplicates("id_student").set_index("id_student")[
            "week6_pre_start_flag"
        ]
        peer_weight = sim_series * (1 + 0.15 * peer_pre_start.reindex(peer_ids).fillna(0))
        scores = user_module.loc[peer_ids].T.dot(peer_weight)
        taken = set(train.loc[train["id_student"] == sid, "code_module"])
        recs = [m for m in scores.sort_values(ascending=False).index if m not in taken][:3]
        return recs

    records = []
    for i, (_, row) in enumerate(holdout.iterrows()):
        actual = row["code_module"]
        sid = int(row["id_student"])
        content = _content_recs_for_vector(sid, holdout_norm[i], best_alpha, taken_train)
        cf = cf_recs(row)
        if not cf:
            cf = content
        records.append(
            {
                "id_student": sid,
                "actual_next_module": actual,
                "code_presentation": row["code_presentation"],
                "content_recs": content,
                "cf_recs": cf,
                "content_hit_at_3": int(actual in content),
                "cf_hit_at_3": int(actual in cf),
            }
        )
    eval_df = pd.DataFrame(records)
    eval_df.to_csv(OUT_DIR / "recommendation_holdout_eval.csv", index=False)
    metrics = {
        "holdout_students": int(len(eval_df)),
        "content_hit_rate_at_3": float(eval_df["content_hit_at_3"].mean()),
        "cf_hit_rate_at_3": float(eval_df["cf_hit_at_3"].mean()),
        "content_coverage": int(len(set(sum(eval_df["content_recs"].tolist(), [])))),
        "cf_coverage": int(len(set(sum(eval_df["cf_recs"].tolist(), [])))),
        "catalog_modules": int(len(modules)),
        "cold_start_strategy": wilson_prior.sort_values(ascending=False).head(3).index.tolist(),
        "similarity_metric": "Cosine similarity between Week-6 student vectors and course profile vectors.",
        "content_features": "Week-6 behaviour, demographics, study background, and Task-1 archetype labels in a shared feature space.",
        "content_alpha": float(best_alpha),
        "evaluation_split": "Train: 2013B/2013J/2014B, Holdout: 2014J",
    }
    (OUT_DIR / "recommendation_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
