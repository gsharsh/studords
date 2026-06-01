from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data_loader import (
    DAYS_SINCE_CLICK_MISSING,
    KEYS,
    OUT_DIR,
    RANDOM_STATE,
    RISK_LABELS,
    SUCCESS_LABELS,
    TEST_SIZE,
    WEEK_CUTOFF,
)
from vle_utils import MATERIAL_ACTIVITY_TYPES, merge_vle_activity_types

FORUM_ACTIVITY_TYPES = ["forumng"]
SOCIAL_ACTIVITY_TYPES = ["forumng", "ouwiki", "oucollaborate", "ouelluminate"]
QUIZ_ACTIVITY_TYPES = ["quiz", "externalquiz"]
HOMEPAGE_ACTIVITY_TYPES = ["homepage"]


def gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or values.sum() <= 0:
        return 0.0
    values = np.sort(values)
    n = len(values)
    return float((2 * np.arange(1, n + 1) @ values) / (n * values.sum()) - (n + 1) / n)


def entropy_from_clicks(values: pd.Series) -> float:
    total = values.sum()
    if total <= 0:
        return 0.0
    p = values / total
    return float(-(p * np.log2(p)).sum())


def add_week(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    out["week"] = np.floor(out[date_col] / 7).astype(int) + 1
    return out


def build_pre_start_features(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Aggregate VLE activity before module day 0 (negative dates).

    In OULAD, date is days since presentation start; negative values mean the
    student interacted with materials before the official start — a proactivity signal.
    """
    vle_events = merge_vle_activity_types(data["student_vle"], data["vle"])
    pre_start = vle_events[vle_events["date"] < 0].copy()
    enrolment_pre = pre_start.groupby(KEYS).agg(
        pre_start_clicks=("sum_click", "sum"),
        pre_start_active_days=("date", "nunique"),
        earliest_pre_start_day=("date", "min"),
        pre_start_activity_diversity=("activity_type", "nunique"),
    ).reset_index()
    enrolment_pre["days_before_start"] = -enrolment_pre["earliest_pre_start_day"]

    material_pre = pre_start[pre_start["activity_type"].isin(MATERIAL_ACTIVITY_TYPES)]
    material_agg = material_pre.groupby(KEYS).agg(pre_start_material_clicks=("sum_click", "sum")).reset_index()
    enrolment_pre = enrolment_pre.merge(material_agg, on=KEYS, how="left")

    roster = data["student_info"][KEYS].drop_duplicates()
    enrolment_pre = roster.merge(enrolment_pre, on=KEYS, how="left")
    fill_zero = [
        "pre_start_clicks",
        "pre_start_active_days",
        "days_before_start",
        "pre_start_activity_diversity",
        "pre_start_material_clicks",
    ]
    enrolment_pre[fill_zero] = enrolment_pre[fill_zero].fillna(0)
    enrolment_pre["pre_start_flag"] = (enrolment_pre["pre_start_clicks"] > 0).astype(int)
    enrolment_pre["pre_start_clicks_log"] = np.log1p(enrolment_pre["pre_start_clicks"])
    enrolment_pre["pre_start_material_clicks_log"] = np.log1p(enrolment_pre["pre_start_material_clicks"])
    return enrolment_pre


def build_weekly_base_features(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    roster = data["student_info"].merge(data["courses"], on=["code_module", "code_presentation"], how="left")
    max_week = int(np.ceil(roster["module_presentation_length"].max() / 7))

    panel = roster[KEYS + ["final_result", "module_presentation_length"]].merge(
        pd.DataFrame({"week": range(1, max_week + 1)}),
        how="cross",
    )
    panel = panel[panel["week"] <= np.ceil(panel["module_presentation_length"] / 7)]

    pre_start_features = build_pre_start_features(data)

    vle_events = merge_vle_activity_types(
        data["student_vle"].merge(data["courses"], on=["code_module", "code_presentation"], how="left"),
        data["vle"],
    )
    # In-week panel uses day 0+ only; pre-start (negative dates) is merged separately.
    vle_events = vle_events[vle_events["date"].between(0, vle_events["module_presentation_length"])]
    vle_events = add_week(vle_events, "date")
    vle_events["has_pace_metadata"] = vle_events["week_from"].notna() & vle_events["week_to"].notna()
    vle_events["is_on_schedule"] = (
        vle_events["has_pace_metadata"] & (vle_events["week"] >= vle_events["week_from"]) & (vle_events["week"] <= vle_events["week_to"])
    ).astype(int)
    vle_events["is_ahead"] = (vle_events["has_pace_metadata"] & (vle_events["week"] < vle_events["week_from"])).astype(int)
    vle_events["is_catchup"] = (vle_events["has_pace_metadata"] & (vle_events["week"] > vle_events["week_to"])).astype(int)
    vle_events["material_lag"] = np.where(vle_events["has_pace_metadata"], vle_events["week"] - vle_events["week_from"], np.nan)

    daily_clicks = (
        vle_events.groupby(KEYS + ["week", "date"])
        .agg(daily_clicks=("sum_click", "sum"))
        .reset_index()
    )
    weekly_day_shape = (
        daily_clicks.groupby(KEYS + ["week"])
        .agg(
            max_daily_clicks=("daily_clicks", "max"),
            click_gini_wk=("daily_clicks", lambda s: gini(s.to_numpy())),
        )
        .reset_index()
    )
    activity_entropy = (
        vle_events.groupby(KEYS + ["week", "activity_type"])["sum_click"]
        .sum()
        .groupby(level=KEYS + ["week"])
        .apply(entropy_from_clicks)
        .reset_index(name="activity_entropy_wk")
    )
    weekly_vle = (
        vle_events.groupby(KEYS + ["week"])
        .agg(
            total_clicks=("sum_click", "sum"),
            active_days=("date", "nunique"),
            active_days_last_7=("date", "nunique"),
            unique_sites_wk=("id_site", "nunique"),
            activity_diversity=("activity_type", "nunique"),
            first_click_day_week=("date", "min"),
            last_click_day=("date", "max"),
        )
        .reset_index()
    )
    material_events = vle_events[vle_events["activity_type"].isin(MATERIAL_ACTIVITY_TYPES)]
    weekly_material = (
        material_events.groupby(KEYS + ["week"])
        .agg(
            material_clicks=("sum_click", "sum"),
            material_active_days=("date", "nunique"),
        )
        .reset_index()
    )
    forum_events = vle_events[vle_events["activity_type"].isin(FORUM_ACTIVITY_TYPES)]
    weekly_forum = (
        forum_events.groupby(KEYS + ["week"])
        .agg(
            forum_clicks=("sum_click", "sum"),
            forum_active_days=("date", "nunique"),
        )
        .reset_index()
    )
    social_events = vle_events[vle_events["activity_type"].isin(SOCIAL_ACTIVITY_TYPES)]
    weekly_social = social_events.groupby(KEYS + ["week"]).agg(social_clicks=("sum_click", "sum")).reset_index()
    quiz_events = vle_events[vle_events["activity_type"].isin(QUIZ_ACTIVITY_TYPES)]
    weekly_quiz = quiz_events.groupby(KEYS + ["week"]).agg(quiz_clicks=("sum_click", "sum")).reset_index()
    homepage_events = vle_events[vle_events["activity_type"].isin(HOMEPAGE_ACTIVITY_TYPES)]
    weekly_homepage = homepage_events.groupby(KEYS + ["week"]).agg(homepage_clicks=("sum_click", "sum")).reset_index()
    pace_events = vle_events[vle_events["has_pace_metadata"]]
    weekly_pace = (
        pace_events.groupby(KEYS + ["week"])
        .agg(
            pace_metadata_clicks=("sum_click", "sum"),
            on_schedule_clicks=("sum_click", lambda s: s[pace_events.loc[s.index, "is_on_schedule"] == 1].sum()),
            ahead_clicks=("sum_click", lambda s: s[pace_events.loc[s.index, "is_ahead"] == 1].sum()),
            catchup_clicks=("sum_click", lambda s: s[pace_events.loc[s.index, "is_catchup"] == 1].sum()),
            avg_material_lag=("material_lag", "mean"),
        )
        .reset_index()
    )
    planned_material_coverage = build_planned_material_coverage(vle_events, data["vle"], max_week)

    weekly = panel.merge(weekly_vle, on=KEYS + ["week"], how="left")
    weekly = weekly.merge(weekly_day_shape, on=KEYS + ["week"], how="left")
    weekly = weekly.merge(activity_entropy, on=KEYS + ["week"], how="left")
    weekly = weekly.merge(weekly_material, on=KEYS + ["week"], how="left")
    weekly = weekly.merge(weekly_forum, on=KEYS + ["week"], how="left")
    weekly = weekly.merge(weekly_social, on=KEYS + ["week"], how="left")
    weekly = weekly.merge(weekly_quiz, on=KEYS + ["week"], how="left")
    weekly = weekly.merge(weekly_homepage, on=KEYS + ["week"], how="left")
    weekly = weekly.merge(weekly_pace, on=KEYS + ["week"], how="left")
    weekly = weekly.merge(planned_material_coverage, on=KEYS + ["week"], how="left")
    weekly = weekly.merge(pre_start_features, on=KEYS, how="left")
    zero_cols = [
        "total_clicks",
        "active_days",
        "active_days_last_7",
        "unique_sites_wk",
        "activity_diversity",
        "max_daily_clicks",
        "click_gini_wk",
        "activity_entropy_wk",
        "material_clicks",
        "material_active_days",
        "forum_clicks",
        "forum_active_days",
        "social_clicks",
        "quiz_clicks",
        "homepage_clicks",
        "pace_metadata_clicks",
        "on_schedule_clicks",
        "ahead_clicks",
        "catchup_clicks",
        "planned_material_coverage",
        "avg_material_lag",
        "pre_start_clicks",
        "pre_start_active_days",
        "days_before_start",
        "pre_start_activity_diversity",
        "pre_start_material_clicks",
        "pre_start_flag",
        "pre_start_clicks_log",
        "pre_start_material_clicks_log",
    ]
    weekly[zero_cols] = weekly[zero_cols].fillna(0)
    weekly["weekly_clicks"] = weekly["total_clicks"]
    weekly["weekly_clicks_log"] = np.log1p(weekly["weekly_clicks"])
    weekly["material_click_share"] = np.where(
        weekly["weekly_clicks"] > 0,
        weekly["material_clicks"] / weekly["weekly_clicks"],
        0,
    ).clip(0, 1)
    weekly["study_regularity_score"] = (weekly["active_days_last_7"] / 7).clip(0, 1)
    weekly["burstiness_score"] = np.where(weekly["weekly_clicks"] > 0, weekly["max_daily_clicks"] / weekly["weekly_clicks"], 0).clip(0, 1)
    weekly["forum_click_ratio"] = np.where(weekly["weekly_clicks"] > 0, weekly["forum_clicks"] / weekly["weekly_clicks"], 0).clip(0, 1)
    weekly["social_click_ratio"] = np.where(weekly["weekly_clicks"] > 0, weekly["social_clicks"] / weekly["weekly_clicks"], 0).clip(0, 1)
    weekly["content_click_ratio"] = weekly["material_click_share"]
    weekly["quiz_click_ratio"] = np.where(weekly["weekly_clicks"] > 0, weekly["quiz_clicks"] / weekly["weekly_clicks"], 0).clip(0, 1)
    weekly["homepage_dependency_ratio"] = np.where(
        weekly["weekly_clicks"] > 0, weekly["homepage_clicks"] / weekly["weekly_clicks"], 0
    ).clip(0, 1)
    weekly["on_schedule_click_ratio"] = np.where(
        weekly["pace_metadata_clicks"] > 0, weekly["on_schedule_clicks"] / weekly["pace_metadata_clicks"], 0
    ).clip(0, 1)
    weekly["ahead_click_ratio"] = np.where(
        weekly["pace_metadata_clicks"] > 0, weekly["ahead_clicks"] / weekly["pace_metadata_clicks"], 0
    ).clip(0, 1)
    weekly["catchup_click_ratio"] = np.where(
        weekly["pace_metadata_clicks"] > 0, weekly["catchup_clicks"] / weekly["pace_metadata_clicks"], 0
    ).clip(0, 1)
    weekly["pace_metadata_coverage_ratio"] = np.where(weekly["weekly_clicks"] > 0, weekly["pace_metadata_clicks"] / weekly["weekly_clicks"], 0).clip(0, 1)
    weekly["week_end_day"] = weekly["week"] * 7 - 1
    weekly["last_click_day"] = weekly.groupby(KEYS)["last_click_day"].ffill()
    weekly["days_since_last_click"] = (weekly["week_end_day"] - weekly["last_click_day"]).fillna(DAYS_SINCE_CLICK_MISSING)
    weekly["first_click_day"] = weekly.groupby(KEYS)["first_click_day_week"].transform("min").fillna(DAYS_SINCE_CLICK_MISSING)

    weekly = add_assessment_features(weekly, data)
    weekly = add_rolling_features(weekly)
    weekly = add_peer_normalized_features(weekly)
    return weekly


def build_planned_material_coverage(vle_events: pd.DataFrame, vle: pd.DataFrame, max_week: int) -> pd.DataFrame:
    planned_sites = vle[vle["week_from"].notna()][["code_module", "code_presentation", "id_site", "week_from"]].drop_duplicates()
    if planned_sites.empty:
        return vle_events[KEYS].drop_duplicates().assign(week=1, planned_material_coverage=0).iloc[0:0]

    modules = planned_sites[["code_module", "code_presentation"]].drop_duplicates().merge(
        pd.DataFrame({"week": range(1, max_week + 1)}),
        how="cross",
    )
    expected = modules.merge(planned_sites, on=["code_module", "code_presentation"], how="left")
    expected = expected[expected["week_from"] <= expected["week"]]
    expected_counts = (
        expected.groupby(["code_module", "code_presentation", "week"])
        .agg(expected_planned_sites=("id_site", "nunique"))
        .reset_index()
    )

    planned_events = vle_events[vle_events["week_from"].notna()]
    first_access = planned_events.groupby(KEYS + ["id_site"]).agg(first_access_week=("week", "min")).reset_index()
    accessed = first_access.groupby(KEYS + ["first_access_week"]).agg(accessed_planned_sites=("id_site", "nunique")).reset_index()
    accessed = accessed.rename(columns={"first_access_week": "week"})

    roster = vle_events[KEYS].drop_duplicates().merge(pd.DataFrame({"week": range(1, max_week + 1)}), how="cross")
    coverage = roster.merge(accessed, on=KEYS + ["week"], how="left")
    coverage["accessed_planned_sites"] = coverage["accessed_planned_sites"].fillna(0)
    coverage["accessed_planned_sites_cum"] = coverage.groupby(KEYS)["accessed_planned_sites"].cumsum()
    coverage = coverage.merge(expected_counts, on=["code_module", "code_presentation", "week"], how="left")
    coverage["expected_planned_sites"] = coverage["expected_planned_sites"].fillna(0)
    coverage["planned_material_coverage"] = np.where(
        coverage["expected_planned_sites"] > 0,
        coverage["accessed_planned_sites_cum"] / coverage["expected_planned_sites"],
        0,
    ).clip(0, 1)
    return coverage[KEYS + ["week", "planned_material_coverage"]]


def add_assessment_features(weekly: pd.DataFrame, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    assessments = data["assessments"].copy()
    assessments = assessments[assessments["date"].notna()]
    assessments = assessments[assessments["assessment_type"] != "Exam"]
    assessments["is_low_weight"] = (assessments["weight"] <= 10).astype(int)
    assessments = add_week(assessments, "date")

    due = (
        assessments.groupby(["code_module", "code_presentation", "week"])
        .agg(
            assessments_due=("id_assessment", "nunique"),
            low_weight_due=("is_low_weight", "sum"),
        )
        .reset_index()
    )
    submissions = data["student_assessment"].merge(
        assessments[["id_assessment", "code_module", "code_presentation", "date", "is_low_weight"]],
        on="id_assessment",
        how="inner",
    )
    submissions = submissions[submissions["date_submitted"].notna()]
    submissions["submission_week"] = np.floor(submissions["date_submitted"] / 7).astype(int) + 1
    submissions["is_late"] = (submissions["date_submitted"] > submissions["date"]).astype(int)
    submissions["is_early"] = (submissions["date_submitted"] < submissions["date"]).astype(int)
    submissions["days_before_due"] = (submissions["date"] - submissions["date_submitted"]).clip(lower=0)
    submissions["low_weight_submitted"] = submissions["is_low_weight"]
    submitted = (
        submissions.groupby(KEYS + ["submission_week"])
        .agg(
            submitted_count=("id_assessment", "nunique"),
            late_count=("is_late", "sum"),
            early_submission_count=("is_early", "sum"),
            avg_days_before_due_week=("days_before_due", "mean"),
            low_weight_submitted=("low_weight_submitted", "sum"),
            avg_score_week=("score", "mean"),
        )
        .reset_index()
        .rename(columns={"submission_week": "week"})
    )
    prep = build_assessment_preparation_features(data, assessments, submissions)

    weekly = weekly.merge(due, on=["code_module", "code_presentation", "week"], how="left")
    weekly = weekly.merge(submitted, on=KEYS + ["week"], how="left")
    weekly = weekly.merge(prep, on=KEYS + ["week"], how="left")
    fill_cols = [
        "assessments_due",
        "low_weight_due",
        "submitted_count",
        "late_count",
        "early_submission_count",
        "avg_days_before_due_week",
        "low_weight_submitted",
        "pre_assessment_clicks_7d",
        "cram_clicks_2d",
        "prep_started_days_before_due_week",
        "post_bad_score_recovery_clicks",
    ]
    weekly[fill_cols] = weekly[
        fill_cols
    ].fillna(0)
    weekly["cram_ratio_week"] = np.where(
        weekly["pre_assessment_clicks_7d"] > 0,
        weekly["cram_clicks_2d"] / weekly["pre_assessment_clicks_7d"],
        0,
    ).clip(0, 1)
    weekly["assessments_due_cum"] = weekly.groupby(KEYS)["assessments_due"].cumsum()
    weekly["low_weight_due_cum"] = weekly.groupby(KEYS)["low_weight_due"].cumsum()
    weekly["submitted_count_cum"] = weekly.groupby(KEYS)["submitted_count"].cumsum()
    weekly["late_submission_count"] = weekly.groupby(KEYS)["late_count"].cumsum()
    weekly["early_submission_count_cum"] = weekly.groupby(KEYS)["early_submission_count"].cumsum()
    weekly["low_weight_submitted_cum"] = weekly.groupby(KEYS)["low_weight_submitted"].cumsum()
    weekly["pre_assessment_clicks_7d_cum"] = weekly.groupby(KEYS)["pre_assessment_clicks_7d"].cumsum()
    weekly["post_bad_score_recovery_clicks_cum"] = weekly.groupby(KEYS)["post_bad_score_recovery_clicks"].cumsum()
    weekly["missed_assessments_cum"] = (weekly["assessments_due_cum"] - weekly["submitted_count_cum"]).clip(lower=0)
    weekly["assessment_submitted_ratio"] = np.where(
        weekly["assessments_due_cum"] > 0,
        weekly["submitted_count_cum"] / weekly["assessments_due_cum"],
        1,
    ).clip(0, 1)
    weekly["low_weight_completion_ratio"] = np.where(
        weekly["low_weight_due_cum"] > 0,
        weekly["low_weight_submitted_cum"] / weekly["low_weight_due_cum"],
        1,
    ).clip(0, 1)
    weekly["punctuality_ratio"] = np.where(
        weekly["submitted_count_cum"] > 0,
        1 - (weekly["late_submission_count"] / weekly["submitted_count_cum"]),
        1,
    ).clip(0, 1)
    weekly["starting_early_score"] = np.where(
        weekly["submitted_count_cum"] > 0,
        weekly.groupby(KEYS)["avg_days_before_due_week"].cumsum() / weekly["submitted_count_cum"],
        0,
    )
    weekly["prep_started_days_before_due"] = np.where(
        weekly["submitted_count_cum"] > 0,
        weekly.groupby(KEYS)["prep_started_days_before_due_week"].cumsum() / weekly["submitted_count_cum"],
        0,
    )
    weekly["avg_score_so_far"] = weekly.groupby(KEYS)["avg_score_week"].ffill()
    return weekly


def build_assessment_preparation_features(
    data: dict[str, pd.DataFrame],
    assessments: pd.DataFrame,
    submissions: pd.DataFrame,
) -> pd.DataFrame:
    events = data["student_vle"].merge(data["courses"], on=["code_module", "code_presentation"], how="left")
    events = events[events["date"].between(0, events["module_presentation_length"])]
    prep_rows = []
    for assessment in assessments.itertuples(index=False):
        due_date = int(assessment.date)
        due_week = int(assessment.week)
        module_mask = (events["code_module"] == assessment.code_module) & (
            events["code_presentation"] == assessment.code_presentation
        )
        window = events[module_mask & events["date"].between(due_date - 7, due_date)]
        if window.empty:
            continue
        agg = window.groupby(KEYS).agg(
            pre_assessment_clicks_7d=("sum_click", "sum"),
            first_prep_day=("date", "min"),
        )
        cram = window[window["date"].between(due_date - 2, due_date)].groupby(KEYS).agg(cram_clicks_2d=("sum_click", "sum"))
        agg = agg.merge(cram, left_index=True, right_index=True, how="left").fillna({"cram_clicks_2d": 0})
        agg["prep_started_days_before_due_week"] = (due_date - agg["first_prep_day"]).clip(lower=0)
        agg["week"] = due_week
        prep_rows.append(agg.reset_index()[KEYS + ["week", "pre_assessment_clicks_7d", "cram_clicks_2d", "prep_started_days_before_due_week"]])

    if prep_rows:
        prep = pd.concat(prep_rows, ignore_index=True)
        prep = prep.groupby(KEYS + ["week"], as_index=False).agg(
            pre_assessment_clicks_7d=("pre_assessment_clicks_7d", "sum"),
            cram_clicks_2d=("cram_clicks_2d", "sum"),
            prep_started_days_before_due_week=("prep_started_days_before_due_week", "mean"),
        )
    else:
        prep = data["student_info"][KEYS].drop_duplicates().assign(
            week=1,
            pre_assessment_clicks_7d=0,
            cram_clicks_2d=0,
            prep_started_days_before_due_week=0,
        ).iloc[0:0]

    low_scores = submissions[submissions["score"].notna() & (submissions["score"] < 50)][KEYS + ["date_submitted"]].copy()
    if not low_scores.empty:
        low_scores["week"] = np.floor(low_scores["date_submitted"] / 7).astype(int) + 1
        recovery_events = events.merge(low_scores, on=KEYS, how="inner")
        recovery_events = recovery_events[
            recovery_events["date"].between(recovery_events["date_submitted"] + 1, recovery_events["date_submitted"] + 7)
        ]
        recovery = recovery_events.groupby(KEYS + ["week"], as_index=False).agg(
            post_bad_score_recovery_clicks=("sum_click", "sum")
        )
    else:
        recovery = prep[KEYS + ["week"]].copy()
        recovery["post_bad_score_recovery_clicks"] = 0

    return prep.merge(recovery, on=KEYS + ["week"], how="outer")


def add_rolling_features(weekly: pd.DataFrame) -> pd.DataFrame:
    weekly = weekly.sort_values(KEYS + ["week"]).copy()
    group = weekly.groupby(KEYS, group_keys=False)

    def slope_last_3(values: pd.Series) -> pd.Series:
        out = []
        arr = values.to_numpy()
        for i in range(len(arr)):
            window = arr[max(0, i - 2) : i + 1]
            if len(window) < 2:
                out.append(0.0)
            else:
                out.append(float(np.polyfit(np.arange(len(window)), window, 1)[0]))
        return pd.Series(out, index=values.index)

    weekly["click_trend_3w"] = group["weekly_clicks"].transform(slope_last_3)
    weekly["active_weeks_cum"] = group["weekly_clicks"].transform(lambda s: (s > 0).cumsum())
    weekly["zero_click_weeks_cum"] = group["weekly_clicks"].transform(lambda s: (s == 0).cumsum())
    weekly["consistency_rate"] = weekly["active_weeks_cum"] / weekly["week"]
    weekly["recent_clicks_2w"] = group["weekly_clicks"].transform(lambda s: s.rolling(2, min_periods=1).sum())
    weekly["recent_material_active_days_2w"] = group["material_active_days"].transform(lambda s: s.rolling(2, min_periods=1).sum())
    weekly["week_to_week_volatility"] = group["weekly_clicks"].transform(lambda s: s.rolling(4, min_periods=2).std()).fillna(0)
    prev_clicks = group["weekly_clicks"].shift(1).fillna(0)
    weekly["recent_activity_drop"] = np.where(
        prev_clicks > 0,
        (prev_clicks - weekly["weekly_clicks"]) / prev_clicks,
        0,
    ).clip(-1, 1)
    weekly["inactive_last_7_days"] = (weekly["days_since_last_click"] >= 7).astype(int)
    weekly["inactive_last_14_days"] = (weekly["days_since_last_click"] >= 14).astype(int)
    weekly["longest_inactive_gap_so_far"] = group["days_since_last_click"].cummax()
    return weekly


def percentile_by_context(df: pd.DataFrame, col: str, ascending: bool = True) -> pd.Series:
    return df.groupby(["code_module", "code_presentation", "week"])[col].rank(pct=True, ascending=ascending)


def add_peer_normalized_features(weekly: pd.DataFrame) -> pd.DataFrame:
    weekly = weekly.copy()
    weekly["pre_start_proactivity_raw"] = (
        0.40 * percentile_by_context(weekly, "pre_start_clicks_log")
        + 0.35 * percentile_by_context(weekly, "days_before_start")
        + 0.25 * percentile_by_context(weekly, "pre_start_active_days")
    )
    weekly["p_pre_start_proactivity"] = percentile_by_context(weekly, "pre_start_proactivity_raw")
    weekly["weekly_clicks_norm"] = percentile_by_context(weekly, "weekly_clicks_log")
    weekly["p_clicks"] = weekly["weekly_clicks_norm"]
    weekly["p_active_days"] = percentile_by_context(weekly, "active_days_last_7")
    weekly["p_studiousness"] = percentile_by_context(weekly, "material_active_days")
    weekly["p_activity_diversity"] = percentile_by_context(weekly, "activity_diversity")
    weekly["p_material_click_share"] = percentile_by_context(weekly, "material_click_share")
    weekly["p_study_regularity"] = percentile_by_context(weekly, "study_regularity_score")
    weekly["p_low_burstiness"] = percentile_by_context(weekly, "burstiness_score", ascending=False)
    weekly["p_recency"] = percentile_by_context(weekly, "days_since_last_click", ascending=False)
    weekly["p_trend"] = percentile_by_context(weekly, "click_trend_3w")
    weekly["p_submission_completion"] = percentile_by_context(weekly, "assessment_submitted_ratio")
    weekly["p_punctuality"] = percentile_by_context(weekly, "punctuality_ratio")
    weekly["p_low_weight_completion"] = percentile_by_context(weekly, "low_weight_completion_ratio")
    return weekly


def make_enrollment_split(weekly: pd.DataFrame) -> pd.DataFrame:
    week6 = weekly[weekly["week"] == WEEK_CUTOFF][KEYS + ["final_result"]].drop_duplicates().copy()
    week6 = week6[week6["final_result"].isin(RISK_LABELS | SUCCESS_LABELS)]
    y = week6["final_result"].isin(RISK_LABELS).astype(int)
    _, test_idx = train_test_split(week6.index, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    week6["split"] = "train"
    week6.loc[test_idx, "split"] = "test"
    week6["risk_label"] = y
    week6.to_csv(OUT_DIR / "enrollment_train_test_split.csv", index=False)
    return week6


def feature_rationale(weekly: pd.DataFrame) -> pd.DataFrame:
    week6 = weekly[weekly["week"] == WEEK_CUTOFF].copy()
    week6["risk"] = week6["final_result"].isin(RISK_LABELS).astype(int)
    rows = []
    feature_catalog = task1_feature_catalog()
    for col in feature_catalog["feature"]:
        if col not in week6.columns:
            continue
        rows.append(
            {
                "feature": col,
                "bucket": feature_catalog.loc[feature_catalog["feature"] == col, "bucket"].iloc[0],
                "description": feature_catalog.loc[feature_catalog["feature"] == col, "description"].iloc[0],
                "withdraw_fail_mean": week6.loc[week6["risk"] == 1, col].mean(),
                "pass_distinction_mean": week6.loc[week6["risk"] == 0, col].mean(),
                "risk_correlation": week6[[col, "risk"]].corr(numeric_only=True).iloc[0, 1],
            }
        )
    rationale = pd.DataFrame(rows)
    rationale.to_csv(OUT_DIR / "feature_rationale_week6.csv", index=False)
    return rationale


def task1_feature_catalog() -> pd.DataFrame:
    rows = [
        ("Activity volume", "weekly_clicks_norm", "Peer-normalized weekly activity volume."),
        ("Activity volume", "weekly_clicks", "Raw weekly click count used for diagnostic plots, not as an unscaled score input."),
        ("Activity volume", "unique_sites_wk", "Breadth of distinct VLE sites touched in the week."),
        ("Consistency and rhythm", "active_days_last_7", "Count of distinct active days in the week."),
        ("Consistency and rhythm", "study_regularity_score", "Active days divided by seven."),
        ("Consistency and rhythm", "click_gini_wk", "Inequality of daily clicks; high values suggest binge-like activity."),
        ("Consistency and rhythm", "burstiness_score", "Share of weekly clicks concentrated on the busiest day."),
        ("Consistency and rhythm", "week_to_week_volatility", "Rolling four-week standard deviation of weekly clicks."),
        ("Consistency and rhythm", "zero_click_weeks_cum", "Cumulative weeks with no VLE activity."),
        ("Recency and disengagement gaps", "days_since_last_click", "Days since the student's most recent VLE interaction."),
        ("Recency and disengagement gaps", "longest_inactive_gap_so_far", "Largest observed inactivity gap up to that week."),
        ("Recency and disengagement gaps", "inactive_last_7_days", "Flag that the student has been absent for at least seven days."),
        ("Recency and disengagement gaps", "inactive_last_14_days", "Flag that the student has been absent for at least fourteen days."),
        ("Recency and disengagement gaps", "recent_activity_drop", "Drop from the previous week's clicks to the current week."),
        ("Recency and disengagement gaps", "recent_clicks_2w", "Total clicks over the current and prior week."),
        ("Resource diversity and intent", "activity_diversity", "Distinct VLE activity types used in the week."),
        ("Resource diversity and intent", "activity_entropy_wk", "Balance of clicks across activity types."),
        ("Resource diversity and intent", "material_active_days", "Number of days with learning-material activity."),
        ("Resource diversity and intent", "material_click_share", "Share of clicks on direct learning materials."),
        ("Resource diversity and intent", "recent_material_active_days_2w", "Recent material-study rhythm over two weeks."),
        ("Resource diversity and intent", "forum_click_ratio", "Share of clicks on forum activity."),
        ("Resource diversity and intent", "forum_active_days", "Days with discussion-board activity."),
        ("Resource diversity and intent", "social_click_ratio", "Share of clicks on social or collaborative activity."),
        ("Resource diversity and intent", "content_click_ratio", "Share of clicks on content/resource/page-style material."),
        ("Resource diversity and intent", "quiz_click_ratio", "Share of clicks on quiz or external quiz activity."),
        ("Resource diversity and intent", "homepage_dependency_ratio", "Share of clicks on homepage navigation."),
        ("Course-pace alignment", "pre_start_proactivity_raw", "Composite of pre-course clicks, days early, and active days."),
        ("Course-pace alignment", "pre_start_clicks", "Clicks before official module day 0."),
        ("Course-pace alignment", "days_before_start", "How many days before start the student first engaged."),
        ("Course-pace alignment", "pre_start_active_days", "Distinct pre-start active days."),
        ("Course-pace alignment", "on_schedule_click_ratio", "Share of metadata-covered clicks within planned weeks."),
        ("Course-pace alignment", "ahead_click_ratio", "Share of metadata-covered clicks before planned week_from."),
        ("Course-pace alignment", "catchup_click_ratio", "Share of metadata-covered clicks after planned week_to."),
        ("Course-pace alignment", "avg_material_lag", "Average current week minus planned week_from for metadata-covered activity."),
        ("Course-pace alignment", "planned_material_coverage", "Cumulative share of planned materials accessed."),
        ("Course-pace alignment", "pace_metadata_coverage_ratio", "Share of clicks where week_from/week_to metadata exists."),
        ("Assessment behavior and recovery", "assessment_submitted_ratio", "Cumulative due assessment completion."),
        ("Assessment behavior and recovery", "missed_assessments_cum", "Cumulative due assessments not submitted."),
        ("Assessment behavior and recovery", "late_submission_count", "Cumulative late submissions."),
        ("Assessment behavior and recovery", "punctuality_ratio", "Share of submitted work that was not late."),
        ("Assessment behavior and recovery", "low_weight_completion_ratio", "Completion of low-weight assignments."),
        ("Assessment behavior and recovery", "starting_early_score", "Average days submitted before due date."),
        ("Assessment behavior and recovery", "pre_assessment_clicks_7d_cum", "Cumulative clicks in seven-day windows before assessment due dates."),
        ("Assessment behavior and recovery", "cram_ratio_week", "Share of pre-assessment clicks concentrated in the final two days."),
        ("Assessment behavior and recovery", "prep_started_days_before_due", "How early preparation started before due dates."),
        ("Assessment behavior and recovery", "post_bad_score_recovery_clicks_cum", "Clicks after low-scoring assessments."),
        ("Overall score", "engagement_score", "Final 0-100 weekly engagement score."),
    ]
    catalog = pd.DataFrame(rows, columns=["bucket", "feature", "description"])
    catalog.to_csv(OUT_DIR / "task1_feature_catalog.csv", index=False)
    return catalog
