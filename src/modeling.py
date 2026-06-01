from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

from data_loader import (
    CAPACITY_QUANTILE,
    KEYS,
    MODEL_DIR,
    OUT_DIR,
    RANDOM_STATE,
    RISK_LABELS,
    SUCCESS_LABELS,
    URGENT_RECALL_FLOOR,
    WEEK_CUTOFF,
)

F2_BETA = 2.0
PROFILE_ACTIVITY_TYPES = ["resource", "oucontent", "questionnaire", "dataplus", "ouwiki", "forumng"]
EDUCATION_ORDER = {
    "No Formal quals": 0,
    "Lower Than A Level": 1,
    "A Level or Equivalent": 2,
    "HE Qualification": 3,
    "Post Graduate Qualification": 4,
}
AGE_ORDER = {"0-35": 0, "35-55": 1, "55<=": 2}


def _imd_midpoint(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.replace("%", "", regex=False)
    parts = cleaned.str.extract(r"(?P<low>\d+)-(?P<high>\d+)")
    midpoint = (pd.to_numeric(parts["low"], errors="coerce") + pd.to_numeric(parts["high"], errors="coerce")) / 2
    return midpoint


def build_week6_activity_mix(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Specific Week 6 activity mix features, using only VLE events up to day 41."""
    vle_cols = ["id_site", "code_module", "code_presentation", "activity_type"]
    events = data["student_vle"].merge(data["vle"][vle_cols], on=["id_site", "code_module", "code_presentation"], how="left")
    events = events[events["date"].between(0, WEEK_CUTOFF * 7 - 1)]
    clicks = (
        events[events["activity_type"].isin(PROFILE_ACTIVITY_TYPES)]
        .groupby(KEYS + ["activity_type"])["sum_click"]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )
    roster = data["student_info"][KEYS].drop_duplicates()
    clicks = roster.merge(clicks, on=KEYS, how="left")
    for activity_type in PROFILE_ACTIVITY_TYPES:
        if activity_type not in clicks:
            clicks[activity_type] = 0
        clicks[activity_type] = clicks[activity_type].fillna(0)
        clicks[f"{activity_type}_clicks_6"] = clicks[activity_type]
    clicks["profile_activity_clicks_6"] = clicks[[f"{a}_clicks_6" for a in PROFILE_ACTIVITY_TYPES]].sum(axis=1)
    for activity_type in PROFILE_ACTIVITY_TYPES:
        clicks[f"{activity_type}_ratio_6"] = np.where(
            clicks["profile_activity_clicks_6"] > 0,
            clicks[f"{activity_type}_clicks_6"] / clicks["profile_activity_clicks_6"],
            0,
        ).clip(0, 1)
    keep_cols = KEYS + ["profile_activity_clicks_6"] + [
        col for activity_type in PROFILE_ACTIVITY_TYPES for col in (f"{activity_type}_clicks_6", f"{activity_type}_ratio_6")
    ]
    return clicks[keep_cols]


def add_profile_features(model_df: pd.DataFrame) -> pd.DataFrame:
    out = model_df.copy()
    out["education_level_ord"] = out["highest_education"].map(EDUCATION_ORDER).fillna(EDUCATION_ORDER["A Level or Equivalent"])
    out["age_band_ord"] = out["age_band"].map(AGE_ORDER).fillna(AGE_ORDER["0-35"])
    out["mature_student_flag"] = out["age_band"].isin(["35-55", "55<="]).astype(int)
    out["young_student_flag"] = out["age_band"].eq("0-35").astype(int)
    out["imd_midpoint"] = _imd_midpoint(out["imd_band"])
    out["imd_missing_flag"] = out["imd_midpoint"].isna().astype(int)
    out["imd_midpoint"] = out["imd_midpoint"].fillna(out["imd_midpoint"].median())
    out["high_imd_flag"] = (out["imd_midpoint"] >= 60).astype(int)
    out["attempt_count_capped"] = out["num_of_prev_attempts"].clip(upper=3)
    out["multiple_attempts_flag"] = (out["num_of_prev_attempts"] >= 2).astype(int)
    credit_median = out.groupby(["code_module", "code_presentation"])["studied_credits"].transform("median")
    credit_std = out.groupby(["code_module", "code_presentation"])["studied_credits"].transform("std").replace(0, np.nan)
    out["credit_load_z_by_module"] = ((out["studied_credits"] - credit_median) / credit_std).fillna(0)
    out["credits_per_active_day_6"] = out["studied_credits"] / np.maximum(out["active_days_last_7_6"], 1)
    out["higher_education_flag"] = (out["education_level_ord"] >= EDUCATION_ORDER["HE Qualification"]).astype(int)
    out["lower_education_flag"] = (out["education_level_ord"] <= EDUCATION_ORDER["Lower Than A Level"]).astype(int)
    out["prepared_mature_high_imd_profile"] = (
        out["higher_education_flag"].eq(1) & out["mature_student_flag"].eq(1) & out["high_imd_flag"].eq(1)
    ).astype(int)
    out["younger_lower_education_low_imd_profile"] = (
        out["lower_education_flag"].eq(1) & out["young_student_flag"].eq(1) & out["low_imd_flag"].eq(1)
    ).astype(int)
    out["profile_group"] = np.select(
        [
            out["prepared_mature_high_imd_profile"].eq(1),
            out["younger_lower_education_low_imd_profile"].eq(1),
            out["higher_education_flag"].eq(1),
            out["low_imd_flag"].eq(1),
        ],
        ["prepared_mature_high_imd", "younger_lower_education_low_imd", "higher_education", "low_imd"],
        default="mixed_profile",
    )
    return out


def add_week6_archetype_features(model_df: pd.DataFrame) -> pd.DataFrame:
    out = model_df.copy()
    engagement = out["engagement_score_6"].fillna(out["engagement_score_6"].median())
    high_engagement = engagement >= 60
    low_engagement = engagement < 40
    low_diversity = out["activity_diversity_6"].fillna(0) <= 2
    low_material = out["material_active_days_6"].fillna(0) <= 2
    good_completion = out["assessment_submitted_ratio_6"].fillna(0) >= 0.8
    good_score_so_far = out["avg_score_so_far_6"].fillna(0) >= 70
    high_burst = out["burstiness_score_6"].fillna(0) >= 0.65
    high_volatility = out["week_to_week_volatility_6"].fillna(0) >= out["week_to_week_volatility_6"].fillna(0).quantile(0.70)
    late_or_cram = (out["late_submission_count_6"].fillna(0) > 0) | (out["cram_ratio_6"].fillna(0) >= 0.6)
    inactive_or_missing = (out["inactive_last_14_days_6"].fillna(0) == 1) | (out["zero_click_weeks_6"].fillna(0) >= 3)

    out["week6_archetype"] = np.select(
        [
            inactive_or_missing & low_engagement,
            (out["repeat_attempt_flag"] == 1) & low_engagement,
            (out["high_credit_load_flag"] == 1) & (engagement < 50),
            late_or_cram & good_score_so_far,
            high_burst | high_volatility,
            good_completion & low_diversity & low_material,
            (out["pre_start_flag_6"] == 1) & high_engagement & (out["activity_diversity_6"].fillna(0) >= 3),
            high_engagement & (out["study_regularity_score_6"].fillna(0) >= 0.35),
        ],
        [
            "early_disengager",
            "struggling_repeater",
            "high_workload_risk",
            "perfectionist_procrastinator",
            "sporadic_burst_engager",
            "surface_compliance_engager",
            "proactive_engager",
            "steady_engager",
        ],
        default="mixed_engagement",
    )
    out["archetype_early_disengager_flag"] = out["week6_archetype"].eq("early_disengager").astype(int)
    out["archetype_burst_flag"] = out["week6_archetype"].eq("sporadic_burst_engager").astype(int)
    out["archetype_compliance_flag"] = out["week6_archetype"].eq("surface_compliance_engager").astype(int)
    out["archetype_proactive_flag"] = out["week6_archetype"].eq("proactive_engager").astype(int)
    return out


def add_interaction_features(model_df: pd.DataFrame) -> pd.DataFrame:
    out = model_df.copy()
    out["credits_x_low_engagement"] = out["studied_credits"] * (100 - out["engagement_score_6"].fillna(50)) / 100
    out["education_x_assessment_completion"] = out["education_level_ord"] * out["assessment_submitted_ratio_6"].fillna(0)
    out["age_x_study_regularity"] = out["age_band_ord"] * out["study_regularity_score_6"].fillna(0)
    out["attempts_x_low_engagement"] = out["attempt_count_capped"] * (100 - out["engagement_score_6"].fillna(50)) / 100
    out["low_imd_x_resource_ratio"] = out["low_imd_flag"] * out.get("resource_ratio_6", 0)
    out["higher_edu_x_oucontent_ratio"] = out["higher_education_flag"] * out.get("oucontent_ratio_6", 0)
    out["younger_lower_edu_x_resource_ratio"] = out["younger_lower_education_low_imd_profile"] * out.get("resource_ratio_6", 0)
    out["prepared_profile_x_questionnaire_ratio"] = out["prepared_mature_high_imd_profile"] * out.get("questionnaire_ratio_6", 0)
    return out


def add_train_only_risk_priors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    train = out[out["split"] == "train"]
    global_risk = train["risk_label"].mean()
    for col in ["week6_archetype", "profile_group"]:
        mapping = train.groupby(col)["risk_label"].mean()
        out[f"{col}_risk_prior"] = out[col].map(mapping).fillna(global_risk)
    archetype_summary = (
        train.groupby("week6_archetype")
        .agg(enrolments=("risk_label", "size"), train_withdraw_fail_rate=("risk_label", "mean"))
        .reset_index()
        .sort_values("train_withdraw_fail_rate", ascending=False)
    )
    archetype_summary.to_csv(OUT_DIR / "week6_archetype_risk_rates.csv", index=False)
    profile_summary = (
        train.groupby("profile_group")
        .agg(enrolments=("risk_label", "size"), train_withdraw_fail_rate=("risk_label", "mean"))
        .reset_index()
        .sort_values("train_withdraw_fail_rate", ascending=False)
    )
    profile_summary.to_csv(OUT_DIR / "profile_group_risk_rates.csv", index=False)
    return out


def write_task2_feature_catalog() -> None:
    rows = [
        ("Engagement", "engagement_score_6", "Latest leakage-safe Week 6 engagement score from Task 1."),
        ("Engagement", "week6_archetype", "Most likely Week 6 behavioural archetype using only early engagement patterns."),
        ("Engagement", "week6_archetype_risk_prior", "Train-only risk rate for each Week 6 archetype."),
        ("Preparedness", "education_level_ord", "Ordered education ladder to capture prior academic preparedness."),
        ("Preparedness", "higher_education_flag", "HE-or-higher indicator, motivated by outcome gradients by education level."),
        ("Socioeconomic context", "imd_midpoint", "Numeric deprivation band midpoint; used as context, not causal judgement."),
        ("Socioeconomic context", "low_imd_flag", "Lower IMD context flag."),
        ("Life stage", "age_band_ord", "Ordered age band."),
        ("Life stage", "mature_student_flag", "Mature student support-context flag."),
        ("Workload", "credit_load_z_by_module", "Credits normalized against peers in the same module-presentation."),
        ("Workload", "credits_x_low_engagement", "High workload combined with low engagement."),
        ("Prior struggle", "repeat_attempt_flag", "Student has attempted the module before."),
        ("Prior struggle", "attempts_x_low_engagement", "Repeat attempts combined with low current engagement."),
        ("Resource mix", "resource_ratio_6", "Share of selected Week 6 activity clicks on resource pages."),
        ("Resource mix", "oucontent_ratio_6", "Share of selected Week 6 activity clicks on OU content."),
        ("Resource mix", "questionnaire_ratio_6", "Share of selected Week 6 activity clicks on questionnaire pages."),
        ("Resource mix", "dataplus_ratio_6", "Share of selected Week 6 activity clicks on dataplus pages."),
        ("Resource mix", "ouwiki_ratio_6", "Share of selected Week 6 activity clicks on wiki collaboration."),
        ("Interaction", "low_imd_x_resource_ratio", "Resource-use pattern for lower IMD students."),
        ("Interaction", "higher_edu_x_oucontent_ratio", "OU-content usage among higher-education students."),
        ("Interaction", "education_x_assessment_completion", "Preparedness plus assessment follow-through."),
        ("Interaction", "profile_group_risk_prior", "Train-only risk rate for broad profile groups."),
    ]
    pd.DataFrame(rows, columns=["bucket", "feature", "reason"]).to_csv(OUT_DIR / "task2_feature_catalog.csv", index=False)


def build_week6_model_dataset(weekly: pd.DataFrame, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    write_task2_feature_catalog()
    week6 = weekly[weekly["week"] <= WEEK_CUTOFF].groupby(KEYS).agg(
        weekly_clicks_6=("weekly_clicks", "sum"),
        weekly_clicks_norm_6=("weekly_clicks_norm", "mean"),
        active_days_last_7_6=("active_days_last_7", "sum"),
        study_regularity_score_6=("study_regularity_score", "mean"),
        click_gini_6=("click_gini_wk", "mean"),
        burstiness_score_6=("burstiness_score", "mean"),
        week_to_week_volatility_6=("week_to_week_volatility", "last"),
        material_active_days_6=("material_active_days", "sum"),
        recent_material_active_days_2w_6=("recent_material_active_days_2w", "last"),
        material_click_share_6=("material_click_share", "mean"),
        content_click_ratio_6=("content_click_ratio", "mean"),
        quiz_click_ratio_6=("quiz_click_ratio", "mean"),
        homepage_dependency_ratio_6=("homepage_dependency_ratio", "mean"),
        forum_clicks_6=("forum_clicks", "sum"),
        forum_active_days_6=("forum_active_days", "sum"),
        forum_click_ratio_6=("forum_click_ratio", "mean"),
        social_click_ratio_6=("social_click_ratio", "mean"),
        unique_sites_6=("unique_sites_wk", "sum"),
        activity_entropy_6=("activity_entropy_wk", "mean"),
        activity_diversity_6=("activity_diversity", "mean"),
        recent_clicks_2w_6=("recent_clicks_2w", "last"),
        recent_activity_drop_6=("recent_activity_drop", "last"),
        click_trend_3w=("click_trend_3w", "last"),
        consistency_rate_6=("consistency_rate", "last"),
        zero_click_weeks_6=("zero_click_weeks_cum", "last"),
        days_since_last_click_6=("days_since_last_click", "last"),
        longest_inactive_gap_6=("longest_inactive_gap_so_far", "last"),
        inactive_last_7_days_6=("inactive_last_7_days", "last"),
        inactive_last_14_days_6=("inactive_last_14_days", "last"),
        first_click_day_6=("first_click_day", "first"),
        on_schedule_click_ratio_6=("on_schedule_click_ratio", "mean"),
        ahead_click_ratio_6=("ahead_click_ratio", "mean"),
        catchup_click_ratio_6=("catchup_click_ratio", "mean"),
        avg_material_lag_6=("avg_material_lag", "mean"),
        planned_material_coverage_6=("planned_material_coverage", "last"),
        pace_metadata_coverage_ratio_6=("pace_metadata_coverage_ratio", "mean"),
        assessment_submitted_ratio_6=("assessment_submitted_ratio", "last"),
        missed_assessments_6=("missed_assessments_cum", "last"),
        submitted_count_6=("submitted_count_cum", "last"),
        late_submission_count_6=("late_submission_count", "last"),
        early_submission_count_6=("early_submission_count_cum", "last"),
        pre_assessment_clicks_7d_6=("pre_assessment_clicks_7d_cum", "last"),
        cram_ratio_6=("cram_ratio_week", "mean"),
        prep_started_days_before_due_6=("prep_started_days_before_due", "last"),
        post_bad_score_recovery_clicks_6=("post_bad_score_recovery_clicks_cum", "last"),
        punctuality_ratio_6=("punctuality_ratio", "last"),
        low_weight_completion_ratio_6=("low_weight_completion_ratio", "last"),
        starting_early_score_6=("starting_early_score", "last"),
        avg_score_so_far_6=("avg_score_so_far", "last"),
        engagement_score_6=("engagement_score", "last"),
        pre_start_clicks_6=("pre_start_clicks", "first"),
        pre_start_clicks_log_6=("pre_start_clicks_log", "first"),
        pre_start_active_days_6=("pre_start_active_days", "first"),
        days_before_start_6=("days_before_start", "first"),
        pre_start_material_clicks_6=("pre_start_material_clicks", "first"),
        pre_start_flag_6=("pre_start_flag", "first"),
        pre_start_proactivity_6=("pre_start_proactivity_raw", "first"),
        final_result=("final_result", "first"),
    ).reset_index()
    week6["module_code"] = week6["code_module"]

    demographics = data["student_info"][KEYS + [
        "gender",
        "highest_education",
        "imd_band",
        "age_band",
        "num_of_prev_attempts",
        "studied_credits",
        "disability",
    ]]
    activity_mix = build_week6_activity_mix(data)
    model_df = week6.merge(demographics, on=KEYS, how="left")
    model_df = model_df.merge(activity_mix, on=KEYS, how="left")
    model_df = model_df[model_df["final_result"].isin(RISK_LABELS | SUCCESS_LABELS)].copy()
    activity_cols = [c for c in model_df.columns if c.endswith("_clicks_6") or c.endswith("_ratio_6")]
    model_df[activity_cols] = model_df[activity_cols].fillna(0)
    model_df["repeat_attempt_flag"] = (model_df["num_of_prev_attempts"] > 0).astype(int)
    model_df["high_credit_load_flag"] = (model_df["studied_credits"] > model_df["studied_credits"].median()).astype(int)
    model_df["low_imd_flag"] = model_df["imd_band"].isin(["0-10%", "10-20%", "20-30%"]).astype(int)
    model_df["active_week_ratio_6"] = 1 - (model_df["zero_click_weeks_6"] / WEEK_CUTOFF)
    model_df["early_submission_ratio_6"] = np.where(
        model_df["submitted_count_6"] > 0,
        model_df["early_submission_count_6"] / model_df["submitted_count_6"],
        0,
    )
    model_df["high_credit_low_engagement"] = (
        model_df["high_credit_load_flag"]
        * (model_df["engagement_score_6"] < model_df["engagement_score_6"].median()).astype(int)
    )
    model_df["motivation_proxy"] = (
        model_df["low_imd_flag"]
        * model_df["high_credit_load_flag"]
        * model_df["engagement_score_6"].fillna(model_df["engagement_score_6"].median())
    )
    model_df = add_profile_features(model_df)
    model_df = add_week6_archetype_features(model_df)
    model_df = add_interaction_features(model_df)
    model_df["risk_label"] = model_df["final_result"].isin(RISK_LABELS).astype(int)
    return model_df


def metrics_at_threshold(y_true: pd.Series, prob: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, pred, beta=F2_BETA)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "alerts": int(pred.sum()),
        "alert_rate": float(pred.mean()),
        "confusion_matrix": cm.tolist(),
    }


def choose_thresholds(y_train: pd.Series, train_prob: np.ndarray) -> dict[str, float]:
    precision, recall, thresholds = precision_recall_curve(y_train, train_prob)
    f1 = (2 * precision * recall) / np.maximum((precision + recall), 1e-9)
    f2 = (5 * precision * recall) / np.maximum((4 * precision + recall), 1e-9)
    balanced_f1 = float(thresholds[np.nanargmax(f1[:-1])])
    watchlist = float(thresholds[np.nanargmax(f2[:-1])])
    candidates = pd.DataFrame({"threshold": thresholds, "precision": precision[:-1], "recall": recall[:-1]})
    urgent_candidates = candidates[candidates["recall"] >= URGENT_RECALL_FLOOR]
    urgent = watchlist if urgent_candidates.empty else float(urgent_candidates.sort_values("threshold", ascending=False).iloc[0]["threshold"])
    capacity_10 = float(np.quantile(train_prob, 0.90))
    capacity_20 = float(np.quantile(train_prob, CAPACITY_QUANTILE))
    capacity_30 = float(np.quantile(train_prob, 0.70))
    return {
        "balanced_f1": balanced_f1,
        "watchlist": watchlist,
        "urgent": urgent,
        "capacity_10pct": capacity_10,
        "capacity_20pct": capacity_20,
        "capacity_30pct": capacity_30,
    }


def split_model_data(model_df: pd.DataFrame, split: pd.DataFrame) -> dict[str, Any]:
    df = model_df.merge(split[KEYS + ["split"]], on=KEYS, how="inner")
    df = add_train_only_risk_priors(df)
    drop_cols = [*KEYS, "final_result", "risk_label", "split"]
    X = df.drop(columns=drop_cols)
    y = df["risk_label"]
    return {
        "df": df,
        "X": X,
        "y": y,
        "X_train": X[df["split"] == "train"],
        "X_test": X[df["split"] == "test"],
        "y_train": y[df["split"] == "train"],
        "y_test": y[df["split"] == "test"],
    }


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical = [c for c in X.columns if pd.api.types.is_string_dtype(X[c]) or X[c].dtype == "object"]
    numeric = [c for c in X.columns if c not in categorical]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ]
    )


def _xgboost_classifier(**overrides: Any) -> Any:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("xgboost is required for the full model comparison. Run: pip install -r requirements.txt") from exc
    params = dict(
        n_estimators=250,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=5,
        reg_lambda=5,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    params.update(overrides)
    return XGBClassifier(**params)


def candidate_classifiers() -> dict[str, Any]:
    logistic = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=RANDOM_STATE)
    random_forest = RandomForestClassifier(
        n_estimators=180,
        min_samples_leaf=20,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    knn = KNeighborsClassifier(n_neighbors=35, weights="distance")
    svc = CalibratedClassifierCV(
        LinearSVC(class_weight="balanced", C=0.5, max_iter=5000, random_state=RANDOM_STATE),
        cv=3,
    )
    xgboost = _xgboost_classifier(n_estimators=180)
    xgboost_regularized = _xgboost_classifier(n_estimators=180, max_depth=2, min_child_weight=10, reg_lambda=10)
    soft_voting = VotingClassifier(
        estimators=[
            ("logistic", logistic),
            ("random_forest", random_forest),
            ("xgboost", xgboost),
        ],
        voting="soft",
        weights=[1, 1, 2],
        n_jobs=-1,
    )
    return {
        "logistic_regression": logistic,
        "random_forest": random_forest,
        "knn": knn,
        "linear_svc_calibrated": svc,
        "xgboost": xgboost,
        "xgboost_regularized": xgboost_regularized,
        "soft_voting": soft_voting,
    }


def make_model_pipeline(X: pd.DataFrame, classifier: Any) -> Pipeline:
    return Pipeline([("preprocessor", build_preprocessor(X)), ("classifier", classifier)])


def threshold_for_f2(y_true: pd.Series, prob: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    beta2 = F2_BETA**2
    f2 = ((1 + beta2) * precision * recall) / np.maximum((beta2 * precision + recall), 1e-9)
    return float(thresholds[np.nanargmax(f2[:-1])])


def threshold_for_f1(y_true: pd.Series, prob: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    f1 = (2 * precision * recall) / np.maximum((precision + recall), 1e-9)
    return float(thresholds[np.nanargmax(f1[:-1])])


def compare_candidate_models(
    model_df: pd.DataFrame,
    split: pd.DataFrame,
    cv_splits: int = 3,
) -> dict[str, Any]:
    parts = split_model_data(model_df, split)
    X_train = parts["X_train"]
    y_train = parts["y_train"]
    X_test = parts["X_test"]
    y_test = parts["y_test"]

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    fitted_models = {}
    oof_predictions = {}
    train_test_rows = []
    for name, classifier in candidate_classifiers().items():
        model = make_model_pipeline(parts["X"], classifier)
        oof_prob = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba", n_jobs=None)[:, 1]
        oof_predictions[name] = oof_prob
        threshold = threshold_for_f1(y_train, oof_prob)
        watchlist_threshold = threshold_for_f2(y_train, oof_prob)
        cv_metrics = metrics_at_threshold(y_train, oof_prob, threshold)
        cv_metrics.update(
            {
                "model_name": name,
                "selection_threshold": threshold,
                "watchlist_threshold": watchlist_threshold,
                "selection_rule": "max_cv_f1",
                "evaluation": "cv_oof",
            }
        )
        rows.append(cv_metrics)

        model.fit(X_train, y_train)
        fitted_models[name] = model
        train_prob = model.predict_proba(X_train)[:, 1]
        test_prob = model.predict_proba(X_test)[:, 1]
        for evaluation, y_eval, prob_eval in [
            ("train_fit", y_train, train_prob),
            ("test", y_test, test_prob),
        ]:
            eval_metrics = metrics_at_threshold(y_eval, prob_eval, threshold)
            eval_metrics.update(
                {
                    "model_name": name,
                    "selection_threshold": threshold,
                    "watchlist_threshold": watchlist_threshold,
                    "selection_rule": "max_cv_f1",
                    "evaluation": evaluation,
                }
            )
            train_test_rows.append(eval_metrics)

    comparison = pd.DataFrame(rows).sort_values(["f1", "f2", "pr_auc", "roc_auc"], ascending=False)
    train_test = pd.DataFrame(train_test_rows)
    selected_row = comparison.iloc[0]
    selected_model_name = str(selected_row["model_name"])
    selected_model = fitted_models[selected_model_name]
    selected_threshold = float(selected_row["selection_threshold"])
    test_prob = selected_model.predict_proba(X_test)[:, 1]
    selected_test_metrics = metrics_at_threshold(y_test, test_prob, selected_threshold)
    selected_test_metrics.update(
        {
            "model_name": selected_model_name,
            "threshold_name": "cv_max_f1",
            "selection_rule": "best_cv_f1_then_test_once",
        }
    )

    comparison.to_csv(OUT_DIR / "model_comparison_cv.csv", index=False)
    train_test.to_csv(OUT_DIR / "model_overfit_check.csv", index=False)
    pd.DataFrame([selected_test_metrics]).to_json(OUT_DIR / "risk_metrics.json", orient="records", indent=2)
    joblib.dump(selected_model, MODEL_DIR / "week6_risk_model.joblib")
    return {
        "model_comparison": comparison,
        "overfit_check": train_test,
        "selected_model_name": selected_model_name,
        "selected_model": selected_model,
        "selected_oof_prob": oof_predictions[selected_model_name],
        "selected_threshold": selected_threshold,
        "selected_test_metrics": selected_test_metrics,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "test_prob": test_prob,
    }


def permutation_feature_drivers(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    baseline_prob: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    baseline_auc = roc_auc_score(y_test, baseline_prob)
    baseline_recall = recall_score(y_test, (baseline_prob >= threshold).astype(int), zero_division=0)
    rows = []
    for col in X_test.columns:
        shuffled = X_test.copy()
        shuffled[col] = rng.permutation(shuffled[col].to_numpy())
        prob = model.predict_proba(shuffled)[:, 1]
        rows.append(
            {
                "feature": col,
                "auc_drop": baseline_auc - roc_auc_score(y_test, prob),
                "recall_drop": baseline_recall - recall_score(y_test, (prob >= threshold).astype(int), zero_division=0),
            }
        )
    drivers = pd.DataFrame(rows).sort_values(["auc_drop", "recall_drop"], ascending=False)
    drivers.to_csv(OUT_DIR / "risk_feature_drivers.csv", index=False)
    behavioral_tokens = (
        "engagement_score",
        "assessment_submitted_ratio",
        "late_submission_count",
        "punctuality",
        "low_weight_completion",
        "starting_early",
        "material_active_days",
        "repeat_attempt",
        "motivation_proxy",
        "days_since_last_click",
        "consistency_rate",
        "active_days",
        "weekly_clicks",
        "click_trend",
        "activity_diversity",
        "pre_start",
        "days_before_start",
        "week6_archetype",
        "profile_group",
        "education_level",
        "imd_midpoint",
        "credit_load",
        "resource_ratio",
        "oucontent_ratio",
        "questionnaire_ratio",
    )
    behavioral = drivers[drivers["feature"].str.contains("|".join(behavioral_tokens), regex=True)].head(8)
    behavioral.to_csv(OUT_DIR / "risk_behavioral_feature_drivers.csv", index=False)
    return drivers


def train_risk_model(model_df: pd.DataFrame, split: pd.DataFrame) -> dict[str, Any]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    comparison_result = compare_candidate_models(model_df, split)
    model = comparison_result["selected_model"]
    y_train = comparison_result["y_train"]
    y_test = comparison_result["y_test"]
    test_prob = comparison_result["test_prob"]

    thresholds = choose_thresholds(y_train, comparison_result["selected_oof_prob"])
    thresholds["selected"] = comparison_result["selected_threshold"]
    primary_metrics = metrics_at_threshold(y_test, test_prob, thresholds["balanced_f1"])
    primary_metrics.update(
        {
            "model_name": comparison_result["selected_model_name"],
            "threshold_name": "cv_max_f1",
            "selection_rule": "best_cv_f1_model_then_max_f1_alert_threshold",
        }
    )
    watchlist_metrics = metrics_at_threshold(y_test, test_prob, thresholds["watchlist"])
    capacity_metrics = metrics_at_threshold(y_test, test_prob, thresholds["capacity_20pct"])

    rows = []
    for name, threshold in [
        ("urgent_balanced_f1", thresholds["balanced_f1"]),
        ("watchlist_f2", thresholds["watchlist"]),
        ("urgent_recall85", thresholds["urgent"]),
        ("selected_cv_max_f1", thresholds["selected"]),
        ("capacity_10pct", thresholds["capacity_10pct"]),
        ("capacity_20pct", thresholds["capacity_20pct"]),
        ("capacity_30pct", thresholds["capacity_30pct"]),
        ("fixed_0.25", 0.25),
        ("fixed_0.35", 0.35),
        ("fixed_0.50", 0.50),
        ("fixed_0.65", 0.65),
        ("fixed_0.75", 0.75),
    ]:
        row = metrics_at_threshold(y_test, test_prob, threshold)
        row["threshold_name"] = name
        rows.append(row)
    threshold_table = pd.DataFrame(rows)
    threshold_table.to_csv(OUT_DIR / "risk_threshold_analysis.csv", index=False)

    prob_true, prob_pred = calibration_curve(y_test, test_prob, n_bins=10, strategy="quantile")
    calibration = pd.DataFrame({"mean_predicted_risk": prob_pred, "observed_risk_rate": prob_true})
    calibration.to_csv(OUT_DIR / "calibration.csv", index=False)

    drivers = permutation_feature_drivers(
        model,
        comparison_result["X_test"],
        y_test,
        test_prob,
        comparison_result["selected_threshold"],
    )
    behavioral_drivers = pd.read_csv(OUT_DIR / "risk_behavioral_feature_drivers.csv")

    joblib.dump(model, MODEL_DIR / "week6_risk_model.joblib")
    pd.DataFrame([primary_metrics]).to_json(OUT_DIR / "risk_metrics.json", orient="records", indent=2)
    pd.DataFrame([watchlist_metrics]).to_json(OUT_DIR / "risk_watchlist_metrics.json", orient="records", indent=2)
    return {
        "model": model,
        "primary_metrics": primary_metrics,
        "watchlist_metrics": watchlist_metrics,
        "capacity_metrics": capacity_metrics,
        "threshold_table": threshold_table,
        "calibration": calibration,
        "feature_drivers": drivers,
        "behavioral_feature_drivers": behavioral_drivers,
        "y_test": y_test,
        "test_prob": test_prob,
        "model_comparison": comparison_result["model_comparison"],
        "overfit_check": comparison_result["overfit_check"],
        "selected_model_name": comparison_result["selected_model_name"],
    }
