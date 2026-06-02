from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sklearn.metrics import ConfusionMatrixDisplay

from data_loader import FIG_DIR, OUT_DIR, REPORT_DIR, RISK_LABELS


def plot_data_quality(audits: dict[str, pd.DataFrame], data: dict[str, pd.DataFrame]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    outcome = data["student_info"].assign(risk=data["student_info"]["final_result"].isin(RISK_LABELS).astype(int))
    module_risk = outcome.groupby("code_module", as_index=False)["risk"].mean().sort_values("risk", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.countplot(data=data["student_info"], x="final_result", order=["Withdrawn", "Fail", "Pass", "Distinction"], ax=axes[0])
    axes[0].set_title("Final Result Distribution")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Enrolments")
    sns.barplot(data=module_risk, x="code_module", y="risk", ax=axes[1])
    axes[1].set_title("Withdraw/Fail Rate By Module")
    axes[1].set_xlabel("Module")
    axes[1].set_ylabel("Risk rate")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "outcome_eda.png", dpi=160)
    plt.close()

    missing = audits["missingness"].sort_values("missing_count", ascending=False).head(10)
    plt.figure(figsize=(9, 4.2))
    if not missing.empty:
        missing = missing.copy()
        missing["field"] = missing["table"] + "." + missing["column"]
        sns.barplot(data=missing, y="field", x="missing_pct", color="#4C78A8")
    plt.title("Top Missingness Patterns")
    plt.xlabel("Missing share")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "data_missingness.png", dpi=160)
    plt.close()


def plot_engagement_outputs(archetypes: pd.DataFrame, score_band_risk: pd.DataFrame, feature_rationale_df: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    core = ["Steady Engager", "Early Dropout", "Late Recoverer"]
    additional = [
        "Sporadic / Burst Engager",
        "Perfectionist Procrastinator",
        "Surface Level / Compliance Engager",
        "Opportunistic Engager",
    ]

    plt.figure(figsize=(11, 6.2))
    sns.lineplot(data=archetypes, x="week", y="engagement_score", hue="archetype", marker="o")
    plt.title("Representative Weekly Engagement Trajectories")
    plt.xlabel("Teaching week")
    plt.ylabel("Engagement score")
    plt.ylim(0, 100)
    plt.legend(title="Student archetype", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "engagement_archetypes.png", dpi=160)
    plt.close()

    for filename, title, subset in [
        ("engagement_archetypes_core.png", "Core Engagement Archetypes", core),
        ("engagement_archetypes_additional.png", "Additional Behavioral Archetypes", additional),
    ]:
        plot_data = archetypes[archetypes["archetype"].isin(subset)]
        plt.figure(figsize=(8.2, 4.2))
        sns.lineplot(data=plot_data, x="week", y="engagement_score", hue="archetype", marker="o")
        plt.title(title)
        plt.xlabel("Teaching week")
        plt.ylabel("Engagement score")
        plt.ylim(0, 100)
        plt.legend(title="", loc="best", fontsize=8)
        plt.tight_layout()
        plt.savefig(FIG_DIR / filename, dpi=160)
        plt.close()

    plt.figure(figsize=(6.5, 4.2))
    sns.barplot(data=score_band_risk, x="score_band", y="observed_withdraw_fail_rate", color="#F58518")
    plt.title("Observed Risk By Week 6 Engagement Score Band")
    plt.xlabel("Week 6 engagement score band")
    plt.ylabel("Observed withdraw/fail rate")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "score_band_risk.png", dpi=160)
    plt.close()

    plot_df = feature_rationale_df.copy()
    plot_df["abs_corr"] = plot_df["risk_correlation"].abs()
    plot_df = plot_df.sort_values("abs_corr", ascending=False).head(14)
    plt.figure(figsize=(9.5, 5.6))
    ax = sns.barplot(data=plot_df, y="feature", x="risk_correlation", hue="risk_correlation", palette="vlag", legend=False)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Week 6 Behavioral Features Most Correlated With Withdraw/Fail")
    plt.xlabel("Correlation with withdraw/fail risk (negative = protective)")
    plt.ylabel("")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "week6_feature_rationale.png", dpi=160)
    plt.close()


def plot_model_outputs(model_result: dict[str, Any]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    primary = model_result["primary_metrics"]
    watchlist = model_result["watchlist_metrics"]
    threshold_table = model_result["threshold_table"]
    intervention_tiers = model_result["intervention_tiers"]
    calibration = model_result["calibration"]
    y_test = model_result["y_test"]
    test_prob = model_result["test_prob"]
    pred = (test_prob >= primary["threshold"]).astype(int)

    plt.figure(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Pass/Distinction", "Withdraw/Fail"], cmap="Blues")
    plt.title("High-Touch Queue Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6.5, 4.2))
    plot_df = threshold_table[
        threshold_table["threshold_name"].isin(
            [
                "high_touch_20pct",
                "light_touch_60pct",
                "cv_max_f1_threshold",
                "cv_max_f2_threshold",
                "capacity_10pct",
                "capacity_20pct",
                "capacity_30pct",
                "capacity_60pct",
                "fixed_0.25",
                "fixed_0.35",
                "fixed_0.50",
                "fixed_0.65",
                "fixed_0.75",
            ]
        )
    ]
    sns.lineplot(data=plot_df.sort_values("threshold"), x="threshold", y="precision", marker="o", label="Precision")
    sns.lineplot(data=plot_df.sort_values("threshold"), x="threshold", y="recall", marker="o", label="Recall")
    plt.axvline(primary["threshold"], color="#E45756", linestyle="--", label="High-touch cutoff")
    plt.axvline(watchlist["threshold"], color="#4C78A8", linestyle=":", label="Light-touch cutoff")
    plt.title("Risk Cutoff Tradeoff")
    plt.xlabel("Risk threshold")
    plt.ylabel("Metric")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "threshold_tradeoff.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7.2, 4.2))
    tier_plot = intervention_tiers.copy()
    sns.barplot(data=tier_plot, x="tier", y="student_share", color="#4C78A8", label="Student share")
    ax = plt.gca()
    ax2 = ax.twinx()
    sns.pointplot(data=tier_plot, x="tier", y="observed_withdraw_fail_rate", color="#E45756", ax=ax2, label="Observed risk")
    ax.set_xlabel("")
    ax.set_ylabel("Share of students")
    ax2.set_ylabel("Observed withdraw/fail rate")
    ax.set_title("Capacity-Based Intervention Tiers")
    ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "intervention_tiers.png", dpi=160)
    plt.close()

    plt.figure(figsize=(5.5, 4.2))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    plt.plot(calibration["mean_predicted_risk"], calibration["observed_risk_rate"], marker="o", label="Model")
    plt.xlabel("Predicted risk")
    plt.ylabel("Observed withdraw/fail rate")
    plt.title("Calibration: Predicted vs Observed Risk")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "calibration.png", dpi=160)
    plt.close()


def plot_task2_behavioral_correlation_heatmap(model_df: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    features = [
        "risk_label",
        "engagement_score_6",
        "assessment_submitted_ratio_6",
        "missed_assessments_6",
        "weekly_clicks_norm_6",
        "active_days_last_7_6",
        "study_regularity_score_6",
        "activity_diversity_6",
        "activity_entropy_6",
        "material_active_days_6",
        "recent_material_active_days_2w_6",
        "pre_start_proactivity_6",
        "days_since_last_click_6",
        "inactive_last_14_days_6",
        "longest_inactive_gap_6",
        "planned_material_coverage_6",
        "low_weight_completion_ratio_6",
    ]
    features = [col for col in features if col in model_df.columns]
    corr = model_df[features].corr(numeric_only=True)
    corr.to_csv(OUT_DIR / "task2_behavioral_correlation_matrix.csv")

    plt.figure(figsize=(12, 9))
    sns.heatmap(
        corr,
        cmap="vlag",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        cbar_kws={"label": "Pearson correlation"},
    )
    plt.title("Task 2 Week 6 Behavioral Feature Correlation Heatmap")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "task2_behavioral_correlation_heatmap.png", dpi=160)
    plt.close()


def make_table(data: list[list[Any]], font_size: int = 8) -> Table:
    table = Table(data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return table


def write_report(
    audits: dict[str, pd.DataFrame],
    weight_rationale: pd.DataFrame,
    score_band_risk: pd.DataFrame,
    feature_rationale_df: pd.DataFrame,
    model_result: dict[str, Any],
    recommender_metrics: dict[str, Any],
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        str(REPORT_DIR / "Studor_PathAI_Report.pdf"),
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    story = []

    def heading(text: str) -> None:
        story.append(Paragraph(text, styles["Heading1"]))
        story.append(Spacer(1, 6))

    def body(text: str) -> None:
        story.append(Paragraph(text, styles["BodyText"]))
        story.append(Spacer(1, 6))

    primary = model_result["primary_metrics"]
    watchlist = model_result["watchlist_metrics"]
    intervention_tiers = model_result["intervention_tiers"]

    heading("Executive Summary")
    body("This report presents a production-oriented PathAI prototype built on one semester of OULAD data. The system combines three outputs: a weekly engagement score, a Week 6 disengagement model, and a next-course recommendation engine.")
    body("Task 1 produces a dynamic 0-100 engagement score from behavior and assessment signals. Task 2 predicts Withdrawn/Fail versus Pass/Distinction using only Week <= 6 information and converts risk into intervention tiers. Task 3 recommends top-3 modules using both collaborative filtering and a shared feature-space content model.")
    summary_rows = [
        ["Metric", "Result"],
        ["Task 2 model", "XGBoost (selected from comparison table)"],
        ["High-touch precision", f"{primary['precision']:.3f}"],
        ["High-touch recall", f"{primary['recall']:.3f}"],
        ["High-touch F1", f"{primary['f1']:.3f}"],
        ["ROC-AUC", f"{primary['roc_auc']:.3f}"],
        ["PR-AUC", f"{primary['pr_auc']:.3f}"],
        ["High-touch queue share", f"{primary['alert_rate']:.1%}"],
        ["Top-60% recall", f"{watchlist['recall']:.3f}"],
        ["Task 3 content hit@3", f"{recommender_metrics['content_hit_rate_at_3']:.3f}"],
        ["Task 3 CF hit@3", f"{recommender_metrics['cf_hit_rate_at_3']:.3f}"],
    ]
    story.append(make_table(summary_rows, font_size=8))
    body("Result: the system is statistically credible and operationally actionable. It is intentionally framed for deployment as decision support, not autonomous intervention.")
    story.append(Image(str(FIG_DIR / "outcome_eda.png"), width=500, height=180))
    story.append(PageBreak())

    heading("Data Cleaning And Archetypes")
    body("The enrollment grain is id_student x code_module x code_presentation. Student ID alone is not unique in OULAD. Cleaning decisions preserve behavior signal while preventing target leakage.")
    cleaning_rows = [["Check", "Issue Count", "Policy"]]
    for _, row in audits["consistency"].iterrows():
        cleaning_rows.append([row["check"], row["issue_count"], row["policy"]])
    story.append(make_table(cleaning_rows, font_size=7))
    story.append(Spacer(1, 6))
    leakage_rows = [["Candidate Feature", "Week 6 Available?", "Used?", "Reason"]]
    for _, row in audits["leakage"].head(8).iterrows():
        leakage_rows.append([row["candidate_feature"], row["available_by_week6"], row["used"], row["reason"]])
    story.append(make_table(leakage_rows, font_size=7))
    body("Negative VLE dates were retained as pre-start proactivity rather than dropped. This provides useful early signal for both risk and recommendation tasks.")
    archetype_path = OUT_DIR / "engagement_archetype_definitions.csv"
    if archetype_path.exists():
        defs = pd.read_csv(archetype_path)
        archetype_rows = [["Archetype", "Behavioral Signature"]]
        for _, row in defs.iterrows():
            archetype_rows.append([row["archetype"], row["feature_signature"]])
        story.append(make_table(archetype_rows, font_size=6))
    story.append(PageBreak())

    heading("Task 1: Engagement Score")
    body("Approach: construct weekly behavioral features, normalize by module-presentation-week peers, and aggregate with train-derived weights. Components with weak train signal are assigned zero weight.")
    weight_rows = [["Component", "Success AUC", "Final Weight"]]
    for _, row in weight_rationale.iterrows():
        weight_rows.append([row["label"], f"{row['success_auc']:.3f}", f"{100 * row['score_weight']:.0f}%"])
    story.append(make_table(weight_rows, font_size=7))
    top = feature_rationale_df.assign(abs_corr=feature_rationale_df["risk_correlation"].abs()).sort_values("abs_corr", ascending=False).head(6)
    top_rows = [["Feature", "Correlation With Withdraw/Fail"]]
    for _, row in top.iterrows():
        top_rows.append([row["feature"], f"{row['risk_correlation']:+.3f}"])
    story.append(make_table(top_rows, font_size=7))
    body("Key decision: prioritize recency, consistency, diversity, and assessment follow-through over raw click count. This keeps the score transparent and resistant to click-spam behavior.")
    story.append(Table([[Image(str(FIG_DIR / "score_band_risk.png"), width=245, height=140), Image(str(FIG_DIR / "week6_feature_rationale.png"), width=245, height=140)]]))
    story.append(PageBreak())

    heading("Task 2: Week 6 Disengagement Model")
    body("Approach: leakage-safe Week 6 binary classification. XGBoost was selected from a reproducible comparison set and then translated into advisor-capacity intervention tiers.")
    cmp_rows = [["Model", "Precision", "Recall", "F1", "ROC-AUC"]]
    for _, row in model_result["model_comparison"].head(4).iterrows():
        cmp_rows.append([row["model_name"], f"{row['precision']:.3f}", f"{row['recall']:.3f}", f"{row['f1']:.3f}", f"{row['roc_auc']:.3f}"])
    story.append(make_table(cmp_rows, font_size=7))
    threshold_rows = [["Cutoff", "Precision", "Recall", "F1", "Queue Share"]]
    threshold_rows.append(["High-touch 20%", f"{primary['precision']:.3f}", f"{primary['recall']:.3f}", f"{primary['f1']:.3f}", f"{primary['alert_rate']:.1%}"])
    threshold_rows.append(["Top-60% light-touch", f"{watchlist['precision']:.3f}", f"{watchlist['recall']:.3f}", f"{watchlist['f1']:.3f}", f"{watchlist['alert_rate']:.1%}"])
    story.append(make_table(threshold_rows, font_size=7))
    body("Key decision: use tiered intervention (high-touch, light-touch, monitoring) instead of a single campus-wide alert to reduce advisor fatigue.")
    story.append(Table([[Image(str(FIG_DIR / "task2_behavioral_correlation_heatmap.png"), width=245, height=170), Image(str(FIG_DIR / "intervention_tiers.png"), width=245, height=170)]]))
    story.append(PageBreak())

    heading("Task 3: Recommendation Engine")
    body("Approach: compare collaborative filtering to a shared feature-space content model. Student Week 6 vectors are matched to module profile vectors via cosine similarity, blended with a Wilson lower-bound pass prior.")
    rec_rows = [
        ["Metric", "Result"],
        ["Evaluation split", recommender_metrics.get("evaluation_split", "Temporal holdout")],
        ["Holdout students", recommender_metrics["holdout_students"]],
        ["Content hit@3", f"{recommender_metrics['content_hit_rate_at_3']:.3f}"],
        ["Collaborative hit@3", f"{recommender_metrics['cf_hit_rate_at_3']:.3f}"],
        ["Content coverage", f"{recommender_metrics['content_coverage']}/{recommender_metrics['catalog_modules']}"],
        ["Collaborative coverage", f"{recommender_metrics['cf_coverage']}/{recommender_metrics['catalog_modules']}"],
        ["Cold-start modules", ", ".join(recommender_metrics["cold_start_strategy"])],
    ]
    story.append(make_table(rec_rows, font_size=8))
    holdout_eval_path = OUT_DIR / "recommendation_holdout_eval.csv"
    if holdout_eval_path.exists():
        eval_df = pd.read_csv(holdout_eval_path)
        by_module = eval_df.groupby("actual_next_module")[["content_hit_at_3", "cf_hit_at_3"]].mean().sort_index().reset_index()
        by_rows = [["Actual Module", "Content hit@3", "CF hit@3"]]
        for _, row in by_module.iterrows():
            by_rows.append([row["actual_next_module"], f"{row['content_hit_at_3']:.3f}", f"{row['cf_hit_at_3']:.3f}"])
        story.append(make_table(by_rows[:8], font_size=7))
    body("Key decision: replace raw pass-rate priors with Wilson lower-bound priors to reduce small-sample bias. This makes content recommendations more stable and defensible.")
    story.append(PageBreak())

    heading("Gaps And 90-Day Plan")
    body("This prototype is intentionally simple and explainable, but not yet production-complete. The table below lists key gaps and concrete next actions.")
    gaps_rows = [
        ["Gap", "Why It Matters", "90-Day Action"],
        ["Limited temporal depth", "One-semester windows can miss drift", "Move to rolling-semester retraining and drift checks"],
        ["No advisor feedback loop", "Model thresholds are not tied to intervention outcomes", "Capture advisor action/outcome labels and recalibrate monthly"],
        ["Recommendation label weakness", "Next module may reflect timetable constraints", "Add schedule constraints and student intent tags"],
        ["Subgroup calibration not fully audited", "Uneven risk quality can harm trust", "Run subgroup calibration audits and apply targeted recalibration"],
        ["Reproducibility hardening", "Notebook and production states can diverge", "Introduce versioned feature snapshots and run manifests"],
    ]
    story.append(make_table(gaps_rows, font_size=7))
    roadmap_rows = [
        ["Phase", "Delivery Focus"],
        ["Days 0-30", "Deploy advisor dashboard in shadow mode; validate data freshness and triage quality"],
        ["Days 31-60", "Tune intervention thresholds by module and staffing constraints"],
        ["Days 61-90", "Retrain with feedback labels; publish impact and calibration report"],
    ]
    story.append(make_table(roadmap_rows, font_size=8))
    body("Why this is not best-in-class yet: it optimizes predictive performance on available labels, but does not yet optimize intervention utility, fairness constraints, and real scheduling feasibility end-to-end.")
    doc.build(story)
