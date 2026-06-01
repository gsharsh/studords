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
    calibration = model_result["calibration"]
    y_test = model_result["y_test"]
    test_prob = model_result["test_prob"]
    pred = (test_prob >= primary["threshold"]).astype(int)

    plt.figure(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Pass/Distinction", "Withdraw/Fail"], cmap="Blues")
    plt.title("Urgent Alert Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6.5, 4.2))
    plot_df = threshold_table[
        threshold_table["threshold_name"].isin(
            [
                "urgent_balanced_f1",
                "watchlist_f2",
                "capacity_10pct",
                "capacity_20pct",
                "capacity_30pct",
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
    plt.axvline(primary["threshold"], color="#E45756", linestyle="--", label="Urgent threshold")
    plt.axvline(watchlist["threshold"], color="#4C78A8", linestyle=":", label="Watchlist threshold")
    plt.title("Alert Threshold Tradeoff")
    plt.xlabel("Risk threshold")
    plt.ylabel("Metric")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "threshold_tradeoff.png", dpi=160)
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

    heading("Studor PathAI: Baseline Product Prototype")
    body("This analysis converts OULAD behavioral, assessment, and student context data into three product-facing assets: a weekly engagement score, a Week 6 disengagement risk model, and next-course recommendations.")
    body(f"The Week 6 model predicts Withdrawn/Fail versus Pass/Distinction. The urgent tier now alerts {primary['alert_rate']:.1%} of the test cohort at precision {primary['precision']:.2f} and recall {primary['recall']:.2f}; the broader watchlist alerts {watchlist['alert_rate']:.1%} for monitoring.")
    story.append(Image(str(FIG_DIR / "outcome_eda.png"), width=480, height=175))
    story.append(PageBreak())

    heading("Data Cleaning And Leakage Controls")
    body("The enrolment key is student + module + presentation; student ID alone is not unique because learners can take multiple modules. Cleaning is conservative: semantic issues are audited rather than silently rewriting the official target label. Negative VLE dates are treated as valid pre-start engagement before day 0.")
    cleaning_rows = [["Check", "Issue Count", "Policy"]]
    for _, row in audits["consistency"].iterrows():
        cleaning_rows.append([row["check"], row["issue_count"], row["policy"]])
    story.append(make_table(cleaning_rows, font_size=7))
    story.append(Spacer(1, 8))
    leakage_rows = [["Feature", "Available Week 6?", "Used?", "Reason"]]
    for _, row in audits["leakage"].head(7).iterrows():
        leakage_rows.append([row["candidate_feature"], row["available_by_week6"], row["used"], row["reason"]])
    story.append(make_table(leakage_rows, font_size=7))
    story.append(PageBreak())

    heading("Task 1: Dynamic Engagement Score")
    body("The 0-100 engagement score updates weekly and is peer-normalized within module, presentation, and week. Component weights are derived from the training split by measuring how well each component separates successful students from withdraw/fail students at Week 6, then rounded into a readable scorecard.")
    top_features = feature_rationale_df.assign(abs_corr=feature_rationale_df["risk_correlation"].abs()).sort_values(
        "abs_corr", ascending=False
    ).head(5)
    top_feature_text = "; ".join(f"{row['feature']} r={row['risk_correlation']:+.3f}" for _, row in top_features.iterrows())
    body(f"Most data-backed Week 6 signals: {top_feature_text}.")
    weight_rows = [["Component", "Success AUC", "Weight"]]
    for _, row in weight_rationale.iterrows():
        weight_rows.append([row["label"], f"{row['success_auc']:.3f}", f"{100 * row['score_weight']:.0f}%"])
    story.append(make_table(weight_rows, font_size=7))
    body("Scorecard guardrail: components with train success AUC <= 0.50 are assigned 0% weight because they are weaker than random as score components in the current definition. They can remain exploratory features, but they should not move the trusted 0-100 score.")
    archetype_path = OUT_DIR / "engagement_archetype_definitions.csv"
    if archetype_path.exists():
        archetype_defs = pd.read_csv(archetype_path)
        archetype_rows = [["Archetype", "Feature signature"]]
        for _, row in archetype_defs.iterrows():
            archetype_rows.append([row["archetype"], row["feature_signature"]])
        story.append(make_table(archetype_rows, font_size=5))
    story.append(
        Table(
            [
                [
                    Image(str(FIG_DIR / "engagement_archetypes_core.png"), width=240, height=105),
                    Image(str(FIG_DIR / "engagement_archetypes_additional.png"), width=240, height=105),
                ]
            ]
        )
    )
    story.append(PageBreak())

    heading("Task 2: Week 6 Risk Model")
    body("The supervised model uses only features available through Week 6. It includes the Task 1 engagement score, a Week 6-safe behavioural archetype, transformed profile context (education order, IMD midpoint, age order, normalized credit load), resource-mix ratios, and selected interaction features. Candidate models are compared with cross-validated F1, PR-AUC, and ROC-AUC for the urgent alert, while a high-recall F2 threshold remains available as a watchlist.")
    body(f"Urgent threshold {primary['threshold']:.2f}: precision {primary['precision']:.3f}, recall {primary['recall']:.3f}, F1 {primary['f1']:.3f}, ROC-AUC {primary['roc_auc']:.3f}. Confusion matrix counts: TN={primary['true_negatives']}, FP={primary['false_positives']}, FN={primary['false_negatives']}, TP={primary['true_positives']}.")
    threshold_rows = [["Tier", "Threshold", "Precision", "Recall", "F1", "Alerts", "Alert Share", "FP", "FN"]]
    for row in model_result["threshold_table"].to_dict("records"):
        if row["threshold_name"] in ["urgent_balanced_f1", "watchlist_f2", "capacity_20pct", "fixed_0.50"]:
            tier = {
                "urgent_balanced_f1": "Urgent",
                "watchlist_f2": "Watchlist",
                "capacity_20pct": "Capacity 20%",
                "fixed_0.50": "Fixed 0.50",
            }[row["threshold_name"]]
            threshold_rows.append(
                [
                    tier,
                    f"{row['threshold']:.2f}",
                    f"{row['precision']:.2f}",
                    f"{row['recall']:.2f}",
                    f"{row['f1']:.2f}",
                    int(row["alerts"]),
                    f"{row['alert_rate']:.1%}",
                    int(row["false_positives"]),
                    int(row["false_negatives"]),
                ]
            )
    story.append(make_table(threshold_rows, font_size=7))
    story.append(Spacer(1, 6))
    story.append(Image(str(FIG_DIR / "confusion_matrix.png"), width=310, height=240))
    story.append(Image(str(FIG_DIR / "threshold_tradeoff.png"), width=310, height=205))
    story.append(PageBreak())

    heading("Calibration And Advisor Alert Design")
    body("The alert design uses two tiers. Urgent alerts use a balanced F1 threshold to create advisor tasks with better precision; watchlist alerts use a higher-recall F2 threshold for monitoring and automated nudges.")
    story.append(Image(str(FIG_DIR / "calibration.png"), width=360, height=255))
    overall = model_result["feature_drivers"].head(3).copy()
    behavioral = model_result["behavioral_feature_drivers"].head(3).copy()
    driver_rows = [["Group", "Feature", "Mechanism"]]
    mechanisms = {
        "avg_score_so_far_6": "Early academic struggle may signal content difficulty before final failure.",
        "engagement_score_6": "Combines recency, consistency, diversity, and assessment behavior into a trajectory signal.",
        "assessment_submitted_ratio_6": "Missing early assessments is both predictive and directly actionable for advisors.",
    }
    for _, row in overall.iterrows():
        feature = row["feature"].replace("num__", "")
        driver_rows.append(["Overall", feature, mechanisms.get(feature, "High-ranking driver in permutation importance.")])
    for _, row in behavioral.iterrows():
        feature = row["feature"].replace("num__", "")
        driver_rows.append(["Behavioral", feature, mechanisms.get(feature, "Behavioral signal advisors can investigate or act on.")])
    story.append(make_table(driver_rows, font_size=7))
    story.append(Spacer(1, 6))
    alert_rows = [
        ["PathAI Advisor Alert", ""],
        ["Student", "S-10482"],
        ["Course", "DDD-2014J"],
        ["Risk tier", "Urgent"],
        ["Predicted risk", "78%"],
        ["Why flagged", "No VLE activity in 12 days; engagement score dropped from 61 to 34 over two weeks; first assessment due by Week 6 was not submitted."],
        ["Suggested action", "Send check-in within 48 hours. Ask about workload, access issues, and confidence with course material. Offer academic support session."],
    ]
    story.append(make_table(alert_rows, font_size=8))
    story.append(PageBreak())

    heading("Task 3: Course Recommendations")
    body("The recommender primarily serves students planning a next semester, while advisors can use the same output in guidance conversations. Content-based recommendations use education, age band, historical module success, and prior engagement band. Collaborative filtering uses cosine similarity over successful module patterns.")
    rec_rows = [
        ["Metric", "Value"],
        ["Holdout students", recommender_metrics["holdout_students"]],
        ["Content hit@3", f"{recommender_metrics['content_hit_rate_at_3']:.3f}"],
        ["Collaborative hit@3", f"{recommender_metrics['cf_hit_rate_at_3']:.3f}"],
        ["Content coverage", f"{recommender_metrics['content_coverage']}/{recommender_metrics['catalog_modules']}"],
        ["Collaborative coverage", f"{recommender_metrics['cf_coverage']}/{recommender_metrics['catalog_modules']}"],
        ["Cold start", ", ".join(recommender_metrics["cold_start_strategy"])],
    ]
    story.append(make_table(rec_rows, font_size=8))
    body("Limitations: OULAD has a small module catalog, recommendation evaluation is a proxy holdout, and production deployment would need advisor feedback loops plus monitoring by module and presentation.")
    roadmap_rows = [
        ["90-day roadmap", "1. Pilot tiered alerts with advisors and track accepted interventions."],
        ["", "2. Add advisor feedback labels to calibrate thresholds by module workload."],
        ["", "3. Extend recommendations with richer course metadata and student goals."],
    ]
    story.append(make_table(roadmap_rows, font_size=8))
    doc.build(story)
