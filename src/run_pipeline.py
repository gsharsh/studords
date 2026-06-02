from __future__ import annotations

from data_loader import FIG_DIR, OUT_DIR, REPORT_DIR, audit_and_clean_data, ensure_dirs, load_data
from evaluation import (
    plot_data_quality,
    plot_engagement_outputs,
    plot_model_outputs,
    plot_task2_behavioral_correlation_heatmap,
    write_report,
)
from features import build_weekly_base_features, feature_rationale, make_enrollment_split
from modeling import build_week6_model_dataset, train_risk_model
from recommendations import evaluate_recommenders
from scoring import apply_engagement_score, derive_engagement_weights, select_archetypes, validate_engagement_score
from write_word_report import write_word_report


def run_pipeline() -> dict:
    ensure_dirs()
    raw_data = load_data()
    data, audits = audit_and_clean_data(raw_data)
    plot_data_quality(audits, data)

    weekly_base = build_weekly_base_features(data)
    split = make_enrollment_split(weekly_base)
    weights, weight_rationale = derive_engagement_weights(weekly_base, split)
    weekly = apply_engagement_score(weekly_base, weights)
    weekly.to_csv(OUT_DIR / "weekly_engagement_features.csv", index=False)
    weekly[["code_module", "code_presentation", "id_student", "week", "final_result", "engagement_score"]].to_csv(
        OUT_DIR / "weekly_engagement_scores.csv",
        index=False,
    )

    score_band_risk = validate_engagement_score(weekly)
    feature_rationale_df = feature_rationale(weekly)
    archetypes = select_archetypes(weekly)
    plot_engagement_outputs(archetypes, score_band_risk, feature_rationale_df)

    model_df = build_week6_model_dataset(weekly, data)
    plot_task2_behavioral_correlation_heatmap(model_df)
    model_result = train_risk_model(model_df, split)
    plot_model_outputs(model_result)

    recommender_metrics = evaluate_recommenders(data, weekly)
    write_report(audits, weight_rationale, score_band_risk, feature_rationale_df, model_result, recommender_metrics)
    write_word_report()

    return {
        "data": data,
        "audits": audits,
        "weekly": weekly,
        "split": split,
        "weights": weights,
        "weight_rationale": weight_rationale,
        "score_band_risk": score_band_risk,
        "feature_rationale": feature_rationale_df,
        "model_result": model_result,
        "recommender_metrics": recommender_metrics,
    }


def main() -> None:
    result = run_pipeline()
    primary = result["model_result"]["primary_metrics"]
    watchlist = result["model_result"]["watchlist_metrics"]
    print("Pipeline complete.")
    print(f"Report (PDF): {REPORT_DIR / 'Studor_PathAI_Report.pdf'}")
    print(f"Report (Word): {REPORT_DIR / 'Studor_PathAI_Report.docx'}")
    print(f"Figures: {FIG_DIR}")
    print(f"Engagement weights: {result['weights']}")
    print(f"High-touch queue metrics: {primary}")
    print(f"Top-60% light-touch cutoff metrics: {watchlist}")
    print(f"Intervention tiers:\n{result['model_result']['intervention_tiers']}")
    print(f"Recommendation metrics: {result['recommender_metrics']}")


if __name__ == "__main__":
    main()
