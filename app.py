
import gradio as gr
import pandas as pd
from synthesis import generate_synthetic_data
from evaluation_insight import evaluate_synthetic_data, summarize_data_insights
from privacy_risk import compute_privacy_risk
from sdv.metadata import SingleTableMetadata
import matplotlib.pyplot as plt
import tempfile
import json

def synthesize(csv_file, model_choice, privacy_mode):
    df = pd.read_csv(csv_file.name)

    # Step 1: Generate Synthetic Data
    synthetic_df = generate_synthetic_data(df, model_choice, privacy_mode)
    output_csv = "synthetic_output.csv"
    synthetic_df.to_csv(output_csv, index=False)

    # Step 2: Metadata for Evaluation
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=df)

    # Step 3: Evaluate Quality & Generate Heatmaps
    quality_score, plots = evaluate_synthetic_data(df, synthetic_df, metadata)

    # Step 4: Insight Summary Table (Mean Differences)
    insight_summaries = summarize_data_insights(df, synthetic_df)
    mean_diff_df = insight_summaries["mean_diff"]
    mean_diff_csv = "mean_difference_summary.csv"
    mean_diff_df.to_csv(mean_diff_csv)

    # Step 5: Privacy Risk Evaluation
    risk_report = compute_privacy_risk(df, synthetic_df)
    risk_summary = json.dumps(risk_report, indent=2)

    # Save figures to temp files for Gradio Gallery
    image_paths = []
    for fig in plots:
        tmpfile = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        fig.savefig(tmpfile.name)
        plt.close(fig)
        image_paths.append(tmpfile.name)

    return output_csv, synthetic_df.head(), f"{quality_score:.2f}", image_paths, mean_diff_csv, risk_summary

gr.Interface(
    fn=synthesize,
    inputs=[
        gr.File(label="Upload CSV", file_types=[".csv"]),
        gr.Radio(["Auto", "CTGAN", "TVAE", "GaussianCopula"], label="Choose Model"),
        gr.Checkbox(label="Enable Privacy Mode (Differential Privacy Constraints)")
    ],
    outputs=[
        gr.File(label="Download Synthetic CSV"),
        gr.Dataframe(label="Preview of Synthetic Data"),
        gr.Text(label="Data Quality Score (0 to 1)"),
        gr.Gallery(label="Real vs Synthetic Correlation Heatmaps"),
        gr.File(label="Download Insight Summary (Mean Differences)"),
        gr.Textbox(label="Privacy Risk Report (Nearest-Neighbor Score)")
    ],
    title="Synthetic Data Generator + Insight & Privacy Dashboard",
    description="Generate privacy-preserving synthetic data with model auto-selection, insight reports, and privacy risk evaluation."
).launch()
