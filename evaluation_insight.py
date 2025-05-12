
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sdv.evaluation.single_table import evaluate_quality

def evaluate_synthetic_data(real_data, synthetic_data, metadata):
    report = evaluate_quality(
        real_data=real_data,
        synthetic_data=synthetic_data,
        metadata=metadata
    )
    score = report.get_score()

    # Generate Correlation Heatmaps
    def plot_correlation(df, title):
        corr = df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr, cmap="coolwarm", ax=ax, annot=False)
        ax.set_title(title)
        return fig

    fig_real = plot_correlation(real_data, "Real Data Correlation")
    fig_synth = plot_correlation(synthetic_data, "Synthetic Data Correlation")

    return score, [fig_real, fig_synth]

def summarize_data_insights(real_data, synthetic_data):
    insights = {}

    real_summary = real_data.describe(include='all').fillna("N/A").astype(str)
    synth_summary = synthetic_data.describe(include='all').fillna("N/A").astype(str)

    insights["real_summary"] = real_summary
    insights["synthetic_summary"] = synth_summary

    diff_summary = pd.DataFrame()
    for col in real_data.columns:
        if col in synthetic_data.columns:
            try:
                real_mean = real_data[col].mean()
                synth_mean = synthetic_data[col].mean()
                diff = abs(real_mean - synth_mean)
                diff_summary.at[col, "Real Mean"] = real_mean
                diff_summary.at[col, "Synthetic Mean"] = synth_mean
                diff_summary.at[col, "Mean Difference"] = diff
            except:
                continue

    insights["mean_diff"] = diff_summary
    return insights
