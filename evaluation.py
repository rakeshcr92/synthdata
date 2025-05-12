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

    numeric_cols = real_data.select_dtypes(include=["float64", "int64"]).columns
    if len(numeric_cols) == 0:
        return score, []

    col = numeric_cols[0]
    fig, ax = plt.subplots()
    sns.histplot(real_data[col], color="blue", label="Real", stat="density", kde=False)
    sns.histplot(synthetic_data[col], color="orange", label="Synthetic", stat="density", kde=False)
    ax.set_title(f"Histogram Comparison: {col}")
    ax.legend()

    return score, [fig]
