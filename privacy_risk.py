
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pandas as pd

def compute_privacy_risk(real_data, synthetic_data, k=1, threshold=0.05):
    # Ensure both datasets are numeric and aligned
    real = real_data.select_dtypes(include=["number"]).dropna().reset_index(drop=True)
    synth = synthetic_data.select_dtypes(include=["number"]).dropna().reset_index(drop=True)

    # Match column order
    synth = synth[real.columns.intersection(synth.columns)]

    # Use NearestNeighbors to find closest real row for each synthetic row
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(real)
    distances, _ = nbrs.kneighbors(synth)

    # Analyze distances
    avg_distance = np.mean(distances)
    min_distance = np.min(distances)
    risky_count = np.sum(distances < threshold)

    risk_report = {
        "average_distance": avg_distance,
        "min_distance": min_distance,
        "rows_below_threshold": int(risky_count),
        "risk_level": "HIGH" if risky_count > 0 else "LOW"
    }

    return risk_report
