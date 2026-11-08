"""Management zones: split the parcel into areas that behave alike, without labels.

The first version of this project trained classifiers to predict "fertile = NDVI > 0.4"
from NDVI itself, and got 100 % accuracy: the label was a copy of the input (data leakage).
There is no ground truth for fertility here (no soil samples, no yields), so a supervised
model cannot be evaluated honestly. What the data can support is unsupervised:

1. standardise a few complementary indices (vigour, chlorophyll, leaf water);
2. group the pixels with k-means;
3. name the groups by their mean NDVI (low / medium / high vigour).

Farmers use exactly these zones to vary irrigation, fertiliser or scouting across a field.
Simple rule-of-thumb flags (low vegetation, water stress) are kept separate and explicit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURES = ["NDVI", "NDRE", "NDMI"]
ZONE_NAMES = {2: ["low", "high"], 3: ["low", "medium", "high"], 4: ["very low", "low", "medium", "high"]}

# Rules of thumb (they depend on the crop and season: tune them, don't trust them blindly)
LOW_VEGETATION_NDVI = 0.3
WATER_STRESS_NDMI = 0.0


def management_zones(df: pd.DataFrame, k: int = 3, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """Add a `zone` column (0 = lowest vigour) and return a per-zone summary."""
    if k not in ZONE_NAMES:
        raise ValueError(f"k must be one of {sorted(ZONE_NAMES)}")
    data = df.dropna(subset=FEATURES).copy()
    if len(data) < k * 5:
        raise ValueError("not enough valid pixels to build zones")
    X = StandardScaler().fit_transform(data[FEATURES])
    model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)

    # order clusters by mean NDVI so that zone 0 is always the weakest
    order = data.groupby(model.labels_)["NDVI"].mean().sort_values().index
    rank = {cluster: i for i, cluster in enumerate(order)}
    data["zone"] = [rank[c] for c in model.labels_]
    data["zone_name"] = data["zone"].map(dict(enumerate(ZONE_NAMES[k])))
    data["low_vegetation"] = data["NDVI"] < LOW_VEGETATION_NDVI
    data["water_stress"] = data["NDMI"] < WATER_STRESS_NDMI

    summary = {
        "k": k,
        "silhouette": round(float(silhouette_score(X, model.labels_)), 3),
        "zones": [
            {
                "zone": int(z),
                "name": ZONE_NAMES[k][z],
                "share_pct": round(100 * len(g) / len(data), 1),
                **{f"mean_{f}": round(float(g[f].mean()), 3) for f in FEATURES},
            }
            for z, g in data.groupby("zone")
        ],
        "flags": {
            "low_vegetation_pct": round(100 * float(data["low_vegetation"].mean()), 1),
            "water_stress_pct": round(100 * float(data["water_stress"].mean()), 1),
        },
    }
    return data, summary


def best_k(df: pd.DataFrame, candidates=(2, 3, 4), seed: int = 0) -> int:
    """The number of zones with the best silhouette score (how well separated they are)."""
    scores = {k: management_zones(df, k, seed)[1]["silhouette"] for k in candidates}
    return int(max(scores, key=scores.get))


def leakage_demo(df: pd.DataFrame) -> float:
    """Why the first version scored 100 %: predict a label made from the very same column.

    Returns the cross-validated accuracy of a decision tree predicting `NDVI > 0.4` from NDVI.
    It is ~1.0 on any data, which proves nothing about the field.
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.tree import DecisionTreeClassifier

    data = df.dropna(subset=["NDVI"])
    y = (data["NDVI"] > 0.4).astype(int)
    if y.nunique() < 2:
        return 1.0
    return float(cross_val_score(DecisionTreeClassifier(max_depth=2), data[["NDVI"]], y, cv=5).mean())
