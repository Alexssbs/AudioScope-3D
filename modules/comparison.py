from __future__ import annotations

import pandas as pd


def comparison_table(first: dict, second: dict, first_name: str, second_name: str) -> pd.DataFrame:
    keys = ["rms", "energy", "zero_crossing_rate", "spectral_centroid_hz", "dominant_frequency_hz", "silence_seconds"]
    rows = []
    for key in keys:
        a, b = float(first[key]), float(second[key])
        rows.append({"Métrica": key.replace("_", " ").title(), first_name: a, second_name: b, "Diferencia": b - a})
    return pd.DataFrame(rows)
