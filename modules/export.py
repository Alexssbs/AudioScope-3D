from __future__ import annotations

import io
import json

import pandas as pd


def features_json(features: dict) -> str:
    return json.dumps(features, ensure_ascii=False, indent=2)


def features_csv(features: dict) -> bytes:
    return pd.DataFrame([features]).to_csv(index=False).encode("utf-8")


def features_excel(features: dict) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame([features]).to_excel(writer, index=False, sheet_name="Análisis")
    return output.getvalue()
