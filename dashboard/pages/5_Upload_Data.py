"""Upload Data page: validate, preview and process a sensor CSV."""

from typing import List

import pandas as pd
import streamlit as st

from dashboard.components._theme import GLOBAL_CSS, page_header
from dashboard.components.alerts import alert_banner
from dashboard.components.sidebar import render_sidebar
from dashboard.utils.api_client import SentinelAPIClient

_REQUIRED_COLUMNS: List[str] = [
    "unit_id",
    "cycle",
] + [f"sensor_{i}" for i in range(1, 15)]
_WINDOW = 30


def render() -> None:
    """Render the full Upload Data page."""
    st.set_page_config(page_title="Upload Data", page_icon="📤", layout="wide")
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    render_sidebar()
    st.markdown(
        page_header(
            "Upload Sensor Data", "Validate and ingest CSV telemetry files"
        ),
        unsafe_allow_html=True,
    )

    client = SentinelAPIClient()
    uploaded = st.file_uploader("Sensor CSV file", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV to validate and process it.")
        return

    frame = pd.read_csv(uploaded)
    missing = [c for c in _REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        alert_banner(
            f"Missing required columns: {', '.join(missing)}", "error"
        )
        return
    alert_banner("All required columns present.", "success")

    st.subheader("Summary Statistics")
    st.dataframe(frame.describe(), use_container_width=True)

    st.subheader("Preview")
    st.dataframe(frame.head(10), use_container_width=True)

    if st.button("Process Data"):
        sensor_cols = [c for c in _REQUIRED_COLUMNS if c.startswith("sensor_")]
        results = []
        for unit_id, group in frame.groupby("unit_id"):
            window = group.sort_values("cycle").tail(_WINDOW)
            readings = window[sensor_cols].values.tolist()
            prediction = client.predict_rul(str(unit_id), readings)
            results.append(
                {
                    "unit_id": unit_id,
                    "predicted_rul": prediction.get("predicted_rul"),
                    "risk_level": prediction.get("risk_level"),
                    "risk_score": prediction.get("risk_score"),
                }
            )
        processed = pd.DataFrame(results)
        st.success(f"Processed {len(processed)} units.")
        st.dataframe(processed, use_container_width=True, hide_index=True)
        st.download_button(
            "Download processed CSV",
            data=processed.to_csv(index=False).encode("utf-8"),
            file_name="processed_predictions.csv",
            mime="text/csv",
        )


render()
