"""Predictions page: run RUL inference from a CSV upload or manual input."""

import io
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from dashboard.components._theme import GLOBAL_CSS, page_header
from dashboard.components.cards import risk_badge
from dashboard.components.charts import rul_gauge, shap_bar_chart
from dashboard.components.sidebar import render_sidebar
from dashboard.utils.api_client import SentinelAPIClient

_WINDOW = 30
_N_SENSORS = 14

# Maintenance guidance shown on the PDF report, keyed by risk level.
_RECOMMENDED_ACTION = {
    "CRITICAL": "Schedule immediate inspection; ground the engine until serviced.",
    "WARNING": "Plan maintenance within the next 30 operating cycles.",
    "HEALTHY": "Continue routine monitoring; no action required.",
}


def _pdf_report(machine_id: str, result: Dict[str, Any]) -> bytes:
    """Render a one-page PDF prediction report.

    Args:
        machine_id: Identifier of the predicted machine.
        result: The prediction response dict.

    Returns:
        The PDF document as bytes.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(72, 720, "SentinelAI — RUL Prediction Report")
    pdf.setFont("Helvetica", 11)
    lines = [
        f"Generated: {datetime.utcnow().isoformat()}",
        f"Machine: {machine_id}",
        f"Predicted RUL: {result.get('predicted_rul', 'N/A')} cycles",
        f"Risk level: {result.get('risk_level', 'N/A')}",
        f"Risk score: {result.get('risk_score', 'N/A')}",
    ]
    interval = result.get("confidence_interval", {})
    if interval:
        lines.append(
            f"Confidence interval: "
            f"[{interval.get('lower')}, {interval.get('upper')}]"
        )
    explanation = result.get("explanation", {}) or {}
    if explanation.get("summary_text"):
        lines.append("")
        lines.append("Explanation:")
        lines.append(str(explanation["summary_text"]))
    risk = str(result.get("risk_level", "HEALTHY")).upper()
    lines.append("")
    lines.append("Recommended action:")
    lines.append(_RECOMMENDED_ACTION.get(risk, _RECOMMENDED_ACTION["HEALTHY"]))
    y = 680
    for line in lines:
        pdf.drawString(72, y, line[:95])
        y -= 20
    pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def _show_result(machine_id: str, result: Dict[str, Any]) -> None:
    """Render the prediction result block.

    Args:
        machine_id: Identifier of the predicted machine.
        result: The prediction response dict.
    """
    if not result:
        st.error("Prediction failed. Check the API service.")
        return
    st.markdown(
        risk_badge(result.get("risk_level", "UNKNOWN")),
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        rul_gauge(float(result.get("predicted_rul", 0.0)), machine_id),
        use_container_width=True,
    )
    explanation = result.get("explanation", {}) or {}
    features = explanation.get("top_features", [])
    if features:
        st.plotly_chart(shap_bar_chart(features), use_container_width=True)
    if explanation.get("summary_text"):
        st.info(explanation["summary_text"])
    st.download_button(
        "Download PDF report",
        data=_pdf_report(machine_id, result),
        file_name=f"rul_report_{machine_id}.pdf",
        mime="application/pdf",
    )


def render() -> None:
    """Render the full Predictions page."""
    st.set_page_config(page_title="Predictions", page_icon="🔮", layout="wide")
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    render_sidebar()
    st.markdown(
        page_header("RUL Predictions", "Run inference on new sensor data"),
        unsafe_allow_html=True,
    )

    client = SentinelAPIClient()
    upload_tab, manual_tab = st.tabs(["Upload CSV", "Manual Input"])

    with upload_tab:
        machine_id = st.text_input("Machine ID", key="csv_machine_id")
        uploaded = st.file_uploader("Sensor CSV", type=["csv"])
        if uploaded is not None:
            frame = pd.read_csv(uploaded)
            numeric = frame.select_dtypes(include="number")
            if numeric.shape[1] < 1:
                st.error("CSV contains no numeric sensor columns.")
            else:
                st.dataframe(frame.head(10), use_container_width=True)
                st.caption(
                    f"{numeric.shape[0]} rows x {numeric.shape[1]} sensors."
                )
                if st.button("Run Prediction", key="csv_run"):
                    readings: List[List[float]] = (
                        numeric.tail(_WINDOW).values.tolist()
                    )
                    result = client.predict_rul(machine_id, readings)
                    _show_result(machine_id, result)

    with manual_tab:
        machine_id = st.text_input("Machine ID", key="manual_machine_id")
        readings: List[List[float]] = []
        for cycle in range(_WINDOW):
            with st.expander(f"Cycle {cycle + 1}"):
                cols = st.columns(_N_SENSORS)
                row = [
                    cols[s].number_input(
                        f"S{s + 1}",
                        key=f"m_{cycle}_{s}",
                        value=0.0,
                        format="%.4f",
                    )
                    for s in range(_N_SENSORS)
                ]
                readings.append(row)
        if st.button("Run Prediction", key="manual_run"):
            result = client.predict_rul(machine_id, readings)
            _show_result(machine_id, result)


render()
