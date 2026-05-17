"""Engine Detail page: status, RUL gauge, sensor, prediction and alert tabs."""

from typing import Any, Dict, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components._theme import (
    GLOBAL_CSS,
    empty_state,
    page_header,
    status_badge,
)
from dashboard.components.alerts import alerts_timeline
from dashboard.components.charts import (
    prediction_confidence_band,
    rul_gauge,
    sensor_timeseries,
    shap_bar_chart,
)
from dashboard.components.sidebar import render_sidebar
from dashboard.utils.api_client import SentinelAPIClient


def _attention_heatmap(matrix: List[List[float]]) -> go.Figure:
    """Build a light-themed attention heatmap.

    Args:
        matrix: A 2D attention-weight matrix.

    Returns:
        A Plotly heatmap figure.
    """
    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            colorscale=[[0.0, "#EEF2FF"], [0.5, "#A5B4FC"], [1.0, "#6366F1"]],
        )
    )
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font={"family": "Inter, sans-serif", "color": "#6B7280"},
        title={"text": "Attention Heatmap", "font": {"color": "#111827"}},
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
    )
    return fig


def render() -> None:
    """Render the full Engine Detail page."""
    st.set_page_config(page_title="Engine Detail", page_icon="✈️", layout="wide")
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    render_sidebar()
    st.markdown(
        page_header(
            "Engine Detail",
            "Per-engine sensor telemetry and health status",
        ),
        unsafe_allow_html=True,
    )

    client = SentinelAPIClient()
    machines = client.get_machines()
    if not machines:
        st.markdown(
            empty_state(
                "🔧", "No engines registered", "Upload sensor data to begin"
            ),
            unsafe_allow_html=True,
        )
        return

    names = [m.get("name", m.get("id", "?")) for m in machines]
    default = st.session_state.get("selected_machine", names[0])
    index = names.index(default) if default in names else 0
    chosen = st.selectbox("Select engine unit", options=names, index=index)
    machine = machines[names.index(chosen)]
    machine_id = machine.get("id", "")

    header = st.columns(4)
    header[0].metric("Name", machine.get("name", "—"))
    header[1].metric("Type", machine.get("type", "—"))
    header[2].metric("Location", machine.get("location", "—"))
    header[3].metric("Installed", str(machine.get("install_date", "—")))

    prediction = machine.get("latest_prediction") or {}
    rul = float(prediction.get("predicted_rul", 0.0))
    risk = prediction.get("risk_level")
    if risk:
        st.markdown(status_badge(risk), unsafe_allow_html=True)
    st.plotly_chart(
        rul_gauge(rul, machine.get("name", "?")), use_container_width=True
    )

    sensors_tab, predictions_tab, alerts_tab = st.tabs(
        ["Sensors", "Predictions", "Alerts"]
    )

    with sensors_tab:
        history = client.get_prediction_history(machine_id)
        if history:
            frame = pd.DataFrame(history)
            numeric = [
                c
                for c in frame.columns
                if pd.api.types.is_numeric_dtype(frame[c])
            ]
            picked = st.multiselect(
                "Sensors / signals",
                options=numeric,
                default=numeric[: min(3, len(numeric))],
            )
            if picked:
                st.plotly_chart(
                    sensor_timeseries(frame, picked),
                    use_container_width=True,
                )
            st.markdown("#### Sensor Readings")
            st.dataframe(
                frame, use_container_width=True, hide_index=True
            )
        else:
            st.info("No sensor history available.")

    with predictions_tab:
        history = client.get_prediction_history(machine_id)
        if history:
            st.plotly_chart(
                prediction_confidence_band(history),
                use_container_width=True,
            )
        explanation: Dict[str, Any] = prediction.get("explanation", {}) or {}
        features = explanation.get("top_features", [])
        if features:
            st.plotly_chart(
                shap_bar_chart(features), use_container_width=True
            )
        summary = explanation.get("summary_text")
        if summary:
            st.markdown("**Explanation**")
            st.info(summary)
        heatmap = explanation.get("attention_heatmap")
        if heatmap:
            st.plotly_chart(
                _attention_heatmap(heatmap), use_container_width=True
            )

    with alerts_tab:
        payload = client.get_alerts()
        alerts = [
            a
            for a in payload.get("alerts", [])
            if a.get("machine_id") == machine_id
        ]
        alerts_timeline(alerts)


render()
