"""Alert banner and timeline components for the SentinelAI dashboard."""

from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st

_SEVERITY_COLORS = {
    "critical": "#EF4444",
    "warning": "#F59E0B",
    "info": "#6366F1",
    "success": "#10B981",
}


def alert_banner(message: str, severity: str) -> None:
    """Render a severity-styled inline banner.

    Args:
        message: The text to display.
        severity: One of ``error``/``critical``, ``warning``, ``info``
            or ``success``; unknown values fall back to ``info``.
    """
    level = (severity or "info").lower()
    if level in ("error", "critical"):
        st.error(message)
    elif level == "warning":
        st.warning(message)
    elif level == "success":
        st.success(message)
    else:
        st.info(message)


def alerts_timeline(alerts: List[Dict[str, Any]]) -> None:
    """Render a Plotly timeline of alerts grouped by severity.

    Args:
        alerts: Alert dicts with ``timestamp``/``severity``/``message``.
    """
    if not alerts:
        st.info("No alerts to display.")
        return

    rows: List[Dict[str, Any]] = []
    for alert in alerts:
        timestamp = pd.to_datetime(alert.get("timestamp"), errors="coerce")
        if pd.isna(timestamp):
            continue
        severity = str(alert.get("severity", "info")).lower()
        rows.append(
            {
                "timestamp": timestamp,
                "severity": severity,
                "machine_id": alert.get("machine_id", "?"),
                "message": alert.get("message", ""),
            }
        )
    if not rows:
        st.info("No alerts with valid timestamps.")
        return

    frame = pd.DataFrame(rows)
    fig = px.scatter(
        frame,
        x="timestamp",
        y="severity",
        color="severity",
        color_discrete_map=_SEVERITY_COLORS,
        hover_data=["machine_id", "message"],
        category_orders={"severity": ["critical", "warning", "info", "success"]},
    )
    fig.update_traces(marker={"size": 13, "symbol": "diamond"})
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0A0A0F",
        plot_bgcolor="#111118",
        font={"family": "Inter, sans-serif", "color": "#94A3B8"},
        title={"text": "Alert Timeline", "font": {"color": "#F1F5F9"}},
        xaxis={"title": "Time", "gridcolor": "#1E1E2E", "linecolor": "#2A2A4A"},
        yaxis={"title": "Severity", "gridcolor": "#1E1E2E", "linecolor": "#2A2A4A"},
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
    )
    st.plotly_chart(fig, use_container_width=True)
