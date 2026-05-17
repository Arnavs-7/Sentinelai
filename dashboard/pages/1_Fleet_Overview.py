"""Fleet Overview page: fleet KPIs, RUL bar chart and an engine table."""

from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from dashboard.components._theme import (
    GLOBAL_CSS,
    empty_state,
    metric_card,
    page_header,
)
from dashboard.components.charts import rul_distribution
from dashboard.components.sidebar import render_sidebar
from dashboard.utils.api_client import SentinelAPIClient

_RISK_ROW_COLORS = {
    "CRITICAL": "#FEE2E2",
    "WARNING": "#FEF3C7",
    "HEALTHY": "#D1FAE5",
}


def _row_style(machines: List[Dict[str, Any]]) -> Any:
    """Build a row-coloring function for the engine table.

    Args:
        machines: Engine dicts in the same order as the table rows.

    Returns:
        A callable for ``Styler.apply`` that colors rows by risk.
    """
    levels = [
        (m.get("latest_prediction") or {}).get("risk_level", "")
        for m in machines
    ]

    def _apply(row: Any) -> List[str]:
        color = _RISK_ROW_COLORS.get(levels[row.name], "")
        background = f"background-color:{color}" if color else ""
        return [background] * len(row)

    return _apply


def render() -> None:
    """Render the full Fleet Overview page."""
    st.set_page_config(page_title="Fleet Overview", page_icon="🛩️", layout="wide")
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    render_sidebar()
    st.markdown(
        page_header(
            "Fleet Overview", "Real-time engine health across your fleet"
        ),
        unsafe_allow_html=True,
    )

    client = SentinelAPIClient()
    health = client.get_fleet_health()
    machines = client.get_machines()

    total = int(health.get("total", len(machines)))
    healthy = int(health.get("healthy", 0))
    warning = int(health.get("warning", 0))
    critical = int(health.get("critical", 0))

    cards = [
        ("Total Engines", str(total), "fleet size", "#6366F1"),
        ("Healthy", str(healthy), "RUL ≥ 100", "#10B981"),
        ("Warning", str(warning), "30 ≤ RUL < 100", "#F59E0B"),
        ("Critical", str(critical), "RUL < 30", "#EF4444"),
    ]
    for col, (title, value, sub, color) in zip(st.columns(4), cards):
        with col:
            st.markdown(
                metric_card(title, value, sub, color), unsafe_allow_html=True
            )

    if not machines:
        st.markdown(
            empty_state(
                "✈️",
                "No engine data available",
                "Demo data loads automatically when the API starts.",
            ),
            unsafe_allow_html=True,
        )
        return

    st.markdown("### RUL by Engine")
    st.plotly_chart(rul_distribution(machines), use_container_width=True)

    st.markdown("### Engine Units")
    rows: List[Dict[str, Any]] = []
    for machine in machines:
        prediction = machine.get("latest_prediction") or {}
        rows.append(
            {
                "Engine": machine.get("name", "?"),
                "Type": machine.get("type", "—"),
                "Location": machine.get("location", "—"),
                "RUL": round(float(prediction.get("predicted_rul", 0.0)), 1),
                "Risk": prediction.get("risk_level", "UNKNOWN"),
                "Last Updated": prediction.get("timestamp", "—"),
            }
        )
    frame = pd.DataFrame(rows)
    styler = frame.style.apply(_row_style(machines), axis=1)
    st.dataframe(styler, use_container_width=True, hide_index=True)

    selected = st.selectbox(
        "Inspect engine unit",
        options=[m.get("name", m.get("id", "?")) for m in machines],
    )
    if st.button("Open in Engine Detail"):
        st.session_state["selected_machine"] = selected
        st.switch_page("pages/2_Machine_Detail.py")


render()
