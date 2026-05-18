"""SentinelAI dashboard entry point: landing page, theme and sidebar."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

# st.set_page_config MUST be the first Streamlit call in the script.
st.set_page_config(
    page_title="SentinelAI",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from dashboard.components._theme import GLOBAL_CSS, empty_state
from dashboard.components.sidebar import render_sidebar
from dashboard.utils.api_client import SentinelAPIClient

_STAT_TILES = [
    ("TCN", "Serving Model", "#6366F1"),
    ("98.8%", "Accuracy @10", "#10B981"),
    ("SHAP", "Explainable AI", "#8B5CF6"),
    ("12", "Engine Fleet", "#F59E0B"),
]


def _stat_tile(value: str, label: str, color: str) -> str:
    """Build one hero stat tile with a colored top border and hover lift.

    Returned as a single line so Streamlit's markdown does not mistake the
    indented HTML for a code block.

    Args:
        value: The large headline figure or label.
        label: The muted caption beneath the value.
        color: Accent color for the 3px top border and value text.

    Returns:
        An HTML string for the tile.
    """
    return (
        f'<div class="metric-card" style="background:#FFFFFF;'
        f"border:1px solid #E5E7EB;border-top:3px solid {color};"
        f"border-radius:12px;padding:1.05rem;text-align:center;"
        f'box-shadow:0 1px 3px rgba(0,0,0,0.06);">'
        f'<div style="color:{color};font-size:1.55rem;font-weight:800;'
        f'letter-spacing:-0.5px;">{value}</div>'
        f'<div style="color:#6B7280;font-size:0.75rem;margin-top:4px;">'
        f"{label}</div></div>"
    )


_HERO = f"""
<div style="background:linear-gradient(90deg,#6366F1,#8B5CF6,#EC4899,
    #8B5CF6,#6366F1);background-size:200% 100%;
    animation:shimmer 5s linear infinite;
    border-radius:20px;padding:1.5px;margin-bottom:2rem;
    box-shadow:0 8px 30px rgba(99,102,241,0.16);">
  <div style="background:rgba(255,255,255,0.85);
      backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);
      border:1px solid rgba(255,255,255,0.9);
      border-radius:18.5px;padding:2.6rem;
      background-image:radial-gradient(ellipse at 80% 0%,#EDE9FE 0%,
      transparent 60%);">
    <div style="display:flex;align-items:center;gap:1.2rem;margin-bottom:1.5rem;">
        <div style="width:58px;height:58px;
            background:linear-gradient(135deg,#6366F1 0%,#8B5CF6 100%);
            border-radius:16px;display:flex;align-items:center;
            justify-content:center;font-size:1.8rem;
            box-shadow:0 6px 18px rgba(99,102,241,0.35);">✈️</div>
        <div>
            <div style="font-size:2.2rem;font-weight:900;letter-spacing:-1.2px;
                background:linear-gradient(135deg,#6366F1 0%,#8B5CF6 50%,
                #EC4899 100%);-webkit-background-clip:text;
                -webkit-text-fill-color:transparent;background-clip:text;
                display:inline-block;">SentinelAI</div>
            <div style="color:#6B7280;font-size:0.85rem;">
                Aviation Turbofan Engine · Predictive Maintenance Platform
            </div>
        </div>
    </div>
    <p style="color:#4B5563;line-height:1.8;margin-bottom:1.8rem;">
        Real-time ML platform predicting
        <strong style="color:#4F46E5;">Remaining Useful Life (RUL)</strong>
        of turbofan engines. Ensemble of
        <strong style="color:#4F46E5;">LSTM · Transformer · TCN</strong>
        trained on NASA C-MAPSS dataset.
        <strong style="color:#059669;">SHAP explainability</strong>
        surfaces which sensors are driving each alert.
    </p>
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;">
        {"".join(_stat_tile(v, lbl, c) for v, lbl, c in _STAT_TILES)}
    </div>
  </div>
</div>
"""

def render() -> None:
    """Render the SentinelAI landing page."""
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    render_sidebar()

    st.markdown(_HERO, unsafe_allow_html=True)

    client = SentinelAPIClient()
    health = client.get_fleet_health()
    if health:
        st.markdown("### Fleet Snapshot")
        cols = st.columns(4)
        cols[0].metric("Total Engine Units", health.get("total", 0))
        cols[1].metric("Healthy", health.get("healthy", 0))
        cols[2].metric("Warning", health.get("warning", 0))
        cols[3].metric("Critical", health.get("critical", 0))
        st.caption("Select a page from the sidebar to explore the fleet.")
    else:
        st.markdown(
            empty_state(
                "🛰️",
                "API service unreachable",
                "Start the SentinelAI backend to load the fleet.",
            ),
            unsafe_allow_html=True,
        )


render()
