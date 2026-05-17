"""Shared branded sidebar: navigation links and live API status.

Streamlit's auto-generated page navigation is hidden by the global CSS,
so this module supplies the single branded navigation rendered on every
page of the dashboard.
"""

from datetime import datetime

import streamlit as st

from dashboard.utils.api_client import SentinelAPIClient

_NAV_PAGES = [
    ("app.py", "Home", "🏠"),
    ("pages/1_Fleet_Overview.py", "Fleet Overview", "🛩️"),
    ("pages/2_Machine_Detail.py", "Engine Detail", "✈️"),
    ("pages/3_Predictions.py", "RUL Predictions", "🔮"),
    ("pages/4_Model_Performance.py", "Model Performance", "📊"),
    ("pages/5_Upload_Data.py", "Upload Data", "📤"),
]


def render_sidebar() -> None:
    """Render the branded sidebar with navigation links and API status."""
    with st.sidebar:
        st.markdown(
            "<div style='padding:0.5rem 0 1.1rem;'>"
            "<div style='font-size:1.55rem;font-weight:800;"
            "letter-spacing:-0.6px;"
            "background:linear-gradient(135deg,#6366F1 0%,#8B5CF6 100%);"
            "-webkit-background-clip:text;-webkit-text-fill-color:transparent;"
            "background-clip:text;display:inline-block;'>✈️ SentinelAI</div>"
            "<div style='color:#9CA3AF;font-size:0.78rem;font-weight:500;'>"
            "Predictive Maintenance</div></div>",
            unsafe_allow_html=True,
        )
        for path, label, icon in _NAV_PAGES:
            st.page_link(path, label=label, icon=icon)
        st.markdown("<hr/>", unsafe_allow_html=True)

        online = bool(SentinelAPIClient().get_fleet_health())
        color = "#10B981" if online else "#EF4444"
        label = "API Online" if online else "API Offline"
        pulse = (
            "animation:pulseDot 1.4s ease-in-out infinite;" if online else ""
        )
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;'>"
            f"<span style='width:9px;height:9px;border-radius:50%;"
            f"background:{color};display:inline-block;box-shadow:0 0 6px {color};"
            f"{pulse}'></span>"
            f"<span style='color:{color};font-weight:700;font-size:0.9rem;'>"
            f"{label}</span></div>",
            unsafe_allow_html=True,
        )
        st.caption(
            f"Last refresh: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )
