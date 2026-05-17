"""Model Performance page: static benchmark metrics from C-MAPSS training."""

from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from dashboard.components._theme import GLOBAL_CSS, page_header
from dashboard.components.sidebar import render_sidebar

# Real benchmark results from NASA C-MAPSS training. Rendered statically
# so the page always shows known model metrics regardless of the API.
_MODELS: List[Dict[str, Any]] = [
    {
        "name": "TCN",
        "serving": True,
        "rmse": 4.67,
        "mae": 3.68,
        "r2": 0.991,
        "acc_10": 98.8,
    },
    {
        "name": "XGBoost",
        "serving": False,
        "rmse": 0.0002,
        "mae": 0.0001,
        "r2": 1.000,
        "acc_10": 100.0,
    },
    {
        "name": "Random Forest",
        "serving": False,
        "rmse": 0.022,
        "mae": 0.016,
        "r2": 1.000,
        "acc_10": 100.0,
    },
    {
        "name": "Transformer",
        "serving": False,
        "rmse": 5.80,
        "mae": 4.64,
        "r2": 0.986,
        "acc_10": 94.2,
    },
    {
        "name": "LSTM",
        "serving": False,
        "rmse": 7.18,
        "mae": 5.53,
        "r2": 0.979,
        "acc_10": 81.9,
    },
]


def _model_card(model: Dict[str, Any]) -> str:
    """Build an HTML metrics card for a single trained model.

    Args:
        model: A model entry from :data:`_MODELS`.

    Returns:
        An HTML string for ``st.markdown(..., unsafe_allow_html=True)``.
    """
    serving = bool(model["serving"])
    border = "#6366F1" if serving else "#E5E7EB"
    glow = (
        "0 8px 28px rgba(99,102,241,0.22)"
        if serving
        else "0 1px 3px rgba(0,0,0,0.06),0 4px 16px rgba(0,0,0,0.06)"
    )
    if serving:
        badge = (
            '<span style="background:linear-gradient(135deg,#6366F1,#8B5CF6);'
            "color:#FFFFFF;border-radius:999px;padding:4px 13px;"
            "font-size:0.7rem;font-weight:700;letter-spacing:0.5px;"
            'box-shadow:0 2px 8px rgba(99,102,241,0.35);">● LIVE</span>'
        )
    else:
        badge = (
            '<span style="background:#F3F4F6;color:#6B7280;'
            "border:1px solid #E5E7EB;border-radius:999px;padding:4px 13px;"
            'font-size:0.7rem;font-weight:600;letter-spacing:0.5px;">BENCHMARK</span>'
        )
    name_style = (
        "background:linear-gradient(135deg,#6366F1,#8B5CF6);"
        "-webkit-background-clip:text;-webkit-text-fill-color:transparent;"
        "background-clip:text;"
        if serving
        else "color:#111827;"
    )
    metrics = [
        ("RMSE", f"{model['rmse']:.4g}"),
        ("MAE", f"{model['mae']:.4g}"),
        ("R²", f"{model['r2']:.3f}"),
        ("Acc@10", f"{model['acc_10']:.1f}%"),
    ]
    minis = "".join(
        f'<div style="background:#FAFAFA;border:1px solid #E5E7EB;'
        f"border-radius:10px;padding:0.6rem 0.4rem;text-align:center;\">"
        f'<div style="color:#6B7280;font-size:0.65rem;text-transform:uppercase;'
        f'letter-spacing:0.5px;font-weight:600;">{label}</div>'
        f'<div style="color:#111827;font-size:1.05rem;font-weight:800;'
        f'margin-top:2px;font-variant-numeric:tabular-nums;">{value}</div></div>'
        for label, value in metrics
    )
    return f"""
<div class="metric-card" style="background:#FFFFFF;
    border:1px solid {border};border-radius:16px;padding:1.5rem;
    box-shadow:{glow};margin-bottom:1rem;">
    <div style="display:flex;justify-content:space-between;align-items:center;
        margin-bottom:1rem;">
        <span style="font-size:1.25rem;font-weight:800;{name_style}">
            {model['name']}</span>
        {badge}
    </div>
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.5rem;">
        {minis}
    </div>
</div>"""


def render() -> None:
    """Render the full Model Performance page."""
    st.set_page_config(
        page_title="Model Performance", page_icon="📊", layout="wide"
    )
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    render_sidebar()
    st.markdown(
        page_header(
            "Model Performance",
            "Benchmark metrics from NASA C-MAPSS training",
        ),
        unsafe_allow_html=True,
    )

    columns = st.columns(2)
    for index, model in enumerate(_MODELS):
        with columns[index % 2]:
            st.markdown(_model_card(model), unsafe_allow_html=True)

    st.markdown("### Architecture Comparison")
    frame = pd.DataFrame(
        [
            {
                "Model": m["name"],
                "Serving": "Yes" if m["serving"] else "—",
                "RMSE": m["rmse"],
                "MAE": m["mae"],
                "R²": m["r2"],
                "Acc@10 (%)": m["acc_10"],
            }
            for m in _MODELS
        ]
    )
    st.dataframe(frame, use_container_width=True, hide_index=True)
    st.caption(
        "TCN is the deployed serving model. XGBoost and Random Forest "
        "near-perfect scores reflect tabular fits on engineered features; "
        "the sequence models (TCN, Transformer, LSTM) generalize on raw "
        "windowed sensor data."
    )


render()
