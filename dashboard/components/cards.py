"""Reusable Streamlit card and badge components (dark theme)."""

from typing import Any, Dict

import streamlit as st

_RISK_COLORS = {
    "HEALTHY": "#10B981",
    "WARNING": "#F59E0B",
    "CRITICAL": "#EF4444",
    "FAILED": "#EF4444",
}
# Gradient end-color per risk accent, for flowing gradient value text.
_RISK_GRADIENTS = {
    "#10B981": "#34D399",
    "#F59E0B": "#FBBF24",
    "#EF4444": "#F87171",
}
# Dark status pills: (translucent-bg, border-color, text-color).
_RISK_TINTS = {
    "HEALTHY": ("rgba(16,185,129,0.13)", "rgba(16,185,129,0.4)", "#10B981"),
    "WARNING": ("rgba(245,158,11,0.13)", "rgba(245,158,11,0.4)", "#F59E0B"),
    "CRITICAL": ("rgba(239,68,68,0.13)", "rgba(239,68,68,0.4)", "#EF4444"),
    "FAILED": ("rgba(239,68,68,0.13)", "rgba(239,68,68,0.4)", "#EF4444"),
}
_SEVERITY_COLORS = {
    "critical": "#EF4444",
    "warning": "#F59E0B",
    "info": "#6366F1",
    "success": "#10B981",
}


def kpi_card(
    title: str, value: str, delta: str, color: str, pulse: bool = False
) -> None:
    """Render a KPI card with a colored top border.

    Args:
        title: The metric label.
        value: The primary metric value.
        delta: A secondary delta or context string.
        color: CSS color for the top border and value text.
        pulse: When ``True``, the card pulses to flag a critical state.
    """
    animation = "animation:pulse 1.6s infinite;" if pulse else ""
    end = _RISK_GRADIENTS.get(color, color)
    st.markdown(
        f"""
        <div class="metric-card" style="
            background:#1A1A2E;
            border:1px solid #2A2A4A;
            border-top:4px solid {color};
            border-radius:12px;
            padding:1.3rem 1.4rem;
            margin-bottom:8px;
            box-shadow:0 4px 24px rgba(0,0,0,0.4);
            {animation}">
            <div style="display:flex;align-items:center;gap:7px;">
                <span style="width:8px;height:8px;border-radius:50%;
                    background:{color};display:inline-block;
                    box-shadow:0 0 8px {color};"></span>
                <span style="color:#94A3B8;font-size:0.75rem;font-weight:600;
                    text-transform:uppercase;letter-spacing:0.6px;">{title}</span>
            </div>
            <div style="font-size:2.1rem;font-weight:800;margin-top:5px;
                font-variant-numeric:tabular-nums;
                background:linear-gradient(135deg,{color} 0%,{end} 100%);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                background-clip:text;">{value}</div>
            <div style="color:#475569;font-size:0.78rem;">{delta}</div>
        </div>
        <style>@keyframes pulse {{0%,100%{{opacity:1;}}50%{{opacity:0.6;}}}}</style>
        """,
        unsafe_allow_html=True,
    )


def risk_badge(risk_level: str) -> str:
    """Return a color-coded HTML badge for a risk level.

    Args:
        risk_level: One of ``HEALTHY``/``WARNING``/``CRITICAL``/``FAILED``.

    Returns:
        An HTML ``span`` string suitable for ``st.markdown``.
    """
    level = (risk_level or "UNKNOWN").upper()
    bg, border, fg = _RISK_TINTS.get(
        level, ("rgba(71,85,105,0.18)", "#2A2A4A", "#94A3B8")
    )
    return (
        f'<span style="background:{bg};border:1px solid {border};'
        f"color:{fg};display:inline-flex;align-items:center;gap:5px;"
        f"padding:4px 13px;border-radius:999px;font-size:0.74rem;"
        f'font-weight:700;letter-spacing:0.4px;">'
        f'<span style="font-size:0.6rem;">●</span>{level}</span>'
    )


def machine_status_card(machine: Dict[str, Any]) -> None:
    """Render a compact status card for a single engine unit.

    Args:
        machine: Engine dict carrying an optional ``latest_prediction``.
    """
    prediction = machine.get("latest_prediction") or {}
    risk = prediction.get("risk_level", "UNKNOWN")
    rul = prediction.get("predicted_rul")
    rul_text = f"{rul:.1f} cycles" if isinstance(rul, (int, float)) else "—"
    updated = prediction.get("timestamp", "—")
    st.markdown(
        f"""
        <div class="metric-card" style="
            background:#1A1A2E;
            border:1px solid #2A2A4A;
            border-radius:12px;
            padding:15px 18px;
            margin-bottom:10px;
            box-shadow:0 4px 24px rgba(0,0,0,0.4);">
            <div style="display:flex;justify-content:space-between;
                align-items:center;">
                <span style="color:#F1F5F9;font-size:1.05rem;font-weight:700;">
                    {machine.get('name', '?')}</span>
                {risk_badge(risk)}
            </div>
            <div style="color:#94A3B8;font-size:0.82rem;margin-top:4px;">
                {machine.get('type', 'Unknown')} ·
                {machine.get('location', 'Unknown')}</div>
            <div style="color:#F1F5F9;font-size:0.9rem;margin-top:6px;">
                RUL: <b style="color:#A5B4FC;">{rul_text}</b></div>
            <div style="color:#475569;font-size:0.72rem;margin-top:4px;">
                Updated: {updated}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def alert_card(alert: Dict[str, Any]) -> None:
    """Render an alert card colored by severity.

    Args:
        alert: Alert dict with ``severity``/``message``/``timestamp``.
    """
    severity = str(alert.get("severity", "info")).lower()
    color = _SEVERITY_COLORS.get(severity, "#6366F1")
    st.markdown(
        f"""
        <div style="
            background:#1A1A2E;
            border:1px solid #2A2A4A;
            border-left:4px solid {color};
            border-radius:0 14px 14px 0;
            padding:13px 17px;
            margin-bottom:8px;
            box-shadow:0 4px 24px rgba(0,0,0,0.4);">
            <div style="display:flex;justify-content:space-between;">
                <span style="color:{color};font-size:0.8rem;font-weight:700;">
                    {severity.upper()} · {alert.get('alert_type', 'alert')}</span>
                <span style="color:#475569;font-size:0.7rem;">
                    {alert.get('timestamp', '')}</span>
            </div>
            <div style="color:#F1F5F9;font-size:0.9rem;margin-top:4px;">
                {alert.get('message', '')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
