"""Plotly chart builders for the SentinelAI dashboard (light theme)."""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

_TEMPLATE = "plotly_white"
_PAPER_BG = "#FFFFFF"
_PLOT_BG = "#FFFFFF"
_GRID = "#F3F4F6"
_AXIS = "#6B7280"
_PRIMARY = "#6366F1"
_VIOLET = "#8B5CF6"
# Continuous indigo→violet scale for gradient-filled bars/histograms.
_GRADIENT_SCALE = [[0.0, "#6366F1"], [1.0, "#8B5CF6"]]
_RISK_COLORS = {
    "HEALTHY": "#10B981",
    "WARNING": "#F59E0B",
    "CRITICAL": "#EF4444",
    "FAILED": "#EF4444",
}


def _style(fig: go.Figure) -> go.Figure:
    """Apply the shared light-theme styling to a figure.

    Args:
        fig: The figure to style in place.

    Returns:
        The same figure, restyled.
    """
    fig.update_layout(
        template=_TEMPLATE,
        paper_bgcolor=_PAPER_BG,
        plot_bgcolor=_PLOT_BG,
        font={"family": "Inter, sans-serif", "color": _AXIS},
        xaxis={"gridcolor": _GRID, "linecolor": "#E5E7EB", "zerolinecolor": _GRID},
        yaxis={"gridcolor": _GRID, "linecolor": "#E5E7EB", "zerolinecolor": _GRID},
        legend={"font": {"color": _AXIS}},
        margin={"l": 20, "r": 20, "t": 44, "b": 20},
    )
    # Style the title only when the figure actually carries title text, so
    # charts without a title are not given a stray "undefined" label.
    if fig.layout.title and fig.layout.title.text:
        fig.update_layout(title={"font": {"color": "#111827", "size": 15}})
    return fig


def rul_gauge(rul: float, machine_name: str) -> go.Figure:
    """Build a 0-150 RUL indicator gauge.

    Args:
        rul: Predicted remaining useful life in cycles.
        machine_name: Name shown in the gauge title.

    Returns:
        A Plotly indicator figure.
    """
    if rul < 30:
        bar_color = "#EF4444"
    elif rul < 100:
        bar_color = "#F59E0B"
    else:
        bar_color = "#10B981"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=max(0.0, min(150.0, rul)),
            title={"text": f"RUL — {machine_name}"},
            number={"suffix": " cycles", "font": {"color": "#111827"}},
            gauge={
                "axis": {"range": [0, 150], "tickcolor": _AXIS},
                "bar": {"color": bar_color},
                "steps": [
                    {"range": [0, 30], "color": "#FEE2E2"},
                    {"range": [30, 100], "color": "#FEF3C7"},
                    {"range": [100, 150], "color": "#D1FAE5"},
                ],
                "bordercolor": "#E5E7EB",
                "threshold": {
                    "line": {"color": "#111827", "width": 3},
                    "value": rul,
                },
            },
        )
    )
    return _style(fig)


def sensor_timeseries(
    df: pd.DataFrame,
    sensors: List[str],
    anomaly_points: Optional[Dict[str, List[int]]] = None,
) -> go.Figure:
    """Build a multi-line sensor time-series chart.

    Args:
        df: Frame indexed by cycle with one column per sensor.
        sensors: Sensor column names to plot.
        anomaly_points: Optional mapping of sensor to anomalous row
            indices, drawn as red ``X`` markers.

    Returns:
        A Plotly figure with one trace per sensor.
    """
    fig = go.Figure()
    x_values = df.index.tolist()
    for sensor in sensors:
        if sensor not in df.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=df[sensor].tolist(),
                mode="lines",
                name=sensor,
            )
        )
        if anomaly_points and sensor in anomaly_points:
            idx = [i for i in anomaly_points[sensor] if 0 <= i < len(df)]
            fig.add_trace(
                go.Scatter(
                    x=[x_values[i] for i in idx],
                    y=[df[sensor].iloc[i] for i in idx],
                    mode="markers",
                    name=f"{sensor} anomaly",
                    marker={"symbol": "x", "size": 11, "color": "#EF4444"},
                    showlegend=False,
                )
            )
    fig.update_layout(
        title="Sensor Readings",
        xaxis_title="Cycle",
        yaxis_title="Value",
    )
    return _style(fig)


def fleet_treemap(machines: List[Dict[str, Any]]) -> go.Figure:
    """Build a treemap of the fleet colored by risk score.

    Args:
        machines: Machine dicts carrying ``latest_prediction``.

    Returns:
        A Plotly treemap figure.
    """
    rows: List[Dict[str, Any]] = []
    for machine in machines:
        prediction = machine.get("latest_prediction") or {}
        rows.append(
            {
                "name": machine.get("name", machine.get("id", "?")),
                "type": machine.get("type", "Unknown"),
                "risk_score": float(prediction.get("risk_score", 0.0)),
                "importance": max(
                    1.0, 150.0 - float(prediction.get("predicted_rul", 75.0))
                ),
            }
        )
    if not rows:
        return _style(go.Figure())
    frame = pd.DataFrame(rows)
    fig = px.treemap(
        frame,
        path=[px.Constant("Fleet"), "type", "name"],
        values="importance",
        color="risk_score",
        color_continuous_scale="RdYlGn_r",
        range_color=[0.0, 1.0],
    )
    fig.update_layout(title="Fleet Risk Map")
    return _style(fig)


def rul_distribution(machines: List[Dict[str, Any]]) -> go.Figure:
    """Build a horizontal bar chart of RUL per machine.

    Args:
        machines: Machine dicts carrying ``latest_prediction``.

    Returns:
        A Plotly bar figure.
    """
    names: List[str] = []
    ruls: List[float] = []
    colors: List[str] = []
    for machine in machines:
        prediction = machine.get("latest_prediction") or {}
        names.append(machine.get("name", machine.get("id", "?")))
        ruls.append(float(prediction.get("predicted_rul", 0.0)))
        colors.append(
            _RISK_COLORS.get(prediction.get("risk_level", ""), _PRIMARY)
        )
    order = sorted(range(len(ruls)), key=lambda i: ruls[i])
    fig = go.Figure(
        go.Bar(
            x=[ruls[i] for i in order],
            y=[names[i] for i in order],
            orientation="h",
            marker={"color": [colors[i] for i in order]},
        )
    )
    fig.update_layout(
        title="Predicted RUL by Machine",
        xaxis_title="RUL (cycles)",
        yaxis_title="Machine",
    )
    return _style(fig)


def shap_bar_chart(features: List[Dict[str, Any]]) -> go.Figure:
    """Build a horizontal bar chart of feature contributions.

    Args:
        features: Feature dicts with ``name`` and ``contribution``.

    Returns:
        A Plotly bar figure; green bars push RUL up, red push it down.
    """
    ordered = sorted(
        features, key=lambda f: abs(float(f.get("contribution", 0.0)))
    )
    names = [f.get("name", "?") for f in ordered]
    values = [float(f.get("contribution", 0.0)) for f in ordered]
    colors = ["#10B981" if v >= 0 else "#EF4444" for v in values]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker={"color": colors},
        )
    )
    fig.update_layout(
        title="Feature Contributions",
        xaxis_title="Contribution to RUL",
        yaxis_title="Feature",
    )
    return _style(fig)


def prediction_confidence_band(history: List[Dict[str, Any]]) -> go.Figure:
    """Build a RUL line chart with a shaded confidence band.

    Args:
        history: History entries with ``timestamp``/``rul`` and
            optional ``confidence_lower``/``confidence_upper``.

    Returns:
        A Plotly figure with the band drawn behind the line.
    """
    if not history:
        return _style(go.Figure())
    times = [h.get("timestamp") for h in history]
    ruls = [float(h.get("rul", 0.0)) for h in history]
    lower = [
        float(h.get("confidence_lower", h.get("rul", 0.0))) for h in history
    ]
    upper = [
        float(h.get("confidence_upper", h.get("rul", 0.0))) for h in history
    ]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=times + times[::-1],
            y=upper + lower[::-1],
            fill="toself",
            fillcolor="rgba(99, 102, 241, 0.08)",
            line={"color": "rgba(0,0,0,0)"},
            name="Confidence",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=times,
            y=ruls,
            mode="lines+markers",
            name="Predicted RUL",
            line={"color": "#6366F1", "width": 3},
            marker={"color": "#6366F1", "size": 7},
        )
    )
    fig.update_layout(
        title="RUL Prediction Confidence",
        xaxis_title="Time",
        yaxis_title="RUL (cycles)",
    )
    return _style(fig)


def model_comparison_scatter(results: pd.DataFrame) -> go.Figure:
    """Build a predicted-vs-actual scatter with a ``y = x`` line.

    Args:
        results: Frame with ``actual`` and ``predicted`` columns.

    Returns:
        A Plotly scatter figure.
    """
    fig = go.Figure()
    if {"actual", "predicted"}.issubset(results.columns) and len(results):
        actual = results["actual"].astype(float)
        predicted = results["predicted"].astype(float)
        fig.add_trace(
            go.Scatter(
                x=actual,
                y=predicted,
                mode="markers",
                name="Predictions",
                marker={"color": "#6366F1", "size": 7, "opacity": 0.7},
            )
        )
        lo = float(min(actual.min(), predicted.min()))
        hi = float(max(actual.max(), predicted.max()))
        fig.add_trace(
            go.Scatter(
                x=[lo, hi],
                y=[lo, hi],
                mode="lines",
                name="Ideal (y = x)",
                line={"color": "#EF4444", "dash": "dash"},
            )
        )
    fig.update_layout(
        title="Predicted vs Actual RUL",
        xaxis_title="Actual RUL",
        yaxis_title="Predicted RUL",
    )
    return _style(fig)


def error_histogram(errors: np.ndarray, model_name: str) -> go.Figure:
    """Build a histogram of prediction errors with a normal overlay.

    Args:
        errors: Array of prediction errors (predicted minus actual).
        model_name: Name shown in the chart title.

    Returns:
        A Plotly figure with a fitted normal-density curve.
    """
    errors = np.asarray(errors, dtype=float).ravel()
    fig = go.Figure()
    if errors.size:
        fig.add_trace(
            go.Histogram(
                x=errors,
                histnorm="probability density",
                name="Errors",
                marker={
                    "color": errors,
                    "colorscale": _GRADIENT_SCALE,
                    "showscale": False,
                },
                opacity=0.85,
            )
        )
        mu = float(errors.mean())
        sigma = float(errors.std()) or 1.0
        grid = np.linspace(errors.min(), errors.max(), 200)
        density = np.exp(-0.5 * ((grid - mu) / sigma) ** 2) / (
            sigma * np.sqrt(2.0 * np.pi)
        )
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=density,
                mode="lines",
                name="Normal fit",
                line={"color": "#EF4444", "width": 2.5},
            )
        )
    fig.update_layout(
        title=f"Error Distribution — {model_name}",
        xaxis_title="Prediction Error",
        yaxis_title="Density",
    )
    return _style(fig)
