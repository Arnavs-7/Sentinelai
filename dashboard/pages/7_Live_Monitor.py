"""Live Monitor page: simulated real-time sensor feed with alerting.

Streams a bounded simulation of the fleet's sensor telemetry into a
single ``st.empty()`` placeholder. When any engine's remaining useful
life falls below the critical threshold a red banner is raised, and an
optional Slack notification is sent when ``SLACK_WEBHOOK_URL`` is set.
"""

import os
import random
import time
from typing import Any, Dict, List

import pandas as pd
import requests
import streamlit as st

from dashboard.components._theme import GLOBAL_CSS, empty_state, page_header
from dashboard.components.sidebar import render_sidebar
from dashboard.utils.api_client import SentinelAPIClient

_CRITICAL_RUL = 30.0
_TICKS = 30
_TICK_SECONDS = 1.0


def _notify_slack(message: str) -> None:
    """Post an alert message to Slack when a webhook is configured.

    Args:
        message: The alert text to send.
    """
    webhook = os.getenv("SLACK_WEBHOOK_URL", "").strip()
    if not webhook:
        return
    try:
        requests.post(webhook, json={"text": message}, timeout=5.0)
    except requests.RequestException:
        # Alerting must never break the live view; failures are ignored.
        pass


def _risk_icon(rul: float) -> str:
    """Return a traffic-light icon for a remaining-useful-life value.

    Args:
        rul: Remaining useful life in cycles.

    Returns:
        A colored circle emoji keyed to the risk band.
    """
    if rul < _CRITICAL_RUL:
        return "🔴"
    if rul < 100.0:
        return "🟡"
    return "🟢"


def _seed_fleet(machines: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    """Build the initial simulation state from the live fleet.

    Args:
        machines: Machine dicts returned by the API.

    Returns:
        A list of ``{name, rul}`` simulation records.
    """
    fleet: List[Dict[str, float]] = []
    for machine in machines:
        prediction = machine.get("latest_prediction") or {}
        fleet.append(
            {
                "name": machine.get("name", machine.get("id", "?")),
                "rul": float(prediction.get("predicted_rul", 120.0)),
            }
        )
    return fleet


def _step(fleet: List[Dict[str, float]]) -> pd.DataFrame:
    """Advance the simulation one tick and return a display frame.

    Each engine degrades by a small noisy amount and emits a jittered
    sensor reading, mimicking a live telemetry stream.

    Args:
        fleet: The mutable simulation state, updated in place.

    Returns:
        A frame with one row per engine for display.
    """
    rows: List[Dict[str, Any]] = []
    for engine in fleet:
        engine["rul"] = max(0.0, engine["rul"] - random.uniform(0.4, 2.6))
        rul = engine["rul"]
        rows.append(
            {
                "Engine": engine["name"],
                "RUL (cycles)": round(rul, 1),
                "Sensor T48 (°R)": round(random.gauss(1400.0, 6.0), 1),
                "Vibration (mm/s)": round(random.gauss(2.0, 0.4), 2),
                "Status": f"{_risk_icon(rul)} "
                + ("CRITICAL" if rul < _CRITICAL_RUL else "Nominal"),
            }
        )
    return pd.DataFrame(rows)


def render() -> None:
    """Render the full Live Monitor page."""
    st.set_page_config(page_title="Live Monitor", page_icon="📡", layout="wide")
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    render_sidebar()
    st.markdown(
        page_header(
            "Live Monitor",
            "Simulated real-time sensor feed with critical-RUL alerting",
        ),
        unsafe_allow_html=True,
    )

    client = SentinelAPIClient()
    machines = client.get_machines()
    if not machines:
        st.markdown(
            empty_state(
                "📡",
                "No fleet data available",
                "Start the SentinelAI API to stream the live feed.",
            ),
            unsafe_allow_html=True,
        )
        return

    st.caption(
        f"Streams {_TICKS} ticks (~{int(_TICKS * _TICK_SECONDS)}s). A red "
        f"banner is raised when any engine RUL drops below {int(_CRITICAL_RUL)} "
        "cycles."
    )
    if not st.button("▶ Start live monitor"):
        st.info("Press **Start live monitor** to begin the simulated feed.")
        return

    fleet = _seed_fleet(machines)
    banner = st.empty()
    feed = st.empty()
    notified: set[str] = set()

    for tick in range(1, _TICKS + 1):
        frame = _step(fleet)
        critical = frame[frame["RUL (cycles)"] < _CRITICAL_RUL]

        if not critical.empty:
            names = ", ".join(critical["Engine"].astype(str))
            banner.error(
                f"🚨 CRITICAL ALERT — RUL below {int(_CRITICAL_RUL)} cycles: "
                f"{names}. Immediate maintenance required."
            )
            for name in critical["Engine"].astype(str):
                if name not in notified:
                    _notify_slack(
                        f"SentinelAI: engine {name} entered CRITICAL state "
                        "(RUL < 30 cycles)."
                    )
                    notified.add(name)
        else:
            banner.success("✅ All engines nominal — no critical alerts.")

        with feed.container():
            st.markdown(f"**Tick {tick}/{_TICKS}** · live telemetry")
            st.dataframe(frame, use_container_width=True, hide_index=True)
        time.sleep(_TICK_SECONDS)

    st.toast("Live monitor simulation complete.", icon="✅")


render()
