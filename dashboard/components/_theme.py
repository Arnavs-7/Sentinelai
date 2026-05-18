"""Shared dark-theme global CSS and HTML helpers (Enterprise Dark).

A fully custom dark UI — near-black backgrounds, purple-tinted borders,
indigo-violet accent, layered glows and gradient text — built to feel
like a Palantir / Datadog / Grafana-grade monitoring platform.
"""

# Palette — enterprise dark design system with indigo-violet accent.
BG = "#0A0A0F"
SURFACE = "#111118"
ELEVATED = "#1A1A2E"
PRIMARY = "#6366F1"
PRIMARY_DARK = "#4F46E5"
VIOLET = "#8B5CF6"
CYAN = "#06B6D4"
HEALTHY = "#10B981"
WARNING = "#F59E0B"
CRITICAL = "#EF4444"
TEXT = "#F1F5F9"
MUTED = "#94A3B8"
DIM = "#475569"
BORDER = "#2A2A4A"
GRID = "#1E1E2E"
GLOW = "rgba(99,102,241,0.15)"

# Gradient end-colors so a card's accent can render as flowing gradient text.
_GRADIENTS = {
    PRIMARY: VIOLET,
    HEALTHY: "#34D399",
    WARNING: "#FBBF24",
    CRITICAL: "#F87171",
}

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"], [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif !important;
}
[data-testid="stAppViewContainer"] {
    background-color: #0A0A0F;
    background-image:
        radial-gradient(ellipse at 12% 0%, rgba(99,102,241,0.10) 0%,
            transparent 55%),
        radial-gradient(ellipse at 88% 100%, rgba(139,92,246,0.08) 0%,
            transparent 55%),
        radial-gradient(circle, rgba(99,102,241,0.07) 1px, transparent 1px);
    background-size: auto, auto, 30px 30px;
    background-attachment: fixed;
    color: #F1F5F9;
}
[data-testid="stHeader"] { background: #0A0A0F !important; }
.stApp { background: #0A0A0F; }

/* Smooth page entry animation. */
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.main .block-container { animation: fadeSlideUp 0.4s ease both; }

/* Animated gradient shimmer used on the hero border. */
@keyframes shimmer {
    0%   { background-position: 0% 50%; }
    100% { background-position: 200% 50%; }
}
@keyframes pulseDot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%      { opacity: 0.45; transform: scale(0.82); }
}

/* ── Sidebar ── */
/* Deep near-black panel with a faint indigo glow behind the wordmark and
   a purple-tinted right border, so it reads as a distinct dark surface. */
[data-testid="stSidebar"] {
    background-color: #0D0D1A !important;
    background-image:
        radial-gradient(ellipse 120% 32% at 50% 0%,
            rgba(99,102,241,0.18) 0%, transparent 72%) !important;
    border-right: 1px solid #2A2A4A !important;
    box-shadow: 1px 0 30px rgba(0,0,0,0.5) !important;
}
/* Streamlit's auto-generated page nav is hidden — a custom branded nav
   (see dashboard/components/sidebar.py) is rendered in its place. */
[data-testid="stSidebarNav"] { display: none !important; }
a[data-testid="stPageLink-NavLink"] {
    border-radius: 9px !important;
    padding: 0.5rem 0.75rem !important;
    transition: all 0.15s ease !important;
    color: #94A3B8 !important;
    border-left: 3px solid transparent !important;
}
a[data-testid="stPageLink-NavLink"]:hover {
    background: #1A1A2E !important;
    color: #F1F5F9 !important;
}
/* Active navigation item: indigo fill + left border indicator. */
a[data-testid="stPageLink-NavLink"][aria-current="page"] {
    background: rgba(99,102,241,0.18) !important;
    border-left: 3px solid #6366F1 !important;
    color: #F1F5F9 !important;
    font-weight: 600 !important;
}
a[data-testid="stPageLink-NavLink"] p { color: inherit !important; }
a[data-testid="stPageLink-NavLink"][aria-current="page"] p {
    color: #F1F5F9 !important;
    font-weight: 600 !important;
}

/* ── Typography ── */
h1 {
    color: #F1F5F9 !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.7px !important;
}
h2 {
    color: #F1F5F9 !important;
    font-weight: 700 !important;
    letter-spacing: -0.3px !important;
}
h3 {
    color: #F1F5F9 !important;
    font-weight: 700 !important;
    letter-spacing: -0.2px !important;
}
p, li { color: #94A3B8 !important; }
strong, b { color: #F1F5F9 !important; }

/* ── Metric cards ── */
.metric-card, [data-testid="stMetric"] {
    background: #1A1A2E !important;
}
[data-testid="stMetric"], [data-testid="metric-container"] {
    background: #1A1A2E !important;
    border: 1px solid #2A2A4A !important;
    border-top: 4px solid #6366F1 !important;
    border-radius: 12px !important;
    padding: 1.2rem 1.3rem !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4) !important;
    transition: box-shadow 0.22s ease, transform 0.22s ease,
        border-color 0.22s ease !important;
}
[data-testid="stMetric"]:hover, [data-testid="metric-container"]:hover,
.metric-card:hover {
    border-color: #6366F1 !important;
    box-shadow: 0 8px 36px rgba(99,102,241,0.28) !important;
    transform: translateY(-4px) !important;
}
.metric-card {
    transition: box-shadow 0.22s ease, transform 0.22s ease,
        border-color 0.22s ease;
}
[data-testid="stMetricLabel"] {
    color: #94A3B8 !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
}
[data-testid="stMetricLabel"] p { color: #94A3B8 !important; }
[data-testid="stMetricValue"] {
    font-weight: 800 !important;
    font-size: 2rem !important;
    font-variant-numeric: tabular-nums !important;
    background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
}
[data-testid="stMetricDelta"] { color: #94A3B8 !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%) !important;
    border: none !important;
    color: #FFFFFF !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.2px !important;
    transition: all 0.15s ease !important;
    padding: 0.55rem 1.5rem !important;
    box-shadow: 0 2px 12px rgba(99,102,241,0.35) !important;
}
.stButton > button:hover {
    filter: brightness(1.12) !important;
    color: #FFFFFF !important;
    box-shadow: 0 6px 22px rgba(99,102,241,0.5) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }
.stDownloadButton > button {
    background: #1A1A2E !important;
    border: 1px solid #2A2A4A !important;
    color: #A5B4FC !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}
.stDownloadButton > button:hover {
    border-color: #6366F1 !important;
    background: #1F1F3A !important;
    color: #C7D2FE !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #2A2A4A !important;
    gap: 0.4rem !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #94A3B8 !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 0.6rem 1.1rem !important;
    font-weight: 600 !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(99,102,241,0.12) !important;
    color: #C7D2FE !important;
    border-bottom: 2px solid #6366F1 !important;
}

/* ── Inputs ── */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] > div > div,
[data-testid="stDateInput"] input,
[data-baseweb="select"] > div,
textarea {
    background: #111118 !important;
    border: 1px solid #2A2A4A !important;
    color: #F1F5F9 !important;
    border-radius: 10px !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus,
textarea:focus {
    border-color: #6366F1 !important;
    box-shadow: 0 0 0 4px rgba(99,102,241,0.2) !important;
}
[data-baseweb="popover"] [role="listbox"] {
    background: #1A1A2E !important;
    border: 1px solid #2A2A4A !important;
}
[data-baseweb="popover"] [role="option"]:hover {
    background: rgba(99,102,241,0.15) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #111118 !important;
    border: 2px dashed #2A2A4A !important;
    border-radius: 14px !important;
    padding: 1.5rem !important;
    transition: all 0.2s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #6366F1 !important;
    background: #15152A !important;
}

/* ── Alert boxes ── */
[data-testid="stInfo"] {
    background: rgba(99,102,241,0.12) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    border-left: 4px solid #6366F1 !important;
    border-radius: 0 10px 10px 0 !important;
    color: #C7D2FE !important;
}
[data-testid="stSuccess"] {
    background: rgba(16,185,129,0.12) !important;
    border: 1px solid rgba(16,185,129,0.3) !important;
    border-left: 4px solid #10B981 !important;
    border-radius: 0 10px 10px 0 !important;
    color: #6EE7B7 !important;
}
[data-testid="stWarning"] {
    background: rgba(245,158,11,0.12) !important;
    border: 1px solid rgba(245,158,11,0.3) !important;
    border-left: 4px solid #F59E0B !important;
    border-radius: 0 10px 10px 0 !important;
    color: #FCD34D !important;
}
[data-testid="stError"] {
    background: rgba(239,68,68,0.12) !important;
    border: 1px solid rgba(239,68,68,0.3) !important;
    border-left: 4px solid #EF4444 !important;
    border-radius: 0 10px 10px 0 !important;
    color: #FCA5A5 !important;
}
[data-testid="stInfo"] p, [data-testid="stSuccess"] p,
[data-testid="stWarning"] p, [data-testid="stError"] p {
    color: inherit !important;
}

/* ── Dataframe / tables ── */
[data-testid="stDataFrame"], [data-testid="stTable"] {
    border: 1px solid #2A2A4A !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4) !important;
}
[data-testid="stTable"] table { background: #111118 !important; }
[data-testid="stTable"] thead th {
    background: #0D0D1A !important;
    color: #94A3B8 !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}
[data-testid="stTable"] tbody tr:nth-child(even) { background: #15152A !important; }
[data-testid="stTable"] tbody tr:nth-child(odd) { background: #111118 !important; }
[data-testid="stTable"] tbody tr:hover { background: #1A1A2E !important; }
[data-testid="stTable"] td { color: #F1F5F9 !important; }

/* ── Plotly charts ── */
.js-plotly-plot {
    border: 1px solid #2A2A4A !important;
    border-radius: 16px !important;
    overflow: hidden !important;
    background: #111118 !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1px solid #2A2A4A !important;
    border-radius: 12px !important;
    background: #111118 !important;
}
[data-testid="stExpander"] summary { color: #F1F5F9 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-track { background: #0D0D1A; }
::-webkit-scrollbar-thumb { background: #2A2A4A; border-radius: 5px; }
::-webkit-scrollbar-thumb:hover { background: #6366F1; }

/* ── Text selection ── */
::selection { background: rgba(99,102,241,0.35); color: #F1F5F9; }

/* ── Status badges ── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 13px;
    border-radius: 999px;
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.4px;
    text-transform: uppercase;
    border: 1px solid transparent;
}
.badge .dot { font-size: 0.6rem; line-height: 1; }
.badge-healthy {
    background: rgba(16,185,129,0.13);
    color: #10B981;
    border-color: rgba(16,185,129,0.4);
}
.badge-warning {
    background: rgba(245,158,11,0.13);
    color: #F59E0B;
    border-color: rgba(245,158,11,0.4);
}
.badge-critical {
    background: rgba(239,68,68,0.13);
    color: #EF4444;
    border-color: rgba(239,68,68,0.4);
}
.badge-critical .dot { animation: pulseDot 1.4s ease-in-out infinite; }

/* ── Divider ── */
hr { border-color: #2A2A4A !important; margin: 1.4rem 0 !important; }

/* ── Main content padding ── */
.main .block-container {
    padding: 2rem 2.5rem !important;
    max-width: 1300px !important;
}

/* ── Captions ── */
[data-testid="stCaptionContainer"], .stCaption { color: #475569 !important; }
[data-testid="stCaptionContainer"] p { color: #475569 !important; }
</style>
"""


def status_badge(status: str) -> str:
    """Build a color-coded HTML status badge with a leading dot.

    Args:
        status: One of ``healthy``, ``warning`` or ``critical`` (case
            insensitive); anything that is not healthy or warning is
            rendered as critical.

    Returns:
        An HTML ``span`` string for ``st.markdown(..., unsafe_allow_html=True)``.
    """
    s = status.lower()
    cls = (
        "badge-healthy"
        if s == "healthy"
        else "badge-warning"
        if s == "warning"
        else "badge-critical"
    )
    return (
        f'<span class="badge {cls}"><span class="dot">●</span>'
        f"{status.upper()}</span>"
    )


def metric_card(
    title: str, value: str, subtitle: str = "", color: str = PRIMARY
) -> str:
    """Build a single metric card with a colored top border.

    Args:
        title: The metric label shown above the value.
        value: The primary metric value.
        subtitle: Optional muted supporting line beneath the value.
        color: CSS color applied to the top border, dot and value gradient.

    Returns:
        An HTML string for ``st.markdown(..., unsafe_allow_html=True)``.
    """
    end = _GRADIENTS.get(color, color)
    return f"""
<div class="metric-card" style="background:#1A1A2E;
    border:1px solid #2A2A4A; border-top:4px solid {color};
    border-radius:12px; padding:1.3rem 1.4rem;
    box-shadow:0 4px 24px rgba(0,0,0,0.4);">
    <div style="display:flex;align-items:center;gap:7px;margin-bottom:0.5rem;">
        <span style="width:8px;height:8px;border-radius:50%;
            background:{color};display:inline-block;
            box-shadow:0 0 8px {color};"></span>
        <span style="color:#94A3B8;font-size:0.75rem;font-weight:600;
            text-transform:uppercase;letter-spacing:0.6px;">{title}</span>
    </div>
    <div style="font-size:2.1rem;font-weight:800;line-height:1;
        font-variant-numeric:tabular-nums;
        background:linear-gradient(135deg,{color} 0%,{end} 100%);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        background-clip:text;">
        {value}
    </div>
    <div style="color:#475569;font-size:0.8rem;margin-top:0.45rem;">{subtitle}</div>
</div>"""


def empty_state(
    icon: str = "✈️",
    message: str = "No data available",
    sub: str = "Data will appear once engines are registered",
) -> str:
    """Build a styled empty-state card.

    Args:
        icon: A leading emoji or glyph.
        message: The primary message to display.
        sub: A muted supporting line beneath the message.

    Returns:
        An HTML string for ``st.markdown(..., unsafe_allow_html=True)``.
    """
    return f"""
<div style="background:#1A1A2E;border:1px solid #2A2A4A;border-radius:16px;
    padding:3rem;text-align:center;margin:2rem 0;
    box-shadow:0 4px 24px rgba(0,0,0,0.4);">
    <div style="font-size:3rem;margin-bottom:1rem;">{icon}</div>
    <div style="color:#F1F5F9;font-size:1.1rem;font-weight:700;margin-bottom:0.5rem;">
        {message}
    </div>
    <div style="color:#475569;font-size:0.85rem;">{sub}</div>
</div>"""


def page_header(title: str, subtitle: str = "") -> str:
    """Build a clean page header with a gradient title and accent bar.

    Args:
        title: The page title text.
        subtitle: Optional supporting line shown beneath the title.

    Returns:
        An HTML string for ``st.markdown(..., unsafe_allow_html=True)``.
    """
    sub = (
        "<p style='color:#94A3B8;margin:0.35rem 0 0;font-size:0.95rem;'>"
        + subtitle
        + "</p>"
        if subtitle
        else ""
    )
    return f"""
<div style="margin-bottom:1.7rem;">
    <h1 style="font-size:2.1rem;font-weight:800;letter-spacing:-0.7px;
        margin:0;border:none;padding:0;
        background:linear-gradient(135deg,#818CF8 0%,#A78BFA 100%);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        background-clip:text;display:inline-block;">
        {title}
    </h1>
    {sub}
    <div style="height:3px;width:48px;border-radius:99px;margin-top:0.85rem;
        background:linear-gradient(135deg,#6366F1 0%,#8B5CF6 100%);
        box-shadow:0 0 12px rgba(99,102,241,0.5);"></div>
</div>"""
