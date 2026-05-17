"""Shared light-theme global CSS and HTML helpers (Premium Light).

A high-impact light UI: indigo-violet accent, layered shadows, gradient
text and smooth motion — built to feel like a modern SaaS product.
"""

# Palette — premium light SaaS aesthetic with indigo-violet accent.
BG = "#F0EEF8"
SURFACE = "#FFFFFF"
PRIMARY = "#6366F1"
PRIMARY_DARK = "#4F46E5"
VIOLET = "#8B5CF6"
HEALTHY = "#10B981"
WARNING = "#F59E0B"
CRITICAL = "#EF4444"
TEXT = "#111827"
MUTED = "#6B7280"
BORDER = "#E5E7EB"
GRID = "#F3F4F6"
TINT = "#F5F3FF"

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
    background-color: #F0EEF8;
    background-image:
        radial-gradient(ellipse at 10% 0%, rgba(99,102,241,0.08) 0%,
            transparent 50%),
        radial-gradient(ellipse at 90% 100%, rgba(139,92,246,0.06) 0%,
            transparent 50%),
        radial-gradient(circle, rgba(99,102,241,0.12) 1px, transparent 1px);
    background-size: auto, auto, 28px 28px;
    background-attachment: fixed;
    color: #111827;
}
[data-testid="stHeader"] { background: transparent !important; }

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
/* Frosted-glass panel: a translucent white that lets the tinted page
   show through softly, a faint indigo glow at the top behind the brand
   wordmark, and a gentle vertical fade into the page background — so the
   sidebar feels like part of the premium surface, not a flat white slab. */
[data-testid="stSidebar"] {
    background-color: rgba(255,255,255,0.62) !important;
    background-image:
        radial-gradient(ellipse 120% 38% at 50% 0%,
            rgba(99,102,241,0.14) 0%, transparent 70%),
        linear-gradient(180deg,
            rgba(255,255,255,0.92) 0%,
            rgba(248,246,253,0.80) 52%,
            rgba(238,236,250,0.74) 100%) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border-right: 1px solid rgba(99,102,241,0.15) !important;
    box-shadow: 1px 0 28px rgba(99,102,241,0.07) !important;
}
/* Streamlit's auto-generated page nav is hidden — a custom branded nav
   (see dashboard/components/sidebar.py) is rendered in its place. */
[data-testid="stSidebarNav"] { display: none !important; }
a[data-testid="stPageLink-NavLink"] {
    border-radius: 9px !important;
    padding: 0.5rem 0.75rem !important;
    transition: all 0.15s ease !important;
    color: #6B7280 !important;
    border-left: 3px solid transparent !important;
}
a[data-testid="stPageLink-NavLink"]:hover {
    background: #F5F3FF !important;
    color: #111827 !important;
}
/* Active navigation item: indigo tint + left border indicator. */
a[data-testid="stPageLink-NavLink"][aria-current="page"] {
    background: #EEF2FF !important;
    border-left: 3px solid #6366F1 !important;
    color: #4F46E5 !important;
    font-weight: 600 !important;
}
a[data-testid="stPageLink-NavLink"] p { color: inherit !important; }
a[data-testid="stPageLink-NavLink"][aria-current="page"] p {
    color: #4F46E5 !important;
    font-weight: 600 !important;
}

/* ── Typography ── */
h1 {
    color: #111827 !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.7px !important;
}
h2 {
    color: #111827 !important;
    font-weight: 700 !important;
    letter-spacing: -0.3px !important;
}
h3 {
    color: #111827 !important;
    font-weight: 700 !important;
    letter-spacing: -0.2px !important;
}
p, li { color: #6B7280 !important; }

/* ── Metric cards ── */
/* Frosted-glass surface: a translucent white over the tinted page
   background gives cards depth without looking flat. */
.metric-card, [data-testid="stMetric"] {
    background: rgba(255,255,255,0.85) !important;
    backdrop-filter: blur(8px) !important;
    -webkit-backdrop-filter: blur(8px) !important;
}
[data-testid="stMetric"], [data-testid="metric-container"] {
    background: rgba(255,255,255,0.85) !important;
    border: 1px solid #E5E7EB !important;
    border-top: 4px solid #6366F1 !important;
    border-radius: 16px !important;
    padding: 1.2rem 1.3rem !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.06) !important;
    transition: box-shadow 0.22s ease, transform 0.22s ease !important;
}
[data-testid="stMetric"]:hover, [data-testid="metric-container"]:hover,
.metric-card:hover {
    box-shadow: 0 12px 40px rgba(99,102,241,0.15) !important;
    transform: translateY(-4px) !important;
}
.metric-card { transition: box-shadow 0.22s ease, transform 0.22s ease; }
[data-testid="stMetricLabel"] {
    color: #6B7280 !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
}
[data-testid="stMetricLabel"] p { color: #6B7280 !important; }
[data-testid="stMetricValue"] {
    font-weight: 800 !important;
    font-size: 2rem !important;
    font-variant-numeric: tabular-nums !important;
    background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
}

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
    box-shadow: 0 2px 8px rgba(99,102,241,0.25) !important;
}
.stButton > button:hover {
    filter: brightness(1.1) !important;
    color: #FFFFFF !important;
    box-shadow: 0 6px 18px rgba(99,102,241,0.35) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }
.stDownloadButton > button {
    background: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    color: #4F46E5 !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}
.stDownloadButton > button:hover {
    border-color: #6366F1 !important;
    background: #F5F3FF !important;
    color: #4F46E5 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #E5E7EB !important;
    gap: 0.4rem !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #6B7280 !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 0.6rem 1.1rem !important;
    font-weight: 600 !important;
}
.stTabs [aria-selected="true"] {
    background: #F5F3FF !important;
    color: #4F46E5 !important;
    border-bottom: 2px solid #6366F1 !important;
}

/* ── Inputs ── */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] > div > div,
[data-testid="stDateInput"] input,
[data-baseweb="select"] > div,
textarea {
    background: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    color: #111827 !important;
    border-radius: 10px !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus,
textarea:focus {
    border-color: #6366F1 !important;
    box-shadow: 0 0 0 4px rgba(99,102,241,0.2) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #FFFFFF !important;
    border: 2px dashed #C7C9F2 !important;
    border-radius: 14px !important;
    padding: 1.5rem !important;
    transition: all 0.2s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #6366F1 !important;
    background: #F5F3FF !important;
}

/* ── Alert boxes ── */
[data-testid="stInfo"] {
    background: #EEF2FF !important;
    border: 1px solid #DDD6FE !important;
    border-left: 4px solid #6366F1 !important;
    border-radius: 0 10px 10px 0 !important;
    color: #3730A3 !important;
}
[data-testid="stSuccess"] {
    background: #ECFDF5 !important;
    border: 1px solid #A7F3D0 !important;
    border-left: 4px solid #10B981 !important;
    border-radius: 0 10px 10px 0 !important;
    color: #065F46 !important;
}
[data-testid="stWarning"] {
    background: #FFFBEB !important;
    border: 1px solid #FDE68A !important;
    border-left: 4px solid #F59E0B !important;
    border-radius: 0 10px 10px 0 !important;
    color: #92400E !important;
}
[data-testid="stError"] {
    background: #FEF2F2 !important;
    border: 1px solid #FECACA !important;
    border-left: 4px solid #EF4444 !important;
    border-radius: 0 10px 10px 0 !important;
    color: #991B1B !important;
}

/* ── Dataframe / tables ── */
[data-testid="stDataFrame"], [data-testid="stTable"] {
    border: 1px solid #E5E7EB !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
}
[data-testid="stTable"] table { background: #FFFFFF !important; }
[data-testid="stTable"] thead th {
    background: rgba(99,102,241,0.06) !important;
    color: #111827 !important;
    font-weight: 700 !important;
}
[data-testid="stTable"] tbody tr:nth-child(even) { background: #FAFAFA !important; }
[data-testid="stTable"] tbody tr:nth-child(odd) { background: #FFFFFF !important; }
[data-testid="stTable"] td { color: #111827 !important; }

/* ── Plotly charts ── */
.js-plotly-plot {
    border: 1px solid #E5E7EB !important;
    border-radius: 16px !important;
    overflow: hidden !important;
    background: #FFFFFF !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.06) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1px solid #E5E7EB !important;
    border-radius: 12px !important;
    background: #FFFFFF !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-track { background: #F3F4F6; }
::-webkit-scrollbar-thumb { background: #D1D5DB; border-radius: 5px; }
::-webkit-scrollbar-thumb:hover { background: #6366F1; }

/* ── Text selection ── */
::selection { background: #EDE9FE; color: #111827; }

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
}
.badge .dot { font-size: 0.6rem; line-height: 1; }
.badge-healthy {
    background: linear-gradient(135deg, #D1FAE5, #A7F3D0);
    color: #065F46;
}
.badge-warning {
    background: linear-gradient(135deg, #FEF3C7, #FDE68A);
    color: #92400E;
}
.badge-critical {
    background: linear-gradient(135deg, #FEE2E2, #FECACA);
    color: #991B1B;
}
.badge-critical .dot { animation: pulseDot 1.4s ease-in-out infinite; }

/* ── Divider ── */
hr { border-color: #E5E7EB !important; margin: 1.4rem 0 !important; }

/* ── Main content padding ── */
.main .block-container {
    padding: 2rem 2.5rem !important;
    max-width: 1300px !important;
}

/* ── Captions ── */
[data-testid="stCaptionContainer"], .stCaption { color: #9CA3AF !important; }
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
<div class="metric-card" style="background:#FFFFFF;
    border:1px solid #E5E7EB; border-top:4px solid {color};
    border-radius:16px; padding:1.3rem 1.4rem;
    box-shadow:0 1px 3px rgba(0,0,0,0.06),0 4px 16px rgba(0,0,0,0.06);">
    <div style="display:flex;align-items:center;gap:7px;margin-bottom:0.5rem;">
        <span style="width:8px;height:8px;border-radius:50%;
            background:{color};display:inline-block;"></span>
        <span style="color:#6B7280;font-size:0.75rem;font-weight:600;
            text-transform:uppercase;letter-spacing:0.6px;">{title}</span>
    </div>
    <div style="font-size:2.1rem;font-weight:800;line-height:1;
        font-variant-numeric:tabular-nums;
        background:linear-gradient(135deg,{color} 0%,{end} 100%);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        background-clip:text;">
        {value}
    </div>
    <div style="color:#9CA3AF;font-size:0.8rem;margin-top:0.45rem;">{subtitle}</div>
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
<div style="background:#FFFFFF;border:1px solid #E5E7EB;border-radius:16px;
    padding:3rem;text-align:center;margin:2rem 0;
    box-shadow:0 1px 3px rgba(0,0,0,0.06),0 4px 16px rgba(0,0,0,0.06);">
    <div style="font-size:3rem;margin-bottom:1rem;">{icon}</div>
    <div style="color:#111827;font-size:1.1rem;font-weight:700;margin-bottom:0.5rem;">
        {message}
    </div>
    <div style="color:#9CA3AF;font-size:0.85rem;">{sub}</div>
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
        "<p style='color:#6B7280;margin:0.35rem 0 0;font-size:0.95rem;'>"
        + subtitle
        + "</p>"
        if subtitle
        else ""
    )
    return f"""
<div style="margin-bottom:1.7rem;">
    <h1 style="font-size:2.1rem;font-weight:800;letter-spacing:-0.7px;
        margin:0;border:none;padding:0;
        background:linear-gradient(135deg,#6366F1 0%,#8B5CF6 100%);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        background-clip:text;display:inline-block;">
        {title}
    </h1>
    {sub}
    <div style="height:3px;width:48px;border-radius:99px;margin-top:0.85rem;
        background:linear-gradient(135deg,#6366F1 0%,#8B5CF6 100%);"></div>
</div>"""
