"""Design tokens, CSS and chart styling for the dashboard (no Streamlit import, so it can be tested).

One place defines the colours, so every card, badge and chart agrees:
    navy/slate = navigation   blue = primary analytics   green = healthy   amber = warning
    red = critical            purple = forecasting / machine learning
Status colours always travel with a text label (never colour alone).
"""
from __future__ import annotations

import altair as alt

# ------------------------------------------------------------------ tokens
NAVY, SLATE = "#0F172A", "#1E293B"
BG, CARD, BORDER = "#F5F7FA", "#FFFFFF", "#E2E8F0"
TEXT, MUTED = "#0F172A", "#64748B"
BLUE, GREEN, AMBER, RED, PURPLE, GREY = "#2563EB", "#16A34A", "#D97706", "#DC2626", "#7C3AED", "#94A3B8"
# darker variants used for TEXT on white (they meet 4.5:1 contrast; the plain colours are for marks and borders)
TEXT_TONES = {"blue": "#1D4ED8", "green": "#15803D", "amber": "#B45309", "red": "#B91C1C", "purple": "#6D28D9", "grey": "#475569"}
TONE_COLORS = {"blue": BLUE, "green": GREEN, "amber": AMBER, "red": RED, "purple": PURPLE, "grey": GREY}

RISK_COLORS = {"Low": GREEN, "Medium": "#EAB308", "High": AMBER, "Critical": RED, "Unknown": GREY}
HEALTH_COLORS = {"Healthy": GREEN, "Low": AMBER, "Critical": RED, "Excess": BLUE, "Not assessed": GREY}
STATUS_COLORS = {"Low stock": AMBER, "Adequate stock": GREEN, "Excess stock": BLUE, "Insufficient history": GREY}
RISK_TONE = {"Low": "green", "Medium": "amber", "High": "amber", "Critical": "red", "Unknown": "grey"}


def color_scale(mapping: dict[str, str]) -> alt.Scale:
    return alt.Scale(domain=list(mapping), range=list(mapping.values()))


# ------------------------------------------------------------------ CSS (only our own classes + a few stable data-testid hooks)
CSS = f"""
:root {{ --navy:{NAVY}; --slate:{SLATE}; --bg:{BG}; --card:{CARD}; --border:{BORDER}; --text:{TEXT}; --muted:{MUTED}; }}
.stApp {{ background:{BG}; }}
.block-container {{ padding-top:4rem; padding-bottom:3rem; max-width:1440px; }}
#MainMenu, footer, .stDeployButton, [data-testid="stAppDeployButton"] {{ visibility:hidden; display:none; }}
h1, h2, h3, h4 {{ color:{TEXT}; letter-spacing:-0.01em; }}
h2 {{ font-size:1.2rem !important; margin-top:1.4rem; }}
h3 {{ font-size:1.02rem !important; }}

/* sidebar */
[data-testid="stSidebar"] {{ background:{NAVY}; border-right:1px solid {SLATE}; }}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stRadio"] label p,
[data-testid="stSidebar"] summary p {{ color:#CBD5E1 !important; }}
[data-testid="stSidebar"] [data-testid="stExpander"] {{ border:1px solid {SLATE}; border-radius:10px; background:#111C33; }}
[data-testid="stSidebar"] .si-brand {{ color:#FFFFFF; font-weight:700; font-size:1.05rem; letter-spacing:-0.01em; line-height:1.25; }}
[data-testid="stSidebar"] .si-brand small {{ display:block; color:#94A3B8; font-weight:500; font-size:.72rem; letter-spacing:.04em; text-transform:uppercase; margin-top:2px; }}
[data-testid="stSidebar"] .si-status {{ display:flex; gap:8px; flex-wrap:wrap; margin:.25rem 0 .5rem; }}
[data-testid="stSidebar"] .si-note {{ color:#94A3B8; font-size:.74rem; line-height:1.35; border-top:1px solid {SLATE}; padding-top:.6rem; margin-top:.6rem; }}

/* navigation: hide the radio dot, highlight the active page (only applies where the browser supports :has) */
[data-testid="stSidebar"] [data-testid="stRadio"] label {{ padding:7px 10px; border-radius:8px; margin-bottom:2px; width:100%; }}
[data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child {{ display:none; }}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{ background:#1E293B; }}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {{ background:{BLUE}; }}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p {{ color:#FFFFFF !important; font-weight:600; }}

[data-testid="stSidebar"] button [data-testid="stMarkdownContainer"] p {{ color:{TEXT} !important; }}

/* page header */
.si-header {{ background:{CARD}; border:1px solid {BORDER}; border-radius:14px; padding:18px 22px; margin-bottom:14px;
  box-shadow:0 1px 2px rgba(15,23,42,.04); }}
.si-header h1 {{ margin:0; padding:0; font-size:1.45rem; line-height:1.25; }}
.si-header p {{ margin:.35rem 0 0; color:{MUTED}; font-size:.92rem; }}
.si-chips {{ display:flex; flex-wrap:wrap; gap:6px; margin-top:.7rem; }}
.si-chip {{ background:#EEF2F7; color:#334155; border-radius:999px; padding:3px 10px; font-size:.76rem; font-weight:500; }}

/* KPI cards: CSS grid so they wrap on small screens */
.si-kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(168px,1fr)); gap:12px; margin:.4rem 0 1rem; }}
.si-kpi {{ background:{CARD}; border:1px solid {BORDER}; border-radius:12px; padding:13px 15px; border-top:3px solid {BLUE};
  box-shadow:0 1px 2px rgba(15,23,42,.04); min-width:0; }}
.si-kpi.blue {{ border-top-color:{BLUE}; }} .si-kpi.green {{ border-top-color:{GREEN}; }} .si-kpi.amber {{ border-top-color:{AMBER}; }}
.si-kpi.red {{ border-top-color:{RED}; }} .si-kpi.purple {{ border-top-color:{PURPLE}; }} .si-kpi.grey {{ border-top-color:{GREY}; }}
.si-kpi .l {{ color:{MUTED}; font-size:.76rem; font-weight:600; letter-spacing:.03em; text-transform:uppercase; }}
.si-kpi .v {{ color:{TEXT}; font-size:1.55rem; font-weight:700; line-height:1.2; margin-top:3px; overflow-wrap:anywhere; }}
.si-kpi .n {{ color:{MUTED}; font-size:.76rem; margin-top:3px; line-height:1.3; }}

/* callouts and badges */
.si-callout {{ border-radius:10px; padding:10px 14px; margin:.5rem 0 .9rem; font-size:.88rem; line-height:1.45; border:1px solid {BORDER}; border-left-width:4px; background:{CARD}; }}
.si-callout.info {{ border-left-color:{BLUE}; }} .si-callout.warn {{ border-left-color:{AMBER}; background:#FFFBEB; }}
.si-callout.danger {{ border-left-color:{RED}; background:#FEF2F2; }} .si-callout.ml {{ border-left-color:{PURPLE}; background:#F5F3FF; }}
.si-callout.ok {{ border-left-color:{GREEN}; background:#F0FDF4; }}
.si-callout b {{ color:{TEXT}; }}
.si-badge {{ display:inline-block; border-radius:6px; padding:2px 8px; font-size:.74rem; font-weight:600; }}
.si-badge.green {{ background:#DCFCE7; color:{TEXT_TONES['green']}; }} .si-badge.amber {{ background:#FEF3C7; color:{TEXT_TONES['amber']}; }}
.si-badge.red {{ background:#FEE2E2; color:{TEXT_TONES['red']}; }} .si-badge.blue {{ background:#DBEAFE; color:{TEXT_TONES['blue']}; }}
.si-badge.purple {{ background:#EDE9FE; color:{TEXT_TONES['purple']}; }} .si-badge.grey {{ background:#E2E8F0; color:{TEXT_TONES['grey']}; }}
[data-testid="stSidebar"] .si-badge.green {{ background:#14532D; color:#BBF7D0; }} [data-testid="stSidebar"] .si-badge.amber {{ background:#78350F; color:#FDE68A; }}
[data-testid="stSidebar"] .si-badge.red {{ background:#7F1D1D; color:#FECACA; }} [data-testid="stSidebar"] .si-badge.purple {{ background:#4C1D95; color:#DDD6FE; }}

/* insight cards */
.si-insights {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:12px; margin:.3rem 0 1rem; }}
.si-insight {{ background:{CARD}; border:1px solid {BORDER}; border-radius:12px; padding:12px 15px; border-left:4px solid {BLUE}; }}
.si-insight.green {{ border-left-color:{GREEN}; }} .si-insight.amber {{ border-left-color:{AMBER}; }} .si-insight.red {{ border-left-color:{RED}; }}
.si-insight.purple {{ border-left-color:{PURPLE}; }} .si-insight.grey {{ border-left-color:{GREY}; }}
.si-insight .t {{ font-weight:700; font-size:.88rem; color:{TEXT}; }} .si-insight .d {{ color:#334155; font-size:.84rem; margin-top:4px; line-height:1.45; }}

/* bordered containers and tabs */
[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius:12px; }}
button[data-baseweb="tab"] {{ font-weight:600; }}
@media (max-width: 640px) {{ .block-container {{ padding-left:.8rem; padding-right:.8rem; }} .si-kpi .v {{ font-size:1.3rem; }} }}
"""


# ------------------------------------------------------------------ Altair styling
def style_chart(chart, height: int | None = None):
    """Consistent fonts, quiet grid, no frame. Works for layered and single charts."""
    if height is not None:
        chart = chart.properties(height=height)
    return (chart.configure_view(strokeWidth=0)
            .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor="#EEF2F7", domainColor=BORDER, tickColor=BORDER,
                            labelFontSize=11, titleFontSize=11, titleFontWeight="normal")
            .configure_legend(labelColor=MUTED, titleColor=MUTED, labelFontSize=11, titleFontSize=11, orient="top")
            .configure_title(color=TEXT, fontSize=13, anchor="start"))
