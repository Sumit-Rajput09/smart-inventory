"""Reusable Streamlit UI pieces: page header, KPI cards, callouts, badges, insight cards, searchable tables.

HTML builders (`*_html`) are pure functions so they can be unit-tested; every piece of dynamic text is HTML-escaped.
Markup is emitted on one line: Markdown turns indented or blank-line-separated HTML into code blocks.
"""
from __future__ import annotations

import html
from dataclasses import dataclass

import pandas as pd
import streamlit as st

from src import app_support as sup
from src.ui_theme import CSS

MAX_TABLE_ROWS = 2000
TONES = {"blue", "green", "amber", "red", "purple", "grey"}


def _e(x) -> str:
    return html.escape(str(x), quote=True)


def _tone(t: str | None, default: str = "blue") -> str:
    return t if t in TONES else default


# ------------------------------------------------------------------ pure HTML builders
@dataclass(frozen=True)
class Kpi:
    label: str
    value: str
    note: str = ""
    tone: str = "blue"


def kpi_cards_html(items: list[Kpi]) -> str:
    cards = "".join(f'<div class="si-kpi {_tone(k.tone)}"><div class="l">{_e(k.label)}</div><div class="v">{_e(k.value)}</div>'
                    + (f'<div class="n">{_e(k.note)}</div>' if k.note else "") + "</div>" for k in items)
    return f'<div class="si-kpis">{cards}</div>'


def badge_html(text: str, tone: str = "blue") -> str:
    return f'<span class="si-badge {_tone(tone)}">{_e(text)}</span>'


def callout_html(kind: str, text: str) -> str:
    """kind: info | warn | danger | ml | ok. `text` may contain **bold** (converted) but is otherwise escaped."""
    kind = kind if kind in {"info", "warn", "danger", "ml", "ok"} else "info"
    parts = _e(text).split("**")
    body = "".join(f"<b>{p}</b>" if i % 2 else p for i, p in enumerate(parts))
    return f'<div class="si-callout {kind}">{body}</div>'


def header_html(title: str, subtitle: str = "", chips: list[str] | None = None) -> str:
    chip_html = "".join(f'<span class="si-chip">{_e(c)}</span>' for c in (chips or []))
    return (f'<div class="si-header"><h1>{_e(title)}</h1>' + (f"<p>{_e(subtitle)}</p>" if subtitle else "")
            + (f'<div class="si-chips">{chip_html}</div>' if chip_html else "") + "</div>")


def insight_cards_html(insights: list[dict]) -> str:
    cards = "".join(f'<div class="si-insight {_tone(i.get("tone"), "blue")}"><div class="t">{_e(i["title"])}</div><div class="d">{_e(i["detail"])}</div></div>'
                    for i in insights)
    return f'<div class="si-insights">{cards}</div>'


# ------------------------------------------------------------------ Streamlit wrappers
def inject_theme() -> None:
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "", chips: list[str] | None = None) -> None:
    st.markdown(header_html(title, subtitle, chips), unsafe_allow_html=True)


def kpi_row(items: list[Kpi]) -> None:
    st.markdown(kpi_cards_html(items), unsafe_allow_html=True)


def callout(kind: str, text: str) -> None:
    st.markdown(callout_html(kind, text), unsafe_allow_html=True)


def insight_cards(insights: list[dict]) -> None:
    if insights:
        st.markdown(insight_cards_html(insights), unsafe_allow_html=True)


def section(title: str, caption: str | None = None) -> None:
    st.markdown(f"## {title}")
    if caption:
        st.caption(caption)


def show_table(df: pd.DataFrame, key: str, search_columns: list[str] | None = None, file_name: str | None = None,
               column_config: dict | None = None, height: int = 420, label: str = "Search table") -> pd.DataFrame:
    """Searchable table with row count and CSV download. Returns the searched (not truncated) table."""
    query = st.text_input(label, key=f"{key}_search", placeholder="Type words, e.g.  S003 toys  (all words must match)")
    found = sup.search_dataframe(df, query, search_columns)
    if found.empty:
        st.info("No rows match your search.")
        return found
    shown = found.head(MAX_TABLE_ROWS)
    st.caption(f"{len(found):,} of {len(df):,} rows" + (f" (showing the first {MAX_TABLE_ROWS:,}; download for all)" if len(found) > MAX_TABLE_ROWS else ""))
    st.dataframe(shown, use_container_width=True, hide_index=True, column_config=column_config, height=height)
    if file_name:
        st.download_button("Download this table (CSV)", sup.df_to_csv_bytes(found), file_name, "text/csv", key=f"{key}_download")
    return found
