"""Metrik kartları ve skor göstergeleri."""

from __future__ import annotations

from typing import Any, Sequence

import plotly.graph_objects as go
import streamlit as st

from utils.helpers import format_bytes, format_number, format_percent

__all__ = [
    "metric_row",
    "dataset_header",
    "overview_cards",
    "quality_badges",
    "score_gauge",
]


def metric_row(
    items: Sequence[tuple[str, Any] | tuple[str, Any, str]], columns: int | None = None
) -> None:
    """Metrik kartlarını yatay bir satırda gösterir.

    Args:
        items: ``(etiket, değer)`` veya ``(etiket, değer, yardım_metni)``
            üçlülerinden oluşan dizi.
        columns: Satır başına kart sayısı (None ise öğe sayısı kadar).
    """
    if not items:
        return
    per_row = columns or len(items)
    for start in range(0, len(items), per_row):
        chunk = items[start : start + per_row]
        for column, item in zip(st.columns(len(chunk)), chunk):
            label, value = item[0], item[1]
            help_text = item[2] if len(item) > 2 else None
            column.metric(label, value, help=help_text)


def dataset_header() -> None:
    """Her sayfanın üstünde gösterilen sade dataset özeti (4 kart).

    Satır/sütun sayısı, boş hücre oranı ve veri kalite skorunu gösterir.
    Hesaplamalar önbelleklidir; sayfa değişimlerinde yeniden hesaplanmaz.
    """
    from core import session  # döngüsel import olmaması için fonksiyon içinde
    from utils import caching

    df = session.get_working_df()
    if df is None:
        return

    info = caching.dataset_shape_info(df)
    try:
        score = caching.quality_score(df)
        score_text = f"{score.overall}/100"
        score_help = score.label
    except Exception:  # skor hesaplanamazsa başlık yine çizilsin
        score_text, score_help = "-", None

    name = str(session.dataset_meta().get("filename") or "veri seti")
    st.caption(f"📄 {name}")
    columns = st.columns(4)
    columns[0].metric("Satır", format_number(info["rows"]))
    columns[1].metric("Sütun", format_number(info["columns"]))
    columns[2].metric(
        "Boş hücre",
        format_percent(info["missing_ratio"]),
        help=f"{format_number(info['missing_cells'])} boş hücre",
    )
    columns[3].metric("Veri kalite skoru", score_text, help=score_help)
    st.divider()


def overview_cards(info: dict[str, Any]) -> None:
    """Dataset Overview metriklerini kartlar halinde gösterir.

    Args:
        info: ``dataframe_utils.dataset_shape_info`` çıktısı.
    """
    metric_row(
        [
            ("Satır", format_number(info.get("rows", 0))),
            ("Sütun", format_number(info.get("columns", 0))),
            ("Toplam Hücre", format_number(info.get("cells", 0))),
            (
                "Bellek Kullanımı",
                format_bytes(info.get("memory_bytes", 0)),
                "Pandas'ın derin (deep) bellek ölçümü.",
            ),
        ]
    )
    metric_row(
        [
            (
                "Boş Hücre",
                format_number(info.get("missing_cells", 0)),
                f"Tüm hücrelerin {format_percent(info.get('missing_ratio', 0))}'i boş.",
            ),
            (
                "Tekrarlı Satır",
                format_number(info.get("duplicate_rows", 0)),
                f"Satırların {format_percent(info.get('duplicate_ratio', 0))}'i tekrar.",
            ),
            ("Sayısal Sütun", format_number(info.get("numeric_cols", 0))),
            ("Kategorik/Metin", format_number(info.get("categorical_cols", 0))),
            ("Tarih Sütunu", format_number(info.get("datetime_cols", 0))),
            ("Mantıksal Sütun", format_number(info.get("boolean_cols", 0))),
        ],
        columns=6,
    )


def quality_badges(counts: dict[str, int]) -> None:
    """Insight sayılarını önem derecesine göre kartlarda gösterir."""
    metric_row(
        [
            ("🔴 Kritik", format_number(counts.get("critical", 0))),
            ("🟠 Uyarı", format_number(counts.get("warning", 0))),
            ("🔵 Bilgi", format_number(counts.get("info", 0))),
        ]
    )


def score_gauge(score: int, label: str = "") -> go.Figure:
    """Veri kalite skoru için gösterge (gauge) grafiği üretir.

    Args:
        score: 0-100 arası genel skor.
        label: Skorun sözel karşılığı.

    Returns:
        Plotly Figure.
    """
    color = "#2e7d32" if score >= 75 else "#f9a825" if score >= 50 else "#c62828"
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=int(score),
            number={"suffix": " / 100", "font": {"size": 34}},
            title={"text": label or "Veri Kalite Skoru"},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": color, "thickness": 0.7},
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "rgba(198,40,40,.12)"},
                    {"range": [40, 75], "color": "rgba(249,168,37,.14)"},
                    {"range": [75, 100], "color": "rgba(46,125,50,.14)"},
                ],
            },
        )
    )
    figure.update_layout(
        height=240, margin={"l": 20, "r": 20, "t": 50, "b": 10}, template="plotly_white"
    )
    return figure
