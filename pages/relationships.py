"""Korelasyon ve İlişki Analizi sayfası."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from components.messages import (
    disclaimer,
    error_boundary,
    require_dataset,
    section_header,
    show_info,
    show_success,
)
from components.metric_cards import metric_row
from components.tables import download_csv_button, show_dataframe
from core import session
from core.chart_builder import PLOTLY_TEMPLATE
from core.correlations import (
    CORRELATION_METHODS,
    DEFAULT_THRESHOLD,
    categorical_association,
    categorical_association_code,
    correlation_code,
    cramers_v,
    pair_correlation,
    strong_pairs,
)
from utils import caching, dataframe_utils as dfu
from utils.helpers import format_number, safe_filename

__all__ = ["render"]

_CAUSALITY_WARNING = (
    "Korelasyon nedensellik göstermez: iki değişkenin birlikte hareket etmesi, "
    "birinin diğerine sebep olduğu anlamına gelmez."
)


def _matrix_section(df) -> None:
    """Korelasyon matrisi ve güçlü ilişkiler bölümü."""
    numeric = dfu.numeric_columns(df)
    if len(numeric) < 2:
        show_info(
            "Korelasyon için en az 2 sayısal sütun gerekir. "
            f"Şu an {len(numeric)} sayısal sütun var."
        )
        return

    selected = st.multiselect(
        "Sayısal sütunlar",
        numeric,
        default=numeric[: min(10, len(numeric))],
        key="corr_columns",
    )
    columns_row = st.columns([1, 1])
    method = columns_row[0].selectbox(
        "Yöntem",
        list(CORRELATION_METHODS),
        format_func=lambda key: CORRELATION_METHODS[key],
        key="corr_method",
    )
    threshold = columns_row[1].slider(
        "Güçlü ilişki eşiği |r|", 0.10, 0.99, float(DEFAULT_THRESHOLD), 0.05, key="corr_threshold"
    )

    if len(selected) < 2:
        show_info("En az 2 sütun seçmelisiniz.")
        return

    with st.spinner("Korelasyon hesaplanıyor..."):
        matrix = caching.correlation_matrix(df, tuple(selected), method)

    figure = px.imshow(
        matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title=f"Korelasyon matrisi ({CORRELATION_METHODS[method]})",
        template=PLOTLY_TEMPLATE,
    )
    figure.update_layout(height=max(380, 60 + 40 * len(selected)))
    st.plotly_chart(figure, width="stretch")

    with st.expander("Korelasyon matrisi tablosu"):
        show_dataframe(matrix.round(4).reset_index(names="Sütun"))
        download_csv_button(
            matrix.round(6).reset_index(names="Sütun"),
            "⬇️ Matrisi CSV indir",
            f"{safe_filename(session.dataset_meta().get('filename'))}_korelasyon.csv",
            key="dl_corr",
        )

    section_header(
        "Güçlü İlişkiler",
        f"|r| ≥ {threshold:.2f} olan çiftler. Bir sütunun kendisiyle korelasyonu listelenmez.",
        icon="🔗",
    )
    pairs = strong_pairs(matrix, threshold)
    if pairs.empty:
        show_success(f"|r| ≥ {threshold:.2f} eşiğini aşan ilişki bulunamadı.")
    else:
        metric_row(
            [
                ("Güçlü ilişki sayısı", format_number(len(pairs))),
                ("En yüksek |r|", format_number(float(pairs['|r|'].max()), 3)),
            ]
        )
        show_dataframe(pairs)
        download_csv_button(
            pairs, "⬇️ Güçlü ilişkileri indir", "guclu_iliskiler.csv", key="dl_pairs"
        )

    with st.expander("🐍 Bu analizin Python kodu"):
        st.code(correlation_code(selected, method), language="python")

    disclaimer(_CAUSALITY_WARNING, icon="⚠️")


def _pair_section(df) -> None:
    """İki sütun arası ilişki grafiği bölümü."""
    numeric = dfu.numeric_columns(df)
    if len(numeric) < 2:
        show_info("İlişki grafiği için en az 2 sayısal sütun gerekir.")
        return

    columns_row = st.columns(3)
    x = columns_row[0].selectbox("X ekseni", numeric, key="pair_x")
    y = columns_row[1].selectbox(
        "Y ekseni", [c for c in numeric if c != x] or numeric, key="pair_y"
    )
    color = columns_row[2].selectbox(
        "Renk (opsiyonel)", ["(yok)", *dfu.categorical_columns(df)], key="pair_color"
    )
    color_column = None if color == "(yok)" else color

    method = st.radio(
        "Korelasyon yöntemi",
        list(CORRELATION_METHODS),
        format_func=lambda key: CORRELATION_METHODS[key],
        horizontal=True,
        key="pair_method",
    )

    result = pair_correlation(df, x, y, method)
    metric_row(
        [
            ("Korelasyon (r)", format_number(result["r"], 4)),
            ("Güç", result["strength"]),
            ("Ortak gözlem", format_number(result["n"])),
        ]
    )

    plot_df = df
    note = ""
    if len(df) > 20_000:
        plot_df = df.sample(20_000, random_state=42)
        note = " (20.000 satırlık rastgele örneklem)"
    figure = px.scatter(
        plot_df,
        x=x,
        y=y,
        color=color_column,
        opacity=0.7,
        title=f"{x} - {y} ilişkisi{note}",
        template=PLOTLY_TEMPLATE,
    )
    st.plotly_chart(figure, width="stretch")
    st.caption(
        "Trendline çizilmedi: ilk sürümde yalnızca temel ilişki grafiği sunulur."
    )
    disclaimer(_CAUSALITY_WARNING, icon="⚠️")


def _categorical_section(df) -> None:
    """Kategorik sütunlar arası ilişki (Chi-Square / Cramér's V) bölümü."""
    candidates = [
        column
        for column in dfu.categorical_columns(df)
        if 2 <= dfu.safe_nunique(df[column]) <= 50
    ]
    if len(candidates) < 2:
        show_info(
            "Kategorik ilişki analizi için en az 2 kategorik sütun gerekir "
            "(her biri en fazla 50 farklı değer)."
        )
        return

    st.caption(
        "Cramér's V, iki kategorik sütunun birlikte değişip değişmediğini 0-1 "
        "aralığında ölçer. Chi-Square testi bu ilişkinin rastlantısal olup "
        "olmadığına dair bir p-değeri verir."
    )
    columns_row = st.columns(2)
    first = columns_row[0].selectbox("Birinci sütun", candidates, key="cat_first")
    second = columns_row[1].selectbox(
        "İkinci sütun", [c for c in candidates if c != first] or candidates, key="cat_second"
    )

    result = cramers_v(df, first, second)
    metric_row(
        [
            ("Cramér's V", format_number(result["v"], 3)),
            ("İlişki gücü", result["strength"]),
            (
                "Chi-Square p-değeri",
                format_number(result["p_value"], 5) if result["p_value"] is not None else "-",
            ),
            ("Gözlem", format_number(result["n"])),
        ],
        columns=4,
    )
    if result["p_value"] is not None:
        if result["significant"]:
            st.success(
                "p < 0.05: gözlenen ilişkinin rastlantısal olma olasılığı düşük.",
                icon="✅",
            )
        else:
            show_info("p >= 0.05: ilişki istatistiksel olarak anlamlı görünmüyor.")

    with st.expander("Çapraz tablo (adet)"):
        show_dataframe(result["table"].reset_index())

    section_header(
        "Tüm Kategorik Çiftler",
        "Eşiği aşan ilişkiler Cramér's V değerine göre sıralanır.",
        icon="🔗",
    )
    threshold = st.slider(
        "Cramér's V eşiği", 0.10, 0.95, 0.30, 0.05, key="cat_threshold"
    )
    table = categorical_association(df, candidates, threshold)
    if table.empty:
        show_success(f"Cramér's V ≥ {threshold:.2f} olan kategorik ilişki bulunamadı.")
    else:
        show_dataframe(table)
        download_csv_button(
            table, "⬇️ İlişkileri indir", "kategorik_iliskiler.csv", key="dl_cat_assoc"
        )

    with st.expander("🐍 Bu analizin Python kodu"):
        st.code(categorical_association_code(first, second), language="python")

    disclaimer(_CAUSALITY_WARNING, icon="⚠️")


def render() -> None:
    """Korelasyon / İlişki sayfasını çizer."""
    st.title("🔗 İlişkiler")
    if not require_dataset("İlişki analizi"):
        return

    df, filtered = session.get_analysis_df()
    if filtered:
        st.caption(f"ℹ️ Filtre aktif: {format_number(len(df))} satır analiz ediliyor.")

    tabs = st.tabs(
        ["🧮 Korelasyon Matrisi", "⚫ İki Sütun İlişkisi", "🔠 Kategorik İlişkiler"]
    )
    with tabs[0], error_boundary("Korelasyon analizi"):
        _matrix_section(df)
    with tabs[1], error_boundary("İlişki grafiği"):
        _pair_section(df)
    with tabs[2], error_boundary("Kategorik ilişki analizi"):
        _categorical_section(df)
