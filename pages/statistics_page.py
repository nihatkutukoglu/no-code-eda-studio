"""İstatistiksel Analiz sayfası: sayısal, kategorik ve tarih sütun özetleri."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from components.messages import (
    disclaimer,
    error_boundary,
    require_dataset,
    section_header,
    show_info,
)
from components.metric_cards import metric_row
from components.tables import download_csv_button, show_dataframe
from core import session, statistics as stats_mod
from core.chart_builder import PLOTLY_TEMPLATE
from utils import dataframe_utils as dfu
from utils.helpers import format_number, safe_filename

__all__ = ["render"]


def _numeric_section(df) -> None:
    """Sayısal sütun istatistikleri."""
    numeric = dfu.numeric_columns(df)
    if not numeric:
        show_info(
            "Veri setinde sayısal sütun yok. Metin olarak saklanan sayıları "
            "Veri Temizleme sayfasından dönüştürebilirsiniz."
        )
        return

    column = st.selectbox("Sayısal sütun", numeric, key="stats_numeric_col")
    summary = stats_mod.numeric_summary(df, column)

    metric_row(
        [
            ("Dolu", format_number(summary["count"])),
            ("Boş", format_number(summary["missing"])),
            ("Ortalama", format_number(summary["mean"], 3)),
            ("Medyan", format_number(summary["median"], 3)),
            ("Std", format_number(summary["std"], 3)),
        ],
        columns=5,
    )
    metric_row(
        [
            ("Min", format_number(summary["min"], 3)),
            ("Q1", format_number(summary["q1"], 3)),
            ("Q3", format_number(summary["q3"], 3)),
            ("Max", format_number(summary["max"], 3)),
            ("IQR", format_number(summary["iqr"], 3)),
        ],
        columns=5,
    )

    left, right = st.columns([1, 1])
    with left:
        show_dataframe(stats_mod.numeric_summary_table(df, column))
    with right:
        st.markdown("**Kısa yorumlar**")
        skewness, kurtosis = summary["skewness"], summary["kurtosis"]
        st.markdown(
            f"- Çarpıklık: **{format_number(skewness, 3)}** — "
            f"{stats_mod.STAT_EXPLANATIONS['skewness']}"
        )
        st.markdown(
            f"- Basıklık: **{format_number(kurtosis, 3)}** — "
            f"{stats_mod.STAT_EXPLANATIONS['kurtosis']}"
        )
        if summary["std"] and abs(summary["mean"] - summary["median"]) > summary["std"]:
            st.markdown(
                "- Ortalama ile medyan arasındaki fark büyük; dağılım muhtemelen "
                "aykırı değerlerden etkileniyor. Medyan daha temsili olabilir."
            )
        if summary["zeros"]:
            st.markdown(f"- Sıfır değeri sayısı: **{format_number(summary['zeros'])}**")
        if summary["negatives"]:
            st.markdown(f"- Negatif değer sayısı: **{format_number(summary['negatives'])}**")
        normality = stats_mod.shapiro_normality(df, column)
        if normality:
            verdict = "normale yakın" if normality["normal"] else "normalden farklı"
            st.markdown(
                f"- Shapiro-Wilk testi (n={format_number(normality['sample_size'])}): "
                f"p = {format_number(normality['p_value'], 4)} → dağılım **{verdict}**."
            )

    bins = st.slider("Histogram bin sayısı", 5, 100, 30, key="stats_bins")
    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.plotly_chart(
            px.histogram(
                df, x=column, nbins=bins, title=f"{column} histogramı",
                template=PLOTLY_TEMPLATE,
            ),
            width="stretch",
        )
    with chart_right:
        st.plotly_chart(
            px.box(
                df, y=column, points="outliers", title=f"{column} kutu grafiği",
                template=PLOTLY_TEMPLATE,
            ),
            width="stretch",
        )

    with st.expander("📊 Tüm sayısal sütunların karşılaştırması"):
        table = stats_mod.describe_all_numeric(df)
        show_dataframe(table)
        download_csv_button(
            table,
            "⬇️ Özet tabloyu indir",
            f"{safe_filename(session.dataset_meta().get('filename'))}_sayisal_ozet.csv",
            key="dl_numeric_summary",
        )


def _categorical_section(df) -> None:
    """Kategorik sütun istatistikleri."""
    categorical = dfu.categorical_columns(df)
    if not categorical:
        show_info("Veri setinde kategorik/metin sütun yok.")
        return

    column = st.selectbox("Kategorik sütun", categorical, key="stats_cat_col")
    top_n = st.slider("Gösterilecek değer sayısı (Top N)", 3, 50, 10, key="stats_cat_topn")
    summary = stats_mod.categorical_summary(df, column, top_n)

    metric_row(
        [
            ("Dolu", format_number(summary["count"])),
            ("Boş", format_number(summary["missing"])),
            ("Tekil değer", format_number(summary["unique"])),
            ("Mod (en sık)", str(summary["mode"])),
            ("Mod frekansı", format_number(summary["mode_frequency"])),
        ],
        columns=5,
    )

    table = summary["top_values"]
    left, right = st.columns([1, 1])
    with left:
        show_dataframe(table)
        download_csv_button(
            table,
            "⬇️ Frekans tablosunu indir",
            f"{safe_filename(column)}_frekans.csv",
            key="dl_freq",
        )
    with right:
        figure = px.bar(
            table,
            x="Frekans",
            y="Değer",
            orientation="h",
            title=f"{column} - en sık {len(table)} değer",
            template=PLOTLY_TEMPLATE,
        )
        figure.update_layout(yaxis={"autorange": "reversed"}, yaxis_title="")
        st.plotly_chart(figure, width="stretch")

    if summary["imbalance_ratio"] >= 0.80:
        st.warning(
            f"'{summary['mode']}' değeri tek başına %"
            f"{summary['mode_ratio']:.1f} pay alıyor; kategori dağılımı çok dengesiz.",
            icon="⚠️",
        )
    disclaimer(stats_mod.STAT_EXPLANATIONS["mode"])


def _datetime_section(df) -> None:
    """Tarih sütunu istatistikleri."""
    datetimes = dfu.datetime_columns(df)
    if not datetimes:
        show_info(
            "Veri setinde tarih tipinde sütun yok. Metin olarak saklanan tarihleri "
            "Veri Temizleme sayfasından `datetime` tipine dönüştürebilirsiniz."
        )
        return

    column = st.selectbox("Tarih sütunu", datetimes, key="stats_date_col")
    summary = stats_mod.datetime_summary(df, column)

    metric_row(
        [
            ("En eski tarih", str(summary["min_date"])[:19]),
            ("En yeni tarih", str(summary["max_date"])[:19]),
            ("Aralık (gün)", format_number(summary["range_days"])),
            ("Tekil gün", format_number(summary["unique_days"])),
            ("Boş", format_number(summary["missing"])),
        ],
        columns=5,
    )

    tabs = st.tabs(["Yıl", "Ay", "Haftanın günü"])
    with tabs[0]:
        st.plotly_chart(
            px.bar(summary["by_year"], x="Yıl", y="Adet", title="Yıla göre dağılım",
                   template=PLOTLY_TEMPLATE),
            width="stretch",
        )
    with tabs[1]:
        st.plotly_chart(
            px.bar(summary["by_month"], x="Ay", y="Adet", title="Aya göre dağılım",
                   template=PLOTLY_TEMPLATE),
            width="stretch",
        )
    with tabs[2]:
        st.plotly_chart(
            px.bar(summary["by_weekday"], x="Gün", y="Adet",
                   title="Haftanın gününe göre dağılım", template=PLOTLY_TEMPLATE),
            width="stretch",
        )


def render() -> None:
    """İstatistiksel Analiz sayfasını çizer."""
    st.title("📐 İstatistiksel Analiz")
    if not require_dataset("İstatistiksel analiz"):
        return

    df, filtered = session.get_analysis_df()
    if filtered:
        st.caption(
            f"ℹ️ Grafik Oluşturucu'da tanımlanan filtre aktif: "
            f"{format_number(len(df))} satır analiz ediliyor."
        )

    section_header(
        "Sütun seçin ve istatistikleri inceleyin",
        "Sayısal, kategorik ve tarih sütunları için ayrı özetler sunulur.",
    )
    tabs = st.tabs(["🔢 Sayısal", "🔠 Kategorik", "📅 Tarih"])
    with tabs[0], error_boundary("Sayısal istatistik"):
        _numeric_section(df)
    with tabs[1], error_boundary("Kategorik istatistik"):
        _categorical_section(df)
    with tabs[2], error_boundary("Tarih istatistiği"):
        _datetime_section(df)
