"""Pivot & GroupBy sayfası: pivot table, group by ve crosstab.

Bu sayfadaki işlemler veri setini değiştirmez; sonuçlar tablo, grafik ve CSV
olarak kullanılabilir. Üretilen Python kodu her zaman gösterilir.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.messages import (
    disclaimer,
    error_boundary,
    require_dataset,
    section_header,
    show_info,
    show_success,
    show_warning,
)
from components.tables import download_csv_button, show_dataframe
from core import session
from core.chart_builder import build_chart
from core.filters import AGGREGATIONS, apply_groupby, groupby_code
from core.pivot import (
    CROSSTAB_NORMALIZE,
    PIVOT_AGGREGATIONS,
    build_crosstab,
    build_pivot,
    chart_ready_pivot,
    crosstab_code,
    pivot_code,
)
from utils import dataframe_utils as dfu
from utils.helpers import format_number, safe_filename
from utils.validators import EdaError

__all__ = ["render"]


def _result_actions(table: pd.DataFrame, code: str, key: str, chart_x: str | None = None) -> None:
    """Sonuç tablosu için indirme, kod ve grafik seçeneklerini gösterir."""
    show_dataframe(table, height=min(480, 60 + 35 * len(table)))
    st.caption(
        f"{format_number(len(table))} satır × {format_number(table.shape[1])} sütun"
    )
    columns = st.columns([1, 1])
    with columns[0]:
        download_csv_button(
            table,
            "⬇️ Sonucu CSV indir",
            f"{safe_filename(session.dataset_meta().get('filename'))}_{key}.csv",
            key=f"dl_{key}",
        )
    with columns[1]:
        if st.toggle("Grafiğe dönüştür", key=f"chart_{key}"):
            with error_boundary("Grafik"):
                long_table = chart_ready_pivot(table)
                series_count = long_table["seri"].nunique()
                result = build_chart(
                    long_table,
                    "bar",
                    {
                        "x": "kategori",
                        "y": "deger",
                        "aggregation": "sum",
                        "color": "seri" if series_count > 1 else None,
                        "barmode": "group",
                        "title": f"{chart_x or 'Sonuç'} özeti",
                        "sort": "value_desc",
                    },
                )
                st.plotly_chart(result.figure, width="stretch")
                if st.button("Grafiği kaydet", key=f"save_chart_{key}"):
                    session.add_chart(result.spec)
                    show_success("Grafik kaydedildi; kod ve notebook export'una eklenecek.")

    with st.expander("🐍 Üretilen Python kodu"):
        st.code(code, language="python")


def _groupby_tab(df: pd.DataFrame) -> None:
    """Group by / aggregation sekmesi."""
    section_header(
        "Group By",
        "Bir veya birden fazla sütuna göre gruplayıp özet değer hesaplayın.",
        icon="🧮",
    )
    all_columns = [str(c) for c in df.columns]
    columns = st.columns([2, 1])
    group_by = columns[0].multiselect("Gruplama sütunları", all_columns, key="gb_cols")
    aggregation = columns[1].selectbox(
        "Aggregation",
        list(AGGREGATIONS),
        format_func=lambda key: AGGREGATIONS[key],
        key="gb_agg",
    )
    value_options = (
        dfu.numeric_columns(df) if aggregation in ("sum", "mean", "median") else all_columns
    )
    values = st.multiselect(
        "Değer sütunları",
        [c for c in value_options if c not in group_by],
        key="gb_values",
        help="count için boş bırakabilirsiniz.",
    )

    if not group_by:
        show_info("Başlamak için en az bir gruplama sütunu seçin.")
        return

    with error_boundary("Group by"):
        table = apply_groupby(df, group_by, values, aggregation)
        code = groupby_code(group_by, values, aggregation)
        _result_actions(table, code, "groupby", group_by[0])


def _pivot_tab(df: pd.DataFrame) -> None:
    """Pivot table sekmesi."""
    section_header(
        "Pivot Table",
        "Satır / sütun kırılımı ve değer özeti seçerek çapraz tablo üretin.",
        icon="🧊",
    )
    all_columns = [str(c) for c in df.columns]
    low_cardinality = [
        c for c in all_columns if 1 < dfu.safe_nunique(df[c]) <= 50
    ] or all_columns

    columns = st.columns(2)
    rows = columns[0].multiselect("Satırlar (Rows)", all_columns, key="pv_rows")
    column_fields = columns[1].multiselect(
        "Sütunlar (Columns)",
        [c for c in low_cardinality if c not in rows],
        key="pv_cols",
        help="Boş bırakabilirsiniz. Çok kategorili sütunlar tabloyu okunamaz hale getirir.",
    )

    options = st.columns(2)
    aggregation = options[0].selectbox(
        "Aggregation",
        list(PIVOT_AGGREGATIONS),
        index=list(PIVOT_AGGREGATIONS).index("mean"),
        format_func=lambda key: PIVOT_AGGREGATIONS[key],
        key="pv_agg",
    )
    value_options = (
        dfu.numeric_columns(df)
        if aggregation in ("sum", "mean", "median", "std")
        else all_columns
    )
    value = options[1].selectbox(
        "Değer (Values)",
        ["(adet say)"] + [c for c in value_options if c not in rows + column_fields],
        key="pv_value",
    )
    values = None if value == "(adet say)" else value

    with st.expander("Gelişmiş ayarlar"):
        fill_zero = st.toggle("Boş hücreleri 0 ile doldur", value=False, key="pv_fill")
        margins = st.toggle("Satır/sütun toplamlarını ekle", value=False, key="pv_margins")

    if not rows:
        show_info("Başlamak için en az bir satır (Rows) sütunu seçin.")
        return

    with error_boundary("Pivot table"):
        table = build_pivot(
            df,
            rows,
            column_fields,
            values,
            aggregation,
            fill_value=0 if fill_zero else None,
            margins=margins,
        )
        code = pivot_code(
            rows,
            column_fields,
            values,
            aggregation,
            fill_value=0 if fill_zero else None,
            margins=margins,
        )
        _result_actions(table, code, "pivot", rows[0])


def _crosstab_tab(df: pd.DataFrame) -> None:
    """Crosstab sekmesi."""
    section_header(
        "Crosstab (Çapraz Tablo)",
        "İki kategorik sütunun kesişim adetlerini veya yüzdelerini görün.",
        icon="🔢",
    )
    candidates = [
        c for c in dfu.categorical_columns(df) if 1 < dfu.safe_nunique(df[c]) <= 50
    ]
    if len(candidates) < 2:
        show_info(
            "Crosstab için en az 2 kategorik sütun gerekir (en fazla 50 farklı değer)."
        )
        return

    columns = st.columns(3)
    row_column = columns[0].selectbox("Satır sütunu", candidates, key="ct_row")
    column_column = columns[1].selectbox(
        "Sütun sütunu", [c for c in candidates if c != row_column], key="ct_col"
    )
    normalize = columns[2].selectbox(
        "Gösterim",
        list(CROSSTAB_NORMALIZE),
        format_func=lambda key: CROSSTAB_NORMALIZE[key][0],
        key="ct_norm",
    )
    margins = st.toggle("Toplamları göster", value=True, key="ct_margins")

    with error_boundary("Crosstab"):
        table = build_crosstab(df, row_column, column_column, normalize, margins)
        code = crosstab_code(row_column, column_column, normalize, margins)
        if normalize != "count":
            st.caption("Değerler yüzde (%) olarak gösteriliyor.")
        _result_actions(table, code, "crosstab", row_column)

        with st.expander("🔗 İki kategorik sütun arasındaki ilişki gücü"):
            from core.correlations import cramers_v

            try:
                association = cramers_v(df, row_column, column_column)
            except EdaError as exc:
                show_warning(str(exc))
            else:
                metrics = st.columns(3)
                metrics[0].metric("Cramér's V", f"{association['v']:.3f}")
                metrics[1].metric("Güç", association["strength"])
                if association["p_value"] is not None:
                    metrics[2].metric(
                        "Chi-Square p-değeri", f"{association['p_value']:.4g}"
                    )
                st.caption(
                    "Cramér's V 0 ile 1 arasındadır: 0 ilişki yok, 1 tam ilişki. "
                    "İstatistiksel ilişki nedensellik anlamına gelmez."
                )


def render() -> None:
    """Pivot & GroupBy sayfasını çizer."""
    st.title("🧊 Pivot & GroupBy")
    if not require_dataset("Pivot ve gruplama"):
        return

    df, filtered = session.get_analysis_df()
    if filtered:
        st.caption(f"ℹ️ Filtre aktif: {format_number(len(df))} satır kullanılıyor.")

    st.caption(
        "Bu sayfadaki işlemler veri setinizi değiştirmez; sonuçları indirebilir "
        "veya grafiğe dönüştürebilirsiniz."
    )

    tabs = st.tabs(["🧮 Group By", "🧊 Pivot Table", "🔢 Crosstab"])
    with tabs[0], error_boundary("Group by"):
        _groupby_tab(df)
    with tabs[1], error_boundary("Pivot"):
        _pivot_tab(df)
    with tabs[2], error_boundary("Crosstab"):
        _crosstab_tab(df)

    disclaimer(
        "Gruplama ve pivot sonuçları yalnızca görüntüleme amaçlıdır. Veri setini "
        "kalıcı değiştirmek için Temizleme ve Dönüşümler sayfalarını kullanın."
    )
