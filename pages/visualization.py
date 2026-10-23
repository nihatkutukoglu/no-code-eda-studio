"""Görselleştirme sayfası: akıllı öneriler, sade grafik formu ve filtreler.

Arayüz ilkesi: zorunlu alanlar görünür, geri kalan her şey "Gelişmiş ayarlar"
içinde. Grafik tipi seçildiğinde form akıllı varsayılanlarla dolar.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from components.filter_builder import filter_summary_caption, render_filter_builder
from components.messages import (
    disclaimer,
    empty_state,
    error_boundary,
    require_dataset,
    section_header,
    show_error,
    show_info,
    show_success,
    show_warning,
)
from components.tables import download_csv_button, show_dataframe
from core import session
from core.chart_builder import (
    AGGREGATIONS,
    BAR_MODES,
    CHART_TYPES,
    SORT_OPTIONS,
    build_chart,
    default_params,
    suggest_charts,
)
from core.correlations import CORRELATION_METHODS
from core.exporter import figure_to_html_bytes
from core.filters import COMBINE_MODES, apply_filters, apply_groupby, filters_to_code, groupby_code
from utils import dataframe_utils as dfu
from utils.helpers import format_number, safe_filename
from utils.validators import EdaError

__all__ = ["render"]

_SOURCE_DATASET = "Veri seti"
_SOURCE_GROUPED = "Group By sonucu"


def _optional(options: list[str], label: str, key: str, help_text: str | None = None) -> str | None:
    """Opsiyonel sütun seçici (boş seçenek dahil)."""
    choice = st.selectbox(label, ["(yok)", *options], key=key, help=help_text)
    return None if choice == "(yok)" else choice


def _index_of(options: list[str], value: Any, fallback: int = 0) -> int:
    """Varsayılan değerin seçenek listesindeki indeksini döndürür."""
    try:
        return options.index(value)
    except (ValueError, TypeError):
        return fallback if options else 0


def _data_source_section() -> tuple[pd.DataFrame, str, str]:
    """Filtre ve group by bölümünü çizer.

    Returns:
        ``(dataframe, kod_değişken_adı, kod_öneki)``
    """
    df = session.get_working_df()

    with st.expander("🔎 Filtreler (yalnızca görselleştirme ve analiz için)"):
        combine = st.radio(
            "Koşulları birleştir",
            list(COMBINE_MODES),
            format_func=lambda key: COMBINE_MODES[key],
            horizontal=True,
            key="viz_combine",
        )
        filters = render_filter_builder(
            df,
            key_prefix="viz",
            title="Geçici filtreler",
            description=(
                "Filtreler veri setini **değiştirmez**. Kalıcı uygulamak için "
                "Temizleme → Filtre sekmesini kullanın."
            ),
            combine=combine,
        )
        enabled = st.toggle(
            "Filtreleri diğer analiz sayfalarında da kullan",
            value=session.filters_active(),
            key="viz_filters_enabled",
            help="Açıkken İstatistik, İlişkiler, Aykırı Değerler ve Pivot "
            "sayfaları da filtrelenmiş veriyi kullanır.",
        )
        session.set_active_filters(filters, enabled, combine)

    source_df, filtered = df, False
    if filters:
        try:
            source_df = apply_filters(df, filters, combine)
            filtered = True
        except EdaError as exc:
            show_error(exc)
    filter_summary_caption(df, source_df, filtered)

    code_prefix = ""
    if filtered:
        code_prefix = "# Grafik için geçici filtre\n" + filters_to_code(
            filters, target="plot_source", combine=combine
        )

    with st.expander("🧮 Hızlı Group By (grafik kaynağı olarak kullanılabilir)"):
        group_cols = st.multiselect(
            "Gruplama sütunları", [str(c) for c in source_df.columns], key="viz_group_cols"
        )
        aggregation = st.selectbox(
            "Aggregation",
            list(AGGREGATIONS),
            format_func=lambda key: AGGREGATIONS[key],
            key="viz_group_agg",
        )
        value_cols = st.multiselect(
            "Değer sütunları",
            dfu.numeric_columns(source_df)
            if aggregation in ("sum", "mean", "median")
            else [str(c) for c in source_df.columns],
            key="viz_group_values",
            help="count için boş bırakabilirsiniz.",
        )
        if st.button("Gruplamayı hesapla", key="viz_group_run"):
            with error_boundary("Group by"):
                grouped = apply_groupby(source_df, group_cols, value_cols, aggregation)
                st.session_state["viz_grouped"] = grouped
                st.session_state["viz_grouped_code"] = groupby_code(
                    group_cols, value_cols, aggregation, target="grouped_df"
                )
                show_success(f"Gruplama tamamlandı: {format_number(len(grouped))} satır.")
        grouped_df = st.session_state.get("viz_grouped")
        if isinstance(grouped_df, pd.DataFrame) and not grouped_df.empty:
            show_dataframe(grouped_df.head(30))
            download_csv_button(
                grouped_df,
                "⬇️ Gruplama sonucunu indir",
                f"{safe_filename(session.dataset_meta().get('filename'))}_gruplama.csv",
                key="dl_grouped",
            )
        st.caption("Daha gelişmiş gruplama ve pivot için **Pivot & GroupBy** sayfası.")

    grouped_df = st.session_state.get("viz_grouped")
    has_grouped = isinstance(grouped_df, pd.DataFrame) and not grouped_df.empty
    source_choice = _SOURCE_DATASET
    if has_grouped:
        source_choice = st.radio(
            "Grafik verisi",
            [_SOURCE_DATASET, _SOURCE_GROUPED],
            horizontal=True,
            key="viz_source",
        )

    if source_choice == _SOURCE_GROUPED and has_grouped:
        return grouped_df, "grouped_df", st.session_state.get("viz_grouped_code", "")
    if filtered:
        return source_df, "plot_source", code_prefix
    return source_df, "df", ""


def _suggestion_section(df: pd.DataFrame) -> None:
    """Kural tabanlı grafik önerilerini gösterir."""
    suggestions = suggest_charts(df)
    if not suggestions:
        return
    section_header(
        "Önerilen Grafikler",
        "Sütun tiplerine göre otomatik öneriler (AI kullanılmaz).",
        icon="✨",
    )
    columns = st.columns(min(3, len(suggestions)))
    for index, suggestion in enumerate(suggestions[:3]):
        with columns[index % len(columns)]:
            with st.container(border=True):
                st.markdown(f"**{suggestion['label']}**")
                st.caption(suggestion["reason"])
                if st.button(
                    "Bu grafiği oluştur",
                    key=f"suggest_{index}",
                    width="stretch",
                ):
                    st.session_state["viz_pending_suggestion"] = suggestion
                    st.rerun()
    if len(suggestions) > 3:
        with st.expander(f"Diğer öneriler ({len(suggestions) - 3})"):
            for index, suggestion in enumerate(suggestions[3:], start=3):
                st.markdown(f"**{suggestion['label']}** — {suggestion['reason']}")
                if st.button("Bu grafiği oluştur", key=f"suggest_{index}"):
                    st.session_state["viz_pending_suggestion"] = suggestion
                    st.rerun()


def _chart_params_form(
    df: pd.DataFrame, chart_type: str, preset: dict[str, Any]
) -> dict[str, Any]:
    """Grafik tipine göre zorunlu alanları ve gelişmiş ayarları çizer."""
    numeric = dfu.numeric_columns(df)
    categorical = dfu.categorical_columns(df)
    datetimes = dfu.datetime_columns(df)
    all_columns = [str(c) for c in df.columns]
    low_cardinality = [c for c in categorical if 2 <= dfu.safe_nunique(df[c]) <= 30]
    params: dict[str, Any] = {}

    def select(label: str, options: list[str], key: str, preset_key: str) -> str:
        choices = options or all_columns
        return st.selectbox(
            label, choices, index=_index_of(choices, preset.get(preset_key)), key=key
        )

    if chart_type == "bar":
        columns = st.columns(3)
        with columns[0]:
            params["x"] = select(
                "Kategori (X)", categorical + datetimes + numeric, "bar_x", "x"
            )
        with columns[1]:
            params["aggregation"] = st.selectbox(
                "Hesap",
                list(AGGREGATIONS),
                index=_index_of(list(AGGREGATIONS), preset.get("aggregation"), 0),
                format_func=lambda key: AGGREGATIONS[key],
                key="bar_agg",
            )
        with columns[2]:
            params["y"] = (
                _optional(numeric, "Değer (Y)", "bar_y")
                if params["aggregation"] != "count"
                else None
            )
        with st.expander("Gelişmiş ayarlar"):
            params["color"] = _optional(low_cardinality, "Renk / seri", "bar_color")
            if params["color"]:
                params["barmode"] = st.radio(
                    "Çubuk modu",
                    list(BAR_MODES),
                    format_func=lambda key: BAR_MODES[key],
                    horizontal=True,
                    key="bar_mode",
                )
            params["sort"] = st.selectbox(
                "Sıralama",
                list(SORT_OPTIONS),
                format_func=lambda key: SORT_OPTIONS[key],
                key="bar_sort",
            )
            params["top_n"] = st.number_input(
                "En büyük N kategori (0 = sınır yok)",
                min_value=0,
                max_value=200,
                value=20,
                key="bar_topn",
            ) or None
            params["horizontal"] = st.toggle("Yatay çubuk", key="bar_horizontal")
            _label_inputs(params, "bar")

    elif chart_type in ("line", "area"):
        columns = st.columns(2)
        with columns[0]:
            params["x"] = select("X ekseni", datetimes + numeric + categorical, f"{chart_type}_x", "x")
        with columns[1]:
            params["y"] = select("Y ekseni (sayısal)", numeric, f"{chart_type}_y", "y")
        with st.expander("Gelişmiş ayarlar"):
            params["color"] = _optional(low_cardinality, "Renk / seri", f"{chart_type}_color")
            if chart_type == "line":
                params["markers"] = st.toggle("Noktaları göster", key="line_markers")
            _label_inputs(params, chart_type)

    elif chart_type in ("scatter", "bubble", "density_heatmap"):
        columns = st.columns(2)
        with columns[0]:
            params["x"] = select("X (sayısal)", numeric, f"{chart_type}_x", "x")
        with columns[1]:
            params["y"] = select("Y (sayısal)", numeric[::-1], f"{chart_type}_y", "y")
        if chart_type == "bubble":
            params["size"] = st.selectbox(
                "Nokta boyutu (sayısal)",
                numeric,
                index=_index_of(numeric, preset.get("size")),
                key="bubble_size",
            )
        if chart_type == "density_heatmap":
            params["bins"] = st.slider("Bin sayısı", 10, 80, 30, key="density_bins")
        with st.expander("Gelişmiş ayarlar"):
            if chart_type != "density_heatmap":
                params["color"] = _optional(
                    low_cardinality + numeric, "Renk", f"{chart_type}_color"
                )
                if chart_type == "scatter":
                    params["size"] = _optional(numeric, "Nokta boyutu", "scatter_size")
            _label_inputs(params, chart_type)

    elif chart_type == "histogram":
        columns = st.columns(2)
        with columns[0]:
            params["column"] = select("Sütun (sayısal)", numeric, "hist_col", "column")
        with columns[1]:
            params["bins"] = st.slider("Bin sayısı", 5, 100, int(preset.get("bins", 30)), key="hist_bins")
        with st.expander("Gelişmiş ayarlar"):
            params["color"] = _optional(low_cardinality, "Renk / seri", "hist_color")
            _label_inputs(params, "hist")

    elif chart_type in ("box", "violin"):
        columns = st.columns(2)
        with columns[0]:
            params["y"] = select("Değer (sayısal)", numeric, f"{chart_type}_y", "y")
        with columns[1]:
            params["x"] = _optional(low_cardinality, "Kategori (opsiyonel)", f"{chart_type}_x")
        with st.expander("Gelişmiş ayarlar"):
            params["color"] = _optional(low_cardinality, "Renk / seri", f"{chart_type}_color")
            _label_inputs(params, chart_type)

    elif chart_type in ("pie", "donut"):
        columns = st.columns(3)
        with columns[0]:
            params["names"] = select("Kategori", categorical, f"{chart_type}_names", "names")
        with columns[1]:
            params["aggregation"] = st.selectbox(
                "Hesap",
                list(AGGREGATIONS),
                format_func=lambda key: AGGREGATIONS[key],
                key=f"{chart_type}_agg",
            )
        with columns[2]:
            params["values"] = (
                _optional(numeric, "Değer", f"{chart_type}_values")
                if params["aggregation"] != "count"
                else None
            )
        with st.expander("Gelişmiş ayarlar"):
            params["top_n"] = st.slider(
                "Gösterilecek kategori sayısı", 2, 30, 8, key=f"{chart_type}_topn"
            )
            _label_inputs(params, chart_type)

    elif chart_type == "category_heatmap":
        columns = st.columns(2)
        with columns[0]:
            params["x"] = select("X (kategori)", low_cardinality or categorical, "cat_heat_x", "x")
        with columns[1]:
            remaining = [c for c in (low_cardinality or categorical) if c != params["x"]]
            params["y"] = select("Y (kategori)", remaining, "cat_heat_y", "y")
        with st.expander("Gelişmiş ayarlar"):
            _label_inputs(params, "cat_heat")

    elif chart_type == "heatmap":
        params["columns"] = st.multiselect(
            "Sayısal sütunlar",
            numeric,
            default=preset.get("columns") or numeric[: min(8, len(numeric))],
            key="heat_cols",
        )
        params["method"] = st.selectbox(
            "Yöntem",
            list(CORRELATION_METHODS),
            format_func=lambda key: CORRELATION_METHODS[key],
            key="heat_method",
        )
        disclaimer("Korelasyon nedensellik göstermez.")

    return params


def _label_inputs(params: dict[str, Any], key_prefix: str) -> None:
    """Başlık ve eksen etiketi girişlerini ekler."""
    params["title"] = st.text_input("Başlık (opsiyonel)", key=f"{key_prefix}_title")
    columns = st.columns(2)
    params["x_label"] = columns[0].text_input("X ekseni etiketi", key=f"{key_prefix}_xlabel")
    params["y_label"] = columns[1].text_input("Y ekseni etiketi", key=f"{key_prefix}_ylabel")
    params["show_legend"] = st.toggle(
        "Açıklama (legend) göster", value=True, key=f"{key_prefix}_legend"
    )


def _render_chart_result(result: Any) -> None:
    """Oluşturulan grafiği, uyarıları ve kodunu gösterir."""
    st.plotly_chart(result.figure, width="stretch")
    for message in result.warnings:
        show_warning(message)
    with st.expander("🐍 Bu grafiğin Python kodu"):
        st.code(result.spec["python_code"], language="python")
    st.download_button(
        "⬇️ Grafiği HTML olarak indir",
        data=figure_to_html_bytes(result.figure),
        file_name=f"{safe_filename(result.spec['title'])}.html",
        mime="text/html",
        key="dl_chart_html",
    )


def _render_saved_charts() -> None:
    """Kaydedilen grafiklerin listesini ve kodlarını gösterir."""
    charts = session.generated_charts()
    section_header(
        "Kaydedilen Grafikler",
        "Bu grafikler Python kodu, Notebook ve HTML rapor export'una dahil edilir.",
        icon="🗂️",
    )
    if not charts:
        empty_state(
            "Henüz grafik kaydetmediniz",
            "Yukarıdan bir grafik oluşturduğunuzda otomatik olarak burada listelenir.",
            icon="📈",
        )
        return

    for index, chart in enumerate(charts, start=1):
        with st.expander(f"{index}. {chart.get('title')} · {chart.get('label')}"):
            st.code(chart.get("python_code", ""), language="python")
            st.caption(f"Veri kaynağı: {chart.get('source_note', 'working_df')}")
            if st.button("🗑️ Grafiği kaldır", key=f"del_chart_{chart.get('id')}"):
                session.remove_chart(chart.get("id"))
                st.rerun()

    if st.button("🧹 Tüm grafikleri temizle", key="clear_charts"):
        session.clear_charts()
        st.rerun()


def render() -> None:
    """Görselleştirme sayfasını çizer."""
    st.title("📊 Görselleştir")
    if not require_dataset("Grafik oluşturma"):
        return

    st.caption(
        "Grafik tipini seçin, iki alanı doldurun ve oluşturun. Ek ayarlar "
        "'Gelişmiş ayarlar' bölümünde."
    )

    with error_boundary("Veri kaynağı"):
        plot_df, data_name, code_prefix = _data_source_section()

    if plot_df is None or plot_df.empty:
        show_warning("Seçili veri kaynağı boş. Filtreleri gevşetin veya gruplamayı güncelleyin.")
        return

    _suggestion_section(plot_df)
    st.divider()

    pending = st.session_state.pop("viz_pending_suggestion", None)
    if pending:
        st.session_state["chart_type"] = pending["chart_type"]
        st.session_state["viz_preset"] = pending["params"]
        show_info(f"Öneri uygulandı: {pending['label']}. Ayarları düzenleyebilirsiniz.")

    section_header("Grafik Oluştur", icon="🎨")
    chart_type = st.selectbox(
        "Grafik tipi",
        list(CHART_TYPES),
        format_func=lambda key: f"{CHART_TYPES[key]['icon']} {CHART_TYPES[key]['label']}",
        key="chart_type",
    )
    st.caption(CHART_TYPES[chart_type]["help"])

    preset = st.session_state.pop("viz_preset", None) or default_params(plot_df, chart_type)

    with error_boundary("Grafik parametreleri"):
        params = _chart_params_form(plot_df, chart_type, preset)

    if len(plot_df) > 100_000 and chart_type in ("scatter", "bubble", "line", "violin"):
        show_warning(
            f"Veri seti büyük ({format_number(len(plot_df))} satır). Bu grafik tipi "
            "örnekleme ile çizilecek; eğilimler korunur ancak tüm noktalar gösterilmez."
        )

    if st.button("📈 Grafiği oluştur", type="primary", key="build_chart"):
        try:
            with st.spinner("Grafik hazırlanıyor..."):
                result = build_chart(
                    plot_df,
                    chart_type,
                    params,
                    source_note=(
                        "group by sonucu"
                        if data_name == "grouped_df"
                        else "filtrelenmiş veri"
                        if data_name == "plot_source"
                        else "working_df"
                    ),
                    data_name=data_name,
                    code_prefix=code_prefix,
                )
        except EdaError as exc:
            show_error(exc)
        else:
            session.add_chart(result.spec)
            show_success("Grafik oluşturuldu ve kod export listesine eklendi.")
            _render_chart_result(result)

    st.divider()
    _render_saved_charts()
