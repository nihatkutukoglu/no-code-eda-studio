"""Dışa Aktar sayfası: veri, kod, notebook, grafikler, analiz sonuçları ve rapor."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.messages import (
    disclaimer,
    error_boundary,
    require_dataset,
    section_header,
    show_info,
    show_warning,
)
from components.metric_cards import metric_row
from core import session
from core.chart_builder import build_chart
from core.correlations import correlation_matrix
from core.exporter import (
    DATA_FORMATS,
    dataframe_to_bytes,
    export_filename,
    figure_to_html_bytes,
    insights_to_json_bytes,
    notebook_bytes,
    notebook_filename,
    report_filename,
    report_to_bytes,
    script_filename,
    script_to_bytes,
)
from core.insights import insights_to_dataframe
from core.operations import history_table
from utils import caching, dataframe_utils as dfu
from utils.helpers import format_bytes, format_number, safe_filename
from utils.validators import EdaError

__all__ = ["render"]

_CSV_SEPARATORS = {"Virgül ( , )": ",", "Noktalı virgül ( ; )": ";", "Tab": "\t"}


def _dataset_section() -> None:
    """Temizlenmiş veri setini indirme bölümü."""
    df = session.get_working_df()
    base = session.dataset_meta().get("filename")

    section_header(
        "Temizlenmiş Veri Seti",
        "Uygulanan tüm temizleme ve dönüşüm işlemlerini içeren güncel veri.",
        icon="🗃️",
    )
    metric_row(
        [
            ("Satır", format_number(len(df))),
            ("Sütun", format_number(df.shape[1])),
            ("Uygulanan işlem", format_number(len(session.operation_history()))),
        ]
    )

    columns = st.columns([1, 1])
    with columns[0]:
        file_format = st.selectbox(
            "Biçim",
            list(DATA_FORMATS),
            format_func=lambda key: DATA_FORMATS[key][0],
            key="exp_format",
        )
    with columns[1]:
        separator = ","
        extra_sheets = None
        if file_format == "csv":
            separator = _CSV_SEPARATORS[
                st.selectbox("CSV ayracı", list(_CSV_SEPARATORS), key="exp_sep")
            ]
        elif file_format == "xlsx":
            if st.toggle(
                "İşlem geçmişi ve bulguları ek sayfa olarak ekle",
                value=True,
                key="exp_extra",
            ):
                extra_sheets = {
                    "Islem Gecmisi": history_table(session.operation_history()),
                    "Bulgular": insights_to_dataframe(caching.generate_insights(df)),
                }

    with error_boundary("Veri export"):
        label, extension, mime = DATA_FORMATS[file_format]
        data = dataframe_to_bytes(
            df, file_format, separator=separator, extra_sheets=extra_sheets
        )
        st.download_button(
            f"⬇️ {label} indir ({format_bytes(len(data))})",
            data=data,
            file_name=export_filename(base, "cleaned", extension),
            mime=mime,
            key=f"dl_data_{file_format}",
            type="primary",
        )
    if file_format == "csv":
        st.caption("Encoding: UTF-8 (BOM) — Excel'de Türkçe karakterler bozulmaz.")
    elif file_format == "parquet":
        st.caption("Parquet: sütun tipleri korunur, dosya boyutu küçüktür.")


def _code_section() -> None:
    """Python script ve Notebook export bölümü."""
    meta = session.dataset_meta()
    history = session.operation_history()
    charts = session.generated_charts()
    df = session.get_working_df()
    ml_code = st.session_state.get("ml_prep_code")

    section_header(
        "Analizin Kodu",
        "Yaptığınız işlemleri çalıştırılabilir Python script'i veya Jupyter "
        "Notebook olarak indirin.",
        icon="🐍",
    )
    if not history and not charts:
        show_info(
            "Henüz veri işlemi veya grafik yok. Yine de iskelet bir script/notebook "
            "indirebilirsiniz."
        )

    options = st.columns(3)
    include_cleaning = options[0].toggle("Veri işlemleri", value=True, key="exp_inc_clean")
    include_charts = options[1].toggle("Grafikler", value=True, key="exp_inc_charts")
    include_ml = options[2].toggle(
        "ML hazırlığı",
        value=bool(ml_code),
        disabled=not ml_code,
        key="exp_inc_ml",
        help="ML Hazırlığı sayfasından kod eklediğinizde aktif olur.",
    )
    selected_ml = ml_code if (include_ml and ml_code) else None

    buttons = st.columns(2)
    with buttons[0], error_boundary("Script export"):
        script = script_to_bytes(
            meta,
            history if include_cleaning else [],
            charts if include_charts else [],
            include_cleaning,
            include_charts,
            ml_code=selected_ml,
        )
        st.download_button(
            f"⬇️ Python Script (.py) — {format_bytes(len(script))}",
            data=script,
            file_name=script_filename(meta.get("filename")),
            mime="text/x-python",
            key="dl_script",
            width="stretch",
            type="primary",
        )
    with buttons[1], error_boundary("Notebook export"):
        include_notes = st.toggle(
            "Notebook'a kural tabanlı notları ekle", value=True, key="exp_notes"
        )
        insights = caching.generate_insights(df) if include_notes else []
        notebook = notebook_bytes(
            meta,
            history if include_cleaning else [],
            charts if include_charts else [],
            insights,
            numeric_columns=dfu.numeric_columns(df)[:10],
            ml_code=selected_ml,
        )
        st.download_button(
            f"⬇️ Jupyter Notebook (.ipynb) — {format_bytes(len(notebook))}",
            data=notebook,
            file_name=notebook_filename(meta.get("filename")),
            mime="application/x-ipynb+json",
            key="dl_notebook",
            width="stretch",
            type="primary",
        )

    st.caption(
        "Notebook bölümleri: Imports → Data Loading → Dataset Overview → "
        "Data Cleaning / Transformations → Statistical Analysis → Relationships "
        "→ Outlier Analysis → Visualizations → ML Preparation → Insights."
    )
    disclaimer(
        "Kullanıcı verisi notebook içine gömülmez; yalnızca işlemlerin kodu yazılır. "
        "Script'i çalıştırmak için kaynak dosyanın aynı klasörde olması gerekir."
    )


def _report_section() -> None:
    """HTML EDA raporu bölümü."""
    df = session.get_working_df()
    meta = session.dataset_meta()
    charts = session.generated_charts()

    section_header(
        "HTML EDA Raporu",
        "Tek dosyalık, paylaşılabilir rapor: özet, kalite, istatistik, "
        "korelasyon, bulgular ve grafikler.",
        icon="📄",
    )
    include_charts = st.toggle(
        "Kaydedilen grafikleri rapora ekle",
        value=bool(charts),
        disabled=not charts,
        key="exp_report_charts",
    )
    with error_boundary("Rapor üretimi"):
        with st.spinner("Rapor hazırlanıyor..."):
            data = report_to_bytes(
                df,
                meta=meta,
                charts=charts if include_charts else [],
                history=session.operation_history(),
                insights=caching.generate_insights(df),
                include_charts=include_charts,
            )
        st.download_button(
            f"⬇️ HTML raporu indir ({format_bytes(len(data))})",
            data=data,
            file_name=report_filename(meta.get("filename")),
            mime="text/html",
            key="dl_report",
            type="primary",
        )
    st.caption(
        "Rapor tarayıcıda açılır; grafikler Plotly CDN üzerinden yüklenir. "
        "PDF'e çevirmek için tarayıcıdan 'Yazdır → PDF olarak kaydet' kullanabilirsiniz."
    )


def _analysis_results_section() -> None:
    """Analiz çıktılarının export bölümü."""
    df, _ = session.get_analysis_df()
    section_header(
        "Analiz Sonuçları",
        "Korelasyon matrisi, bulgular ve işlem geçmişini ayrı dosyalar olarak indirin.",
        icon="📑",
    )
    base = session.dataset_meta().get("filename")
    columns = st.columns(3)

    with columns[0], error_boundary("Korelasyon export"):
        numeric = dfu.numeric_columns(df)
        if len(numeric) >= 2:
            matrix = correlation_matrix(df, numeric[:15], "pearson").round(6)
            st.download_button(
                "⬇️ Korelasyon matrisi (CSV)",
                data=matrix.reset_index(names="Sütun").to_csv(index=False).encode("utf-8-sig"),
                file_name=export_filename(base, "korelasyon", "csv"),
                mime="text/csv",
                key="dl_corr_csv",
                width="stretch",
            )
        else:
            show_info("Korelasyon için en az 2 sayısal sütun gerekir.")

    with columns[1], error_boundary("Bulgular export"):
        insights = caching.generate_insights(session.get_working_df())
        table = insights_to_dataframe(insights)
        st.download_button(
            "⬇️ Bulgular (CSV)",
            data=table.to_csv(index=False).encode("utf-8-sig"),
            file_name=export_filename(base, "bulgular", "csv"),
            mime="text/csv",
            key="dl_insights_csv",
            width="stretch",
        )
        st.download_button(
            "⬇️ Bulgular (JSON)",
            data=insights_to_json_bytes(insights),
            file_name=export_filename(base, "bulgular", "json"),
            mime="application/json",
            key="dl_insights_json",
            width="stretch",
        )

    with columns[2], error_boundary("Geçmiş export"):
        history = history_table(session.operation_history())
        if history.empty:
            show_info("Henüz işlem geçmişi yok.")
        else:
            st.download_button(
                "⬇️ İşlem geçmişi (CSV)",
                data=history.to_csv(index=False).encode("utf-8-sig"),
                file_name=export_filename(base, "islem_gecmisi", "csv"),
                mime="text/csv",
                key="dl_history_csv",
                width="stretch",
            )

    grouped = st.session_state.get("viz_grouped")
    if isinstance(grouped, pd.DataFrame) and not grouped.empty:
        st.download_button(
            "⬇️ Son gruplama sonucu (CSV)",
            data=grouped.to_csv(index=False).encode("utf-8-sig"),
            file_name=export_filename(base, "gruplama", "csv"),
            mime="text/csv",
            key="dl_group_csv",
        )


def _charts_section() -> None:
    """Grafikleri HTML olarak indirme bölümü."""
    charts = session.generated_charts()
    section_header(
        "Grafikler", "Kaydedilen grafikleri interaktif HTML olarak indirin.", icon="📈"
    )
    if not charts:
        show_info("Kaydedilmiş grafik yok. Görselleştir sayfasından ekleyebilirsiniz.")
        return

    df, _ = session.get_analysis_df()
    for index, chart in enumerate(charts, start=1):
        with st.expander(f"{index}. {chart.get('title')} · {chart.get('label')}"):
            st.code(chart.get("python_code", ""), language="python")
            try:
                result = build_chart(
                    df,
                    str(chart.get("type")),
                    dict(chart.get("params") or {}),
                    source_note=chart.get("source_note"),
                )
            except EdaError as exc:
                show_warning(
                    "Grafik güncel veriyle yeniden oluşturulamadı "
                    f"(veri değişmiş olabilir): {exc}"
                )
                continue
            st.download_button(
                "⬇️ HTML indir",
                data=figure_to_html_bytes(result.figure),
                file_name=f"{safe_filename(chart.get('title'))}.html",
                mime="text/html",
                key=f"dl_html_{chart.get('id')}",
            )


def render() -> None:
    """Dışa Aktar sayfasını çizer."""
    st.title("📤 Dışa Aktar")
    if not require_dataset("Dışa aktarma"):
        return

    st.caption(
        "Tüm çıktılar tarayıcınız üzerinden indirilir; hiçbir veri sunucuya "
        "kaydedilmez veya üçüncü bir servise gönderilmez."
    )

    tabs = st.tabs(
        ["🗃️ Veri", "🐍 Kod & Notebook", "📄 Rapor", "📑 Analiz Sonuçları", "📈 Grafikler"]
    )
    with tabs[0], error_boundary("Veri export"):
        _dataset_section()
    with tabs[1], error_boundary("Kod export"):
        _code_section()
    with tabs[2], error_boundary("Rapor export"):
        _report_section()
    with tabs[3], error_boundary("Analiz export"):
        _analysis_results_section()
    with tabs[4], error_boundary("Grafik export"):
        _charts_section()
