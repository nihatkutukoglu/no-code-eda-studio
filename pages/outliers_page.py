"""Aykırı Değer Analizi sayfası (IQR yöntemi)."""

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
    show_warning,
)
from components.metric_cards import metric_row
from components.tables import download_csv_button, show_dataframe
from core import session
from core.chart_builder import PLOTLY_TEMPLATE
from core.code_generator import build_operation_code
from core.outliers import (
    DEFAULT_IQR_FACTOR,
    DEFAULT_ZSCORE_THRESHOLD,
    OUTLIER_METHODS,
    detect_outliers,
)
from utils import caching, dataframe_utils as dfu
from utils.helpers import format_number, format_percent, safe_filename

__all__ = ["render"]

_OUTLIER_WARNING = (
    "Bir değerin istatistiksel olarak aykırı olması hatalı olduğu anlamına gelmez. "
    "Silmeden önce verinin iş bağlamını değerlendirin."
)


def _delete_outliers(column: str, factor: float, method: str = "iqr") -> None:
    """Aykırı satırları kalıcı olarak siler (operation history'ye kaydeder)."""
    with error_boundary("Aykırı değer silme"):
        operation = session.apply_and_record(
            "drop_outlier_rows",
            {"column": column, "factor": float(factor), "method": method},
        )
        st.session_state["outlier_message"] = operation.get("description")
        st.rerun()


def render() -> None:
    """Aykırı Değer Analizi sayfasını çizer."""
    st.title("🎯 Aykırı Değer Analizi")
    if not require_dataset("Aykırı değer analizi"):
        return

    message = st.session_state.pop("outlier_message", None)
    if message:
        show_success(message)

    df, filtered = session.get_analysis_df()
    if filtered:
        st.caption(f"ℹ️ Filtre aktif: {format_number(len(df))} satır analiz ediliyor.")

    numeric = dfu.numeric_columns(df)
    if not numeric:
        show_info(
            "Veri setinde sayısal sütun yok. Aykırı değer analizi için sayısal bir "
            "sütun gerekir."
        )
        return

    controls = st.columns([2, 1, 1])
    column = controls[0].selectbox("Sayısal sütun", numeric, key="out_col")
    method = controls[1].selectbox(
        "Yöntem",
        list(OUTLIER_METHODS),
        format_func=lambda key: OUTLIER_METHODS[key],
        key="out_method",
    )
    if method == "zscore":
        factor = controls[2].slider(
            "|z| eşiği", 1.5, 5.0, float(DEFAULT_ZSCORE_THRESHOLD), 0.1, key="out_z"
        )
        st.caption(
            "Z-Score: sınırlar ortalama ± eşik × standart sapma ile hesaplanır. "
            "Aşırı değerler standart sapmayı büyüttüğü için küçük veri setlerinde "
            "IQR daha güvenilir olabilir."
        )
    else:
        factor = controls[2].slider(
            "IQR çarpanı (k)", 1.0, 3.0, float(DEFAULT_IQR_FACTOR), 0.1, key="out_factor"
        )
        st.caption(
            "IQR: sınırlar Q1 - k×IQR ve Q3 + k×IQR formülleriyle hesaplanır "
            "(yaygın k değeri 1.5)."
        )

    with error_boundary("Aykırı değer analizi"):
        report = detect_outliers(df, column, factor, method)

        metric_row(
            [
                ("Q1", format_number(report.q1, 3)),
                ("Q3", format_number(report.q3, 3)),
                ("IQR", format_number(report.iqr, 3)),
                ("Alt sınır", format_number(report.lower_bound, 3)),
                ("Üst sınır", format_number(report.upper_bound, 3)),
            ],
            columns=5,
        )
        st.caption(f"Yöntem: {report.method_label}")
        metric_row(
            [
                ("Aykırı değer sayısı", format_number(report.count)),
                ("Aykırı oranı", format_percent(report.ratio)),
                ("Alt sınır altında", format_number(report.lower_count)),
                ("Üst sınır üstünde", format_number(report.upper_count)),
            ],
            columns=4,
        )

        charts = st.columns(2)
        with charts[0]:
            st.plotly_chart(
                px.box(
                    df, y=column, points="outliers", title=f"{column} kutu grafiği",
                    template=PLOTLY_TEMPLATE,
                ),
                width="stretch",
            )
        with charts[1]:
            figure = px.histogram(
                df, x=column, nbins=40, title=f"{column} histogramı",
                template=PLOTLY_TEMPLATE,
            )
            figure.add_vline(x=report.lower_bound, line_dash="dash", line_color="#c62828")
            figure.add_vline(x=report.upper_bound, line_dash="dash", line_color="#c62828")
            st.plotly_chart(figure, width="stretch")

        st.divider()
        section_header("Aykırı Kayıtlar", "Sınırların dışında kalan satırlar.", icon="🔍")
        if report.count == 0:
            show_success("Bu sütunda IQR yöntemine göre aykırı değer bulunamadı.")
        else:
            outlier_rows = df.loc[report.mask]
            preview_count = st.slider(
                "Gösterilecek satır", 5, min(200, len(outlier_rows)),
                min(20, len(outlier_rows)), key="out_preview",
            )
            show_dataframe(outlier_rows.head(preview_count), hide_index=False)
            download_csv_button(
                outlier_rows,
                "⬇️ Aykırı kayıtları CSV indir",
                f"{safe_filename(column)}_aykiri_kayitlar.csv",
                key="dl_outliers",
            )

            st.divider()
            section_header(
                "İşlem", "Aykırı satırları kalıcı olarak silebilirsiniz.", icon="🗑️"
            )
            show_warning(
                f"Bu işlem **{format_number(report.count)}** satırı veri setinden "
                "siler ve işlem geçmişine kaydedilir (geri alınabilir)."
            )
            with st.expander("🐍 Üretilecek Python kodu"):
                st.code(
                    build_operation_code(
                        "drop_outlier_rows",
                        {"column": column, "factor": float(factor), "method": method},
                    ),
                    language="python",
                )
            if filtered:
                show_info(
                    "Filtre aktifken silme işlemi tüm veri setine (filtresiz) "
                    "uygulanır; sınırlar filtrelenmiş veriden hesaplanmıştır."
                )
            if st.button("🗑️ Aykırı satırları sil", type="primary", key="btn_del_outliers"):
                _delete_outliers(column, float(factor), method)

        disclaimer(_OUTLIER_WARNING, icon="⚠️")

        st.divider()
        with st.expander("📋 Tüm sayısal sütunların aykırı değer özeti"):
            table = caching.outlier_summary_table(df, float(factor))
            show_dataframe(table)
            download_csv_button(
                table, "⬇️ Özeti indir", "aykiri_deger_ozeti.csv", key="dl_out_summary"
            )
