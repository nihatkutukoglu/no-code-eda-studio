"""Python Kodu sayfası.

Arayüzde yapılan her işlemin gerçek Python/Pandas karşılığını gösterir.
Sekmeler: Tüm akış, Temizleme, Dönüşümler, Analiz, Görselleştirme, ML Hazırlığı.
"""

from __future__ import annotations

import streamlit as st

from components.messages import (
    disclaimer,
    error_boundary,
    require_dataset,
    section_header,
    show_info,
)
from components.metric_cards import metric_row
from core import session
from core.code_generator import (
    CODE_SECTIONS,
    action_category,
    filter_history,
    generate_charts_section,
    generate_full_script,
    generate_imports,
    generate_loading_section,
    generate_sectioned_cleaning,
)
from core.correlations import correlation_code
from core.exporter import script_filename
from utils import dataframe_utils as dfu
from utils.helpers import format_number

__all__ = ["render"]


def _analysis_code(df) -> str:
    """İstatistik ve korelasyon analizlerinin kodunu üretir."""
    numeric = dfu.numeric_columns(df)
    lines = [
        "# Genel özet",
        "df.info()",
        "df.describe().T",
    ]
    if numeric:
        lines += [
            "",
            "# Boş değer dağılımı",
            "df.isna().sum().sort_values(ascending=False)",
        ]
    if len(numeric) >= 2:
        lines += ["", correlation_code(numeric[: min(10, len(numeric))], "pearson")]
    return "\n".join(lines)


def _show_code(code: str, label: str, key: str, note: str | None = None) -> None:
    """Kod bloğunu ve indirme butonunu gösterir."""
    if not code.strip():
        show_info("Bu bölümde gösterilecek kod yok.")
        return
    if note:
        st.caption(note)
    st.code(code, language="python")
    st.download_button(
        f"⬇️ {label} indir (.py)",
        data=code.encode("utf-8"),
        file_name=script_filename(session.dataset_meta().get("filename")),
        mime="text/x-python",
        key=key,
    )


def render() -> None:
    """Python Kodu sayfasını çizer."""
    st.title("🐍 Python Kodu")
    if not require_dataset("Kod görüntüleme"):
        return

    meta = session.dataset_meta()
    history = session.operation_history()
    charts = session.generated_charts()
    df = session.get_working_df()
    ml_code = st.session_state.get("ml_prep_code")

    st.caption(
        "Bu kod arayüzde yaptığınız işlemlerden otomatik üretilir. Sütun adları "
        "güvenli biçimde kaçışlanır ve veri değerleri koda gömülmez."
    )
    metric_row(
        [
            ("Veri işlemi", format_number(len(history))),
            ("Grafik", format_number(len(charts))),
            ("ML hazırlığı", "Eklendi" if ml_code else "Yok"),
            ("Kaynak dosya", str(meta.get("filename", "-"))),
        ],
        columns=4,
    )

    with_comments = st.toggle("Açıklama satırlarını ekle", value=True, key="code_comments")
    header = generate_imports(include_charts=bool(charts)) + "\n\n" + generate_loading_section(meta)

    tabs = st.tabs(
        [
            "📜 Tüm Akış",
            "🧼 Temizleme",
            "🧬 Dönüşümler",
            "📐 Analiz",
            "📊 Görselleştirme",
            "🤖 ML Hazırlığı",
        ]
    )

    with tabs[0], error_boundary("Kod üretimi"):
        code = generate_full_script(
            meta,
            history,
            charts,
            analysis_code=None,
            ml_code=ml_code,
        )
        _show_code(code, "Tüm kodu", "dl_code_all")

    with tabs[1], error_boundary("Kod üretimi"):
        cleaning = filter_history(history, ["cleaning"])
        _show_code(
            header + "\n\n" + generate_sectioned_cleaning(cleaning, with_comments)
            if cleaning
            else "",
            "Temizleme kodunu",
            "dl_code_clean",
            "Yalnızca veri temizleme adımları.",
        )

    with tabs[2], error_boundary("Kod üretimi"):
        transforms = filter_history(history, ["transformation", "feature", "arrangement"])
        _show_code(
            header + "\n\n" + generate_sectioned_cleaning(transforms, with_comments)
            if transforms
            else "",
            "Dönüşüm kodunu",
            "dl_code_transform",
            "Dönüşüm, feature engineering ve düzen adımları. Not: Bu alt küme "
            "temizleme adımlarını içermez; tam akış için 'Tüm Akış' sekmesini kullanın.",
        )

    with tabs[3], error_boundary("Kod üretimi"):
        _show_code(
            header + "\n\n" + _analysis_code(df),
            "Analiz kodunu",
            "dl_code_analysis",
            "İstatistik özetleri ve korelasyon analizi.",
        )

    with tabs[4], error_boundary("Kod üretimi"):
        _show_code(
            generate_imports(include_charts=True)
            + "\n\n"
            + generate_loading_section(meta)
            + "\n\n"
            + generate_charts_section(charts, with_comments)
            if charts
            else "",
            "Grafik kodunu",
            "dl_code_charts",
            "Oluşturduğunuz grafiklerin Plotly kodu.",
        )

    with tabs[5], error_boundary("Kod üretimi"):
        if ml_code:
            _show_code(
                ml_code,
                "ML kodunu",
                "dl_code_ml",
                "train_test_split ve scikit-learn ön işleme pipeline'ı.",
            )
        else:
            show_info(
                "ML hazırlık kodu yok. **ML Hazırlığı** sayfasından hedef sütunu "
                "seçip 'Export'a ekle' butonuna basın."
            )

    st.divider()
    section_header("Adım adım kodlar", "Her işlemin kendi kod parçası.", icon="🧩")
    if history:
        for index, operation in enumerate(history, start=1):
            with st.expander(f"{index}. {operation.get('description')}"):
                category = action_category(operation.get("action", ""))
                st.caption(f"Bölüm: {CODE_SECTIONS.get(category, '-')}")
                st.code(operation.get("python_code", ""), language="python")
    else:
        show_info("Henüz veri işlemi uygulanmadı.")

    disclaimer(
        "Kod, dosyanın uygulamaya yüklenen adıyla okunacak şekilde üretilir "
        "(ör. pd.read_csv('dosya.csv')). Script'i çalıştırmadan önce dosyanın "
        "aynı klasörde olduğundan emin olun."
    )
