"""ML Hazırlığı sayfası.

Model **eğitmez**: hedef sütuna göre dataset'in modellemeye uygunluğunu
analiz eder ve train/test ayrımı + scikit-learn ön işleme pipeline kodunu
üretir. Üretilen kod Python/Notebook export'una eklenir.
"""

from __future__ import annotations

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
from components.tables import show_dataframe
from core import session
from core.ml_preparation import (
    CATEGORICAL_ENCODERS,
    CATEGORICAL_IMPUTERS,
    NUMERIC_IMPUTERS,
    SCALERS,
    analyze_readiness,
    full_ml_code,
)
from utils import dataframe_utils as dfu
from utils.helpers import format_number, format_percent

__all__ = ["render"]


def render() -> None:
    """ML Hazırlığı sayfasını çizer."""
    st.title("🤖 ML Hazırlığı")
    if not require_dataset("ML hazırlığı"):
        return

    df = session.get_working_df()
    st.caption(
        "Hedef sütununuzu seçin; uygulama veri setini modellemeye uygunluk "
        "açısından kontrol eder ve hazır scikit-learn kodu üretir. "
        "**Model eğitimi yapılmaz.**"
    )

    all_columns = [str(c) for c in df.columns]
    controls = st.columns([2, 3])
    target = controls[0].selectbox("Hedef sütun (target)", all_columns, key="ml_target")
    features = controls[1].multiselect(
        "Özellikler (features)",
        [c for c in all_columns if c != target],
        default=[c for c in all_columns if c != target],
        key="ml_features",
    )

    if not features:
        show_info("En az bir özellik sütunu seçin.")
        return

    with error_boundary("ML hazırlık analizi"):
        report = analyze_readiness(df, target, features)

        section_header(
            "Dataset Değerlendirmesi", "Hedef tipi ve özellik dağılımı.", icon="🧭"
        )
        metric_row(
            [
                ("Problem tipi", report.task_label),
                ("Satır", format_number(report.n_rows)),
                ("Özellik", format_number(report.n_features)),
                ("Sayısal", format_number(len(report.numeric_features))),
                ("Kategorik", format_number(len(report.categorical_features))),
            ],
            columns=5,
        )

        if report.blockers:
            for message in report.blockers:
                st.error(message, icon="🚫")
        elif report.is_ready:
            show_success(
                "Dataset modellemeye uygun görünüyor. Aşağıdaki uyarıları gözden geçirin."
            )
        for message in report.warnings:
            show_warning(message)

        if report.target_classes:
            with st.expander("Hedef sınıf dağılımı", expanded=False):
                import pandas as pd

                distribution = pd.DataFrame(
                    {
                        "Sınıf": list(report.target_classes),
                        "Adet": list(report.target_classes.values()),
                    }
                )
                distribution["Oran %"] = (
                    distribution["Adet"] / distribution["Adet"].sum() * 100
                ).round(2)
                show_dataframe(distribution)
                st.caption(
                    f"En büyük sınıfın payı: {format_percent(report.imbalance_ratio * 100)}"
                )

        findings = report.summary_table()
        section_header("Bulgular", "Modellemede sorun çıkarabilecek sütunlar.", icon="🔍")
        if findings.empty:
            show_success("Özelliklerde belirgin bir sorun bulunamadı.")
        else:
            show_dataframe(findings)
        for note in report.notes:
            st.caption(f"ℹ️ {note}")

        st.divider()
        section_header(
            "Train / Test Ayrımı", "Kod üretimi için ayarları belirleyin.", icon="✂️"
        )
        split_columns = st.columns(3)
        test_size = split_columns[0].slider(
            "Test oranı", 0.1, 0.5, 0.2, 0.05, key="ml_test_size"
        )
        random_state = split_columns[1].number_input(
            "random_state", min_value=0, max_value=9999, value=42, key="ml_random_state"
        )
        can_stratify = report.task_type in (
            "binary_classification",
            "multiclass_classification",
        )
        stratify = split_columns[2].toggle(
            "Stratify (sınıf oranını koru)",
            value=can_stratify,
            disabled=not can_stratify,
            key="ml_stratify",
            help="Yalnızca sınıflandırma problemlerinde kullanılabilir.",
        )

        drop_suggestions = report.recommended_drops()
        drop_columns = st.multiselect(
            "Modelden çıkarılacak sütunlar",
            [c for c in features],
            default=[c for c in drop_suggestions if c in features],
            key="ml_drops",
            help="Sabit, kimlik ve sızıntı şüphesi olan sütunlar varsayılan olarak seçilidir.",
        )

        selected_features = [c for c in features if c not in drop_columns]
        if not selected_features:
            show_warning("Tüm özellikleri çıkardınız; en az bir özellik kalmalı.")
            return

        numeric_features = [c for c in selected_features if c in dfu.numeric_columns(df)]
        categorical_features = [
            c
            for c in selected_features
            if c in dfu.categorical_columns(df) and c not in numeric_features
        ]
        ignored = [
            c
            for c in selected_features
            if c not in numeric_features and c not in categorical_features
        ]
        if ignored:
            st.caption(
                "Pipeline'a eklenmeyen sütunlar (ör. tarih): " + ", ".join(ignored)
            )

        st.divider()
        section_header(
            "Ön İşleme Pipeline'ı",
            "Sayısal ve kategorik sütunlar için ayrı adımlar tanımlanır.",
            icon="⚙️",
        )
        pipeline_columns = st.columns(2)
        with pipeline_columns[0]:
            st.markdown("**Sayısal sütunlar**")
            numeric_imputer = st.selectbox(
                "Boş değer doldurma",
                list(NUMERIC_IMPUTERS),
                index=list(NUMERIC_IMPUTERS).index("median"),
                format_func=lambda key: NUMERIC_IMPUTERS[key],
                key="ml_num_imputer",
            )
            numeric_scaler = st.selectbox(
                "Ölçekleme",
                list(SCALERS),
                index=list(SCALERS).index("standard"),
                format_func=lambda key: SCALERS[key],
                key="ml_scaler",
            )
        with pipeline_columns[1]:
            st.markdown("**Kategorik sütunlar**")
            categorical_imputer = st.selectbox(
                "Boş değer doldurma",
                list(CATEGORICAL_IMPUTERS),
                index=list(CATEGORICAL_IMPUTERS).index("most_frequent"),
                format_func=lambda key: CATEGORICAL_IMPUTERS[key],
                key="ml_cat_imputer",
            )
            categorical_encoder = st.selectbox(
                "Encoding",
                list(CATEGORICAL_ENCODERS),
                format_func=lambda key: CATEGORICAL_ENCODERS[key],
                key="ml_encoder",
            )

        code = full_ml_code(
            target=target,
            features=selected_features,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            test_size=float(test_size),
            random_state=int(random_state),
            stratify=bool(stratify),
            numeric_imputer=numeric_imputer,
            numeric_scaler=numeric_scaler,
            categorical_imputer=categorical_imputer,
            categorical_encoder=categorical_encoder,
            drop_columns=drop_columns,
        )

        st.divider()
        section_header(
            "Üretilen scikit-learn Kodu",
            "Bu kod Python script ve Notebook export'una eklenir.",
            icon="🐍",
        )
        st.code(code, language="python")

        buttons = st.columns([1, 1, 2])
        with buttons[0]:
            if st.button("💾 Export'a ekle", type="primary", key="ml_save"):
                st.session_state["ml_prep_code"] = code
                st.session_state["ml_prep_meta"] = {
                    "target": target,
                    "features": selected_features,
                    "task_type": report.task_type,
                }
                show_success(
                    "ML hazırlık kodu kaydedildi; Python Kodu ve Dışa Aktar "
                    "sayfalarında yer alacak."
                )
        with buttons[1]:
            st.download_button(
                "⬇️ .py indir",
                data=code.encode("utf-8"),
                file_name="ml_preparation.py",
                mime="text/x-python",
                key="ml_download",
            )
        with buttons[2]:
            if st.session_state.get("ml_prep_code"):
                st.caption("✅ Bu kod export'lara dahil edilecek.")

        disclaimer(
            "Bu sayfa model eğitmez ve performans garantisi vermez. Bulgular kural "
            "tabanlıdır; nihai kararlar veri bağlamına göre sizin tarafınızdan verilmelidir."
        )
