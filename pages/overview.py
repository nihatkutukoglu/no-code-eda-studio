"""Dataset Overview sayfası: genel metrikler, önizleme ve sütun profili."""

from __future__ import annotations

import streamlit as st

from components.messages import (
    disclaimer,
    error_boundary,
    require_dataset,
    section_header,
    show_info,
    show_warning_list,
)
from components.metric_cards import overview_cards
from components.tables import download_csv_button, preview_selector, show_dataframe, show_preview
from core import session
from utils import caching
from utils.helpers import format_number, safe_filename

__all__ = ["render"]


def _render_column_profile() -> None:
    """Sütun özet tablosunu filtreleme seçenekleriyle gösterir."""
    df = session.get_working_df()
    profile = caching.column_profile(df)

    left, right = st.columns([2, 1])
    search = left.text_input(
        "Sütun ara", placeholder="Sütun adının bir parçasını yazın", key="overview_search"
    )
    only_problem = right.toggle(
        "Yalnızca boş değer içerenler", value=False, key="overview_only_missing"
    )

    view = profile
    if search:
        view = view[view["Sütun"].str.contains(search, case=False, na=False)]
    if only_problem:
        view = view[view["Boş"] > 0]

    if view.empty:
        show_info("Filtreye uyan sütun bulunamadı.")
        return

    show_dataframe(view, height=min(560, 60 + 35 * len(view)))
    download_csv_button(
        profile,
        "⬇️ Sütun profilini CSV indir",
        f"{safe_filename(session.dataset_meta().get('filename'))}_sutun_profili.csv",
        key="dl_profile",
    )


def render() -> None:
    """Dataset Overview sayfasını çizer."""
    st.title("🔍 Dataset Overview")
    if not require_dataset("Veri setine genel bakış"):
        return

    df = session.get_working_df()
    meta = session.dataset_meta()
    st.caption(
        f"Kaynak: **{meta.get('filename')}** · "
        f"{format_number(len(session.operation_history()))} temizleme işlemi uygulandı."
    )
    show_warning_list(st.session_state.get("load_warnings") or [], "Dosya okuma notları")

    with error_boundary("Dataset özeti"):
        if caching.is_large_dataset(df):
            show_info(
                f"Veri seti büyük ({format_number(len(df))} satır). Bazı hesaplamalar "
                "biraz daha uzun sürebilir; sonuçlar önbelleğe alınır."
            )
        info = caching.dataset_shape_info(df)
        overview_cards(info)

        st.divider()
        section_header(
            "Dataset Preview", "Ham veriye hızlı bir bakış atın.", icon="👀"
        )
        rows = preview_selector(key="overview_preview_rows")
        show_preview(df, rows)

        st.divider()
        section_header(
            "Sütun Özeti",
            "Her sütunun tipi, boş değer ve tekillik durumu.",
            icon="🧾",
        )
        _render_column_profile()
        disclaimer(
            "Boş % ve Tekil % değerleri toplam satır sayısına göre hesaplanır. "
            "Tekil oranı yüksek sütunlar kimlik (ID) sütunu olabilir."
        )
