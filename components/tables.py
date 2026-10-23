"""Tablo gösterimi ve küçük indirme yardımcıları."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.messages import empty_state
from core.exporter import dataframe_to_csv_bytes
from utils import dataframe_utils as dfu
from utils.helpers import format_number

__all__ = [
    "PREVIEW_ROW_OPTIONS",
    "show_dataframe",
    "preview_selector",
    "show_preview",
    "download_csv_button",
    "show_table_or_empty",
]

#: Dataset preview için satır sayısı seçenekleri.
PREVIEW_ROW_OPTIONS = (10, 20, 50, 100)


def show_dataframe(
    df: pd.DataFrame,
    height: int | None = None,
    max_rows: int | None = None,
    hide_index: bool = True,
) -> None:
    """DataFrame'i Arrow uyumlu güvenli biçimde gösterir.

    Args:
        df: Gösterilecek veri.
        height: Piksel yüksekliği; None ise Streamlit varsayılanı kullanılır
            (``height=None`` doğrudan geçilemez, bu yüzden parametre atlanır).
        max_rows: Gösterilecek maksimum satır.
        hide_index: İndeks sütununu gizle.
    """
    options: dict[str, object] = {"width": "stretch", "hide_index": hide_index}
    if height is not None:
        options["height"] = int(height)
    st.dataframe(dfu.safe_display_df(df, max_rows=max_rows), **options)


def show_table_or_empty(
    df: pd.DataFrame,
    empty_title: str,
    empty_description: str = "",
    icon: str = "✅",
    height: int | None = None,
) -> None:
    """Tablo boşsa boş durum, değilse tabloyu gösterir."""
    if df is None or df.empty:
        empty_state(empty_title, empty_description, icon=icon)
        return
    show_dataframe(df, height=height)


def preview_selector(key: str = "preview_rows", label: str = "Gösterilecek satır") -> int:
    """Satır sayısı seçici (10 / 20 / 50 / 100) gösterir ve seçimi döndürür."""
    return int(
        st.radio(
            label,
            PREVIEW_ROW_OPTIONS,
            index=0,
            horizontal=True,
            key=key,
        )
    )


def show_preview(df: pd.DataFrame, rows: int) -> None:
    """Dataset önizlemesini (ilk N satır) gösterir.

    Args:
        df: Önizlenecek veri.
        rows: Gösterilecek satır sayısı.
    """
    total = len(df)
    st.caption(
        f"İlk {format_number(min(rows, total))} satır gösteriliyor "
        f"(toplam {format_number(total)} satır)."
    )
    show_dataframe(df.head(rows), height=min(420, 60 + 28 * min(rows, total)), hide_index=False)


def download_csv_button(
    df: pd.DataFrame,
    label: str,
    filename: str,
    key: str | None = None,
    help_text: str | None = None,
) -> None:
    """Tabloyu CSV olarak indirme butonu ekler."""
    if df is None or df.empty:
        return
    st.download_button(
        label,
        data=dataframe_to_csv_bytes(df),
        file_name=filename,
        mime="text/csv",
        key=key,
        help=help_text,
    )
