"""Streamlit Session State yönetimi.

Tek kural: **veriye yalnızca bu modül üzerinden dokunulur.** Sayfalar
``st.session_state`` anahtarlarını doğrudan değiştirmez; böylece
``original_df`` korunur, ``operation_history`` tutarlı kalır ve undo/replay
mantığı tek yerde toplanır.

Session State anahtarları:
    original_df         Yüklenen dosyanın hiç değiştirilmemiş kopyası.
    working_df          Kullanıcı işlemlerinin uygulandığı DataFrame.
    uploaded_filename   Orijinal dosya adı.
    dataset_meta        Dosya okuma parametreleri (kod üretimi için).
    operation_history   Uygulanan operasyonların sözlük listesi.
    generated_charts    Oluşturulan grafik tanımları (kod dahil).
    active_filters      Analiz/grafik ekranlarında kullanılan geçici filtreler.
    filters_enabled     Geçici filtrelerin aktif olup olmadığı.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from core import operations
from core.filters import apply_filters
from core.loader import LoadResult
from utils import dataframe_utils as dfu
from utils.validators import EdaError

__all__ = [
    "SESSION_DEFAULTS",
    "init_session_state",
    "has_dataset",
    "get_original_df",
    "get_working_df",
    "get_analysis_df",
    "dataset_meta",
    "set_dataset",
    "clear_dataset",
    "apply_and_record",
    "undo_last_operation",
    "reset_dataset",
    "operation_history",
    "generated_charts",
    "add_chart",
    "remove_chart",
    "clear_charts",
    "set_active_filters",
    "get_active_filters",
    "filters_active",
    "filters_combine",
]

#: Session State varsayılan değerleri.
SESSION_DEFAULTS: dict[str, Any] = {
    "original_df": None,
    "working_df": None,
    "uploaded_filename": None,
    "dataset_meta": {},
    "operation_history": [],
    "generated_charts": [],
    "active_filters": [],
    "filters_enabled": False,
    "filters_combine": "and",
    "load_warnings": [],
    "replay_errors": [],
}


def init_session_state() -> None:
    """Eksik session anahtarlarını varsayılan değerlerle oluşturur.

    Her sayfa çalıştırmasında güvenle çağrılabilir; var olan değerleri bozmaz.
    """
    for key, default in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = (
                list(default)
                if isinstance(default, list)
                else dict(default)
                if isinstance(default, dict)
                else default
            )


# ---------------------------------------------------------------------------
# Dataset erişimi
# ---------------------------------------------------------------------------


def has_dataset() -> bool:
    """Yüklü ve kullanılabilir bir veri seti olup olmadığını döndürür."""
    df = st.session_state.get("working_df")
    return isinstance(df, pd.DataFrame) and df.shape[1] > 0


def get_original_df() -> pd.DataFrame | None:
    """Orijinal (değiştirilmemiş) DataFrame'i döndürür."""
    return st.session_state.get("original_df")


def get_working_df() -> pd.DataFrame | None:
    """Üzerinde çalışılan DataFrame'i döndürür."""
    return st.session_state.get("working_df")


def get_analysis_df() -> tuple[pd.DataFrame | None, bool]:
    """Analiz/grafik ekranları için DataFrame döndürür.

    Geçici filtreler aktifse filtrelenmiş **kopya** döner; ``working_df``
    değişmez.

    Returns:
        ``(dataframe, filtre_uygulandı_mı)``
    """
    df = get_working_df()
    if df is None:
        return None, False
    filters = get_active_filters()
    if not filters or not st.session_state.get("filters_enabled"):
        return df, False
    try:
        return apply_filters(df, filters, filters_combine()), True
    except EdaError:
        return df, False


def dataset_meta() -> dict[str, Any]:
    """Kod/notebook üretimi için dosya okuma bilgilerini döndürür."""
    meta = dict(st.session_state.get("dataset_meta") or {})
    meta.setdefault("filename", st.session_state.get("uploaded_filename") or "dataset.csv")
    meta.setdefault("file_type", "csv")
    return meta


def set_dataset(result: LoadResult) -> None:
    """Yeni veri setini session'a yerleştirir ve geçmişi sıfırlar.

    ``original_df`` yalnızca burada yazılır ve sonradan hiçbir temizleme
    işleminde değiştirilmez.
    """
    # Son savunma hattı: tekrarlı sütun adları uygulamaya hiç girmesin.
    frame = dfu.deduplicate_columns(result.df)
    st.session_state["original_df"] = frame.copy(deep=True)
    st.session_state["working_df"] = frame.copy(deep=True)
    st.session_state["uploaded_filename"] = result.filename
    st.session_state["dataset_meta"] = {
        "filename": result.filename,
        "file_type": result.file_type,
        "separator": result.separator,
        "encoding": result.encoding,
        "sheet_name": result.sheet_name,
    }
    st.session_state["operation_history"] = []
    st.session_state["generated_charts"] = []
    st.session_state["active_filters"] = []
    st.session_state["filters_enabled"] = False
    st.session_state["load_warnings"] = list(result.warnings)
    st.session_state["replay_errors"] = []


def clear_dataset() -> None:
    """Tüm veri ve geçmişi temizler (yeni dosya yüklemeye hazırlar)."""
    for key, default in SESSION_DEFAULTS.items():
        st.session_state[key] = (
            list(default)
            if isinstance(default, list)
            else dict(default)
            if isinstance(default, dict)
            else default
        )


# ---------------------------------------------------------------------------
# Operasyonlar
# ---------------------------------------------------------------------------


def operation_history() -> list[dict[str, Any]]:
    """Operasyon geçmişini döndürür."""
    return list(st.session_state.get("operation_history") or [])


def apply_and_record(action: str, params: dict[str, Any]) -> dict[str, Any]:
    """Operasyonu ``working_df`` üzerinde uygular ve geçmişe kaydeder.

    Args:
        action: Operasyon adı.
        params: Operasyon parametreleri.

    Returns:
        Kaydedilen operasyonun sözlük temsili.

    Raises:
        EdaError: İşlem uygulanamazsa (veri değişmez).
    """
    df = get_working_df()
    if df is None:
        raise EdaError("Önce bir veri seti yüklemelisiniz.")

    new_df, operation = operations.apply_operation(df, action, params)
    st.session_state["working_df"] = new_df
    history = list(st.session_state.get("operation_history") or [])
    history.append(operation.to_dict())
    st.session_state["operation_history"] = history
    _drop_stale_filters(new_df)
    return operation.to_dict()


def undo_last_operation() -> dict[str, Any] | None:
    """Son operasyonu geri alır (kalan geçmişi replay ederek).

    Returns:
        Geri alınan operasyonun sözlüğü; geçmiş boşsa None.
    """
    original = get_original_df()
    if original is None:
        return None
    history = operation_history()
    if not history:
        return None
    new_df, remaining, removed, errors = operations.undo_last(original, history)
    st.session_state["working_df"] = new_df
    st.session_state["operation_history"] = remaining
    st.session_state["replay_errors"] = errors
    _drop_stale_filters(new_df)
    return removed


def reset_dataset() -> None:
    """Veri setini orijinal haline döndürür ve geçmişi temizler."""
    original = get_original_df()
    if original is None:
        return
    st.session_state["working_df"] = operations.reset_dataframe(original)
    st.session_state["operation_history"] = []
    st.session_state["replay_errors"] = []
    st.session_state["active_filters"] = []
    st.session_state["filters_enabled"] = False


# ---------------------------------------------------------------------------
# Grafikler
# ---------------------------------------------------------------------------


def generated_charts() -> list[dict[str, Any]]:
    """Oluşturulan grafik tanımlarını döndürür."""
    return list(st.session_state.get("generated_charts") or [])


def add_chart(spec: dict[str, Any]) -> None:
    """Grafik tanımını session'a ekler (kod export için saklanır)."""
    charts = list(st.session_state.get("generated_charts") or [])
    charts.append(spec)
    st.session_state["generated_charts"] = charts


def remove_chart(chart_id: str) -> None:
    """Grafik tanımını listeden çıkarır."""
    st.session_state["generated_charts"] = [
        chart for chart in generated_charts() if chart.get("id") != chart_id
    ]


def clear_charts() -> None:
    """Tüm grafik tanımlarını temizler."""
    st.session_state["generated_charts"] = []


# ---------------------------------------------------------------------------
# Filtreler
# ---------------------------------------------------------------------------


def get_active_filters() -> list[dict[str, Any]]:
    """Geçici (görselleştirme/analiz) filtrelerini döndürür."""
    return list(st.session_state.get("active_filters") or [])


def set_active_filters(
    filters: list[dict[str, Any]],
    enabled: bool | None = None,
    combine: str | None = None,
) -> None:
    """Geçici filtreleri ve birleştirme mantığını günceller.

    Args:
        filters: Filtre sözlükleri.
        enabled: Filtrelerin diğer analiz sayfalarında da kullanılıp kullanılmayacağı.
        combine: ``and`` veya ``or``.
    """
    st.session_state["active_filters"] = list(filters or [])
    if combine is not None:
        st.session_state["filters_combine"] = str(combine)
    if enabled is not None:
        st.session_state["filters_enabled"] = bool(enabled)
    elif not filters:
        st.session_state["filters_enabled"] = False


def filters_combine() -> str:
    """Geçici filtrelerin birleştirme mantığını döndürür (``and``/``or``)."""
    return str(st.session_state.get("filters_combine") or "and")


def filters_active() -> bool:
    """Geçici filtrelerin aktif olup olmadığını döndürür."""
    return bool(st.session_state.get("filters_enabled")) and bool(get_active_filters())


def _drop_stale_filters(df: pd.DataFrame) -> None:
    """Veri değiştiğinde artık geçerli olmayan filtreleri temizler."""
    filters = get_active_filters()
    if not filters:
        return
    valid = [f for f in filters if f.get("column") in df.columns]
    if len(valid) != len(filters):
        set_active_filters(valid)
