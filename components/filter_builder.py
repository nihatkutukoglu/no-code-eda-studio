"""No-code filtre oluşturucu arayüzü.

Hem Grafik Oluşturucu hem de Veri Temizleme sayfasında kullanılabilir.
Filtreler session state içinde ``{key_prefix}_filters`` anahtarında tutulur ve
``core.filters`` tarafından uygulanır.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from components.messages import show_error, show_info
from core.filters import (
    apply_filters,
    describe_filter,
    filter_impact,
    needs_second_value,
    needs_value,
    operators_for_column,
)
from utils import dataframe_utils as dfu
from utils.helpers import format_number
from utils.validators import EdaError

__all__ = ["render_filter_builder", "filter_summary_caption"]

#: Kategorik değer seçimi için açılır liste gösterilecek maksimum tekil sayı.
_MAX_CHOICES = 200


def _value_inputs(
    df: pd.DataFrame, column: str, operator: str, key_prefix: str
) -> tuple[Any, Any]:
    """Operatöre ve sütun tipine uygun değer giriş alanlarını çizer.

    Returns:
        ``(value, value2)`` ikilisi.
    """
    if not needs_value(operator):
        return None, None

    kind = dfu.column_kind(df[column])
    series = df[column].dropna()

    if kind == "numeric":
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        low = float(numeric.min()) if not numeric.empty else 0.0
        high = float(numeric.max()) if not numeric.empty else 1.0
        if needs_second_value(operator):
            left, right = st.columns(2)
            value = left.number_input("Alt sınır", value=low, key=f"{key_prefix}_v1")
            value2 = right.number_input("Üst sınır", value=high, key=f"{key_prefix}_v2")
            return value, value2
        default = float(numeric.median()) if not numeric.empty else 0.0
        return st.number_input("Değer", value=default, key=f"{key_prefix}_v1"), None

    if kind == "datetime":
        dates = pd.to_datetime(series, errors="coerce").dropna()
        min_date = dates.min().date() if not dates.empty else None
        max_date = dates.max().date() if not dates.empty else None
        if needs_second_value(operator):
            left, right = st.columns(2)
            start = left.date_input("Başlangıç", value=min_date, key=f"{key_prefix}_v1")
            end = right.date_input("Bitiş", value=max_date, key=f"{key_prefix}_v2")
            return str(start), str(end)
        picked = st.date_input("Tarih", value=min_date, key=f"{key_prefix}_v1")
        return str(picked), None

    # Metin / kategorik
    unique_values = [str(v) for v in pd.unique(series.astype(str))][:_MAX_CHOICES]
    if operator in ("is_in", "not_in"):
        selection = st.multiselect(
            "Değerler", unique_values, key=f"{key_prefix}_v1",
            help="Birden fazla değer seçebilirsiniz.",
        )
        return selection, None
    if operator in ("equals", "not_equals") and len(unique_values) <= _MAX_CHOICES:
        return st.selectbox("Değer", unique_values, key=f"{key_prefix}_v1"), None
    return st.text_input("Metin", key=f"{key_prefix}_v1"), None


def render_filter_builder(
    df: pd.DataFrame,
    key_prefix: str,
    title: str = "Filtreler",
    description: str | None = None,
    combine: str = "and",
) -> list[dict[str, Any]]:
    """Filtre oluşturma arayüzünü çizer ve geçerli filtre listesini döndürür.

    Args:
        df: Filtrelerin tanımlandığı DataFrame.
        key_prefix: Widget/state anahtarları için önek.
        title: Bölüm başlığı.
        description: Açıklama metni.
        combine: ``and`` veya ``or`` (etki özetini doğru hesaplamak için).

    Returns:
        Tanımlı filtrelerin listesi.
    """
    state_key = f"{key_prefix}_filters"
    if state_key not in st.session_state:
        st.session_state[state_key] = []
    # Veri değiştiyse geçersiz filtreleri temizle.
    st.session_state[state_key] = [
        flt for flt in st.session_state[state_key] if flt.get("column") in df.columns
    ]

    st.markdown(f"**{title}**")
    if description:
        st.caption(description)

    columns = [str(c) for c in df.columns]
    selector, operator_box, value_box = st.columns([1.2, 1.2, 1.6])
    with selector:
        column = st.selectbox("Sütun", columns, key=f"{key_prefix}_col")
    operators = operators_for_column(df, column)
    with operator_box:
        operator = st.selectbox(
            "Koşul",
            list(operators),
            format_func=lambda key: operators[key],
            key=f"{key_prefix}_op",
        )
    with value_box:
        value, value2 = _value_inputs(df, column, operator, key_prefix)

    if st.button("➕ Filtre ekle", key=f"{key_prefix}_add"):
        candidate = {
            "column": column,
            "operator": operator,
            "value": value,
            "value2": value2,
        }
        try:
            apply_filters(df, [candidate])  # doğrulama amaçlı deneme
        except EdaError as exc:
            show_error(exc)
        else:
            st.session_state[state_key] = [*st.session_state[state_key], candidate]
            st.rerun()

    filters: list[dict[str, Any]] = list(st.session_state[state_key])
    if not filters:
        show_info("Henüz filtre yok. Tüm veri kullanılıyor.")
        return []

    joiner = "VE" if combine == "and" else "VEYA"
    st.caption(f"Aktif filtreler ({joiner} mantığıyla birleştirilir):")
    for index, flt in enumerate(filters):
        row = st.columns([6, 1])
        row[0].markdown(f"`{index + 1}.` {describe_filter(flt)}")
        if row[1].button("🗑️", key=f"{key_prefix}_del_{index}", help="Filtreyi kaldır"):
            remaining = [f for i, f in enumerate(filters) if i != index]
            st.session_state[state_key] = remaining
            st.rerun()

    if st.button("🧹 Tüm filtreleri temizle", key=f"{key_prefix}_clear"):
        st.session_state[state_key] = []
        st.rerun()

    st.caption(f"Sonuç: {filter_impact(df, filters, combine)}")
    return filters


def filter_summary_caption(df: pd.DataFrame, filtered: pd.DataFrame, active: bool) -> None:
    """Filtre durumunu tek satırda özetler."""
    if not active:
        st.caption(f"Veri: tüm satırlar ({format_number(len(df))}).")
        return
    st.caption(
        f"Veri: filtrelenmiş ({format_number(len(filtered))} / "
        f"{format_number(len(df))} satır). Filtre yalnızca görüntüleme/analiz "
        "içindir; veri setiniz değişmedi."
    )
