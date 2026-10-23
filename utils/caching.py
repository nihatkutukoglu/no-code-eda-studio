"""Pahalı hesaplamalar için Streamlit cache sarmalayıcıları.

``core`` modülleri Streamlit'ten bağımsız kalsın diye cache yalnızca burada
tanımlanır. Böylece aynı fonksiyonlar testlerde cache olmadan çağrılabilir.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from core import correlations, insights, outliers, profiler
from core.data_quality_score import QualityScore, compute_quality_score
from utils import dataframe_utils as dfu

__all__ = [
    "column_profile",
    "dataset_shape_info",
    "dtype_check_table",
    "unique_value_table",
    "correlation_matrix",
    "generate_insights",
    "quality_score",
    "outlier_summary_table",
    "is_large_dataset",
    "LARGE_DATASET_ROWS",
]

#: Bu satır sayısının üzerindeki veri setleri "büyük" sayılır.
LARGE_DATASET_ROWS = 200_000

_CACHE_TTL = 900  # 15 dakika


def is_large_dataset(df: pd.DataFrame | None) -> bool:
    """Veri setinin ağır hesaplamalar için büyük olup olmadığını söyler."""
    return df is not None and len(df) > LARGE_DATASET_ROWS


@st.cache_data(show_spinner=False, ttl=_CACHE_TTL)
def column_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Sütun profil tablosu (cache'li)."""
    return dfu.column_profile(df)


@st.cache_data(show_spinner=False, ttl=_CACHE_TTL)
def dataset_shape_info(df: pd.DataFrame) -> dict[str, Any]:
    """Dataset özet metrikleri (cache'li)."""
    return dfu.dataset_shape_info(df)


@st.cache_data(show_spinner=False, ttl=_CACHE_TTL)
def dtype_check_table(df: pd.DataFrame) -> pd.DataFrame:
    """Veri tipi tutarsızlık tablosu (cache'li)."""
    return profiler.dtype_check_table(df)


@st.cache_data(show_spinner=False, ttl=_CACHE_TTL)
def unique_value_table(df: pd.DataFrame) -> pd.DataFrame:
    """Tekillik tablosu (cache'li)."""
    return profiler.unique_value_table(df)


@st.cache_data(show_spinner=False, ttl=_CACHE_TTL)
def correlation_matrix(
    df: pd.DataFrame, columns: tuple[str, ...], method: str
) -> pd.DataFrame:
    """Korelasyon matrisi (cache'li).

    Not: cache anahtarı hashlenebilir olsun diye ``columns`` tuple alınır.
    """
    return correlations.correlation_matrix(df, list(columns), method)


@st.cache_data(show_spinner=False, ttl=_CACHE_TTL)
def generate_insights(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Kural tabanlı insight listesi (cache'li)."""
    return insights.generate_insights(df)


@st.cache_data(show_spinner=False, ttl=_CACHE_TTL)
def quality_score(df: pd.DataFrame) -> QualityScore:
    """Veri kalite skoru (cache'li)."""
    return compute_quality_score(df)


@st.cache_data(show_spinner=False, ttl=_CACHE_TTL)
def outlier_summary_table(df: pd.DataFrame, factor: float) -> pd.DataFrame:
    """Tüm sayısal sütunlar için aykırı değer özeti (cache'li)."""
    return outliers.outlier_summary_table(df, None, factor)
