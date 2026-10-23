"""Doğrulama yardımcıları ve kullanıcı dostu hata tipleri.

Buradaki hata sınıfları UI katmanında yakalanıp Türkçe mesaj olarak
gösterilir; uygulamanın hiçbir noktada traceback ile çökmemesi hedeflenir.
"""

from __future__ import annotations

import pandas as pd

from utils import dataframe_utils as dfu

__all__ = [
    "EdaError",
    "ValidationError",
    "LoaderError",
    "CleaningError",
    "ChartError",
    "FilterError",
    "require_column",
    "require_columns",
    "require_numeric_column",
]


class EdaError(Exception):
    """Uygulama içinde kullanıcıya gösterilmek üzere üretilen hata."""


class ValidationError(EdaError):
    """Geçersiz kullanıcı seçimi / parametre hatası."""


class LoaderError(EdaError):
    """Dosya okuma veya ayrıştırma hatası."""


class CleaningError(EdaError):
    """Veri temizleme işlemi sırasında oluşan hata."""


class ChartError(EdaError):
    """Grafik oluşturma sırasında oluşan hata."""


class FilterError(EdaError):
    """Filtre tanımı veya uygulaması sırasında oluşan hata."""


def require_column(df: pd.DataFrame, column: str, context: str = "İşlem") -> str:
    """Sütunun DataFrame içinde bulunduğunu doğrular.

    Raises:
        ValidationError: Sütun bulunamazsa.
    """
    if column is None or str(column) == "":
        raise ValidationError(f"{context} için bir sütun seçmelisiniz.")
    if column not in df.columns:
        raise ValidationError(
            f"'{column}' sütunu veri setinde bulunamadı. "
            "Sütun adı değiştirilmiş veya silinmiş olabilir."
        )
    return str(column)


def require_columns(df: pd.DataFrame, columns: list[str], context: str = "İşlem") -> list[str]:
    """Birden fazla sütunun varlığını doğrular."""
    if not columns:
        raise ValidationError(f"{context} için en az bir sütun seçmelisiniz.")
    missing = [str(c) for c in columns if c not in df.columns]
    if missing:
        raise ValidationError(
            "Şu sütunlar veri setinde bulunamadı: " + ", ".join(missing)
        )
    return [str(c) for c in columns]


def require_numeric_column(df: pd.DataFrame, column: str, context: str = "Bu analiz") -> str:
    """Sütunun sayısal olduğunu doğrular.

    Raises:
        ValidationError: Sütun sayısal değilse.
    """
    column = require_column(df, column, context)
    if column not in dfu.numeric_columns(df):
        raise ValidationError(
            f"{context} yalnızca sayısal sütunlarla çalışır. "
            f"'{column}' sütunu sayısal değil ({df[column].dtype}). "
            "Veri Temizleme sayfasından tipi dönüştürebilirsiniz."
        )
    return column
