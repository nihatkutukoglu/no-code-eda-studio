"""DataFrame seviyesinde tip tespiti, profil çıkarma ve güvenli dönüşümler.

Streamlit'e bağımsızdır; tüm fonksiyonlar saf pandas üzerinde çalışır.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from utils.helpers import unique_name

__all__ = [
    "numeric_columns",
    "categorical_columns",
    "datetime_columns",
    "boolean_columns",
    "column_kind",
    "describe_dtype",
    "memory_usage_bytes",
    "duplicate_count",
    "total_missing",
    "dataset_shape_info",
    "safe_nunique",
    "get_series",
    "deduplicate_columns",
    "infer_column_role",
    "column_profile",
    "DISPLAY_ROW_LIMIT",
    "safe_display_df",
    "looks_numeric_object",
    "coerce_numeric",
    "constant_columns",
    "empty_columns",
    "potential_id_columns",
    "high_cardinality_columns",
]

#: Olası ID sütunu için varsayılan tekillik oranı eşiği.
DEFAULT_ID_RATIO = 0.95
#: Yüksek kardinalite eşiği (kategorik sütunlar için).
DEFAULT_HIGH_CARDINALITY_RATIO = 0.50
#: Tablo gösteriminde bir seferde gönderilecek maksimum satır (performans sınırı).
DISPLAY_ROW_LIMIT = 20_000


def numeric_columns(df: pd.DataFrame) -> list[str]:
    """Sayısal (int/float) sütun adlarını döndürür. Mantıksal sütunlar hariçtir."""
    cols: list[str] = []
    for col in df.columns:
        series = get_series(df, col)
        if ptypes.is_bool_dtype(series):
            continue
        if ptypes.is_numeric_dtype(series):
            cols.append(str(col))
    return cols


def categorical_columns(df: pd.DataFrame) -> list[str]:
    """Kategorik / metin / mantıksal sütun adlarını döndürür."""
    cols: list[str] = []
    for col in df.columns:
        series = get_series(df, col)
        if ptypes.is_datetime64_any_dtype(series):
            continue
        if ptypes.is_bool_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
            cols.append(str(col))
        elif not ptypes.is_numeric_dtype(series):
            cols.append(str(col))
    return cols


def datetime_columns(df: pd.DataFrame) -> list[str]:
    """Tarih/saat sütun adlarını döndürür."""
    return [
        str(c) for c in df.columns if ptypes.is_datetime64_any_dtype(get_series(df, c))
    ]


def boolean_columns(df: pd.DataFrame) -> list[str]:
    """Mantıksal (bool) sütun adlarını döndürür."""
    return [str(c) for c in df.columns if ptypes.is_bool_dtype(get_series(df, c))]


def column_kind(series: pd.Series) -> str:
    """Sütunun kaba tipini döndürür.

    Returns:
        "numeric", "datetime", "boolean", "categorical" veya "text".
    """
    if ptypes.is_bool_dtype(series):
        return "boolean"
    if ptypes.is_datetime64_any_dtype(series):
        return "datetime"
    if ptypes.is_numeric_dtype(series):
        return "numeric"
    if isinstance(series.dtype, pd.CategoricalDtype):
        return "categorical"
    return "text"


def describe_dtype(series: pd.Series) -> str:
    """Veri tipini kullanıcıya gösterilecek kısa metne çevirir."""
    labels = {
        "numeric": "Sayısal",
        "datetime": "Tarih/Saat",
        "boolean": "Mantıksal",
        "categorical": "Kategorik",
        "text": "Metin",
    }
    return f"{labels[column_kind(series)]} ({series.dtype})"


def memory_usage_bytes(df: pd.DataFrame) -> int:
    """DataFrame'in yaklaşık bellek kullanımını bayt olarak döndürür."""
    try:
        return int(df.memory_usage(deep=True).sum())
    except (TypeError, ValueError):
        return int(df.memory_usage(deep=False).sum())


def duplicate_count(df: pd.DataFrame) -> int:
    """Tam olarak tekrar eden satır sayısını döndürür."""
    if df.empty:
        return 0
    try:
        return int(df.duplicated().sum())
    except TypeError:
        return int(df.astype(str).duplicated().sum())


def total_missing(df: pd.DataFrame) -> int:
    """Toplam boş hücre sayısını döndürür."""
    if df.empty:
        return 0
    return int(df.isna().sum().sum())


def dataset_shape_info(df: pd.DataFrame) -> dict[str, Any]:
    """Dataset'in genel özet metriklerini döndürür.

    Returns:
        rows, columns, cells, memory_bytes, missing_cells, missing_ratio,
        duplicate_rows, duplicate_ratio, numeric_cols, categorical_cols,
        datetime_cols, boolean_cols anahtarlarını içeren sözlük.
    """
    rows, cols = int(df.shape[0]), int(df.shape[1])
    cells = rows * cols
    missing = total_missing(df)
    dups = duplicate_count(df)
    return {
        "rows": rows,
        "columns": cols,
        "cells": cells,
        "memory_bytes": memory_usage_bytes(df),
        "missing_cells": missing,
        "missing_ratio": (missing / cells * 100) if cells else 0.0,
        "duplicate_rows": dups,
        "duplicate_ratio": (dups / rows * 100) if rows else 0.0,
        "numeric_cols": len(numeric_columns(df)),
        "categorical_cols": len(categorical_columns(df)),
        "datetime_cols": len(datetime_columns(df)),
        "boolean_cols": len(boolean_columns(df)),
    }


def safe_nunique(series: pd.Series) -> int:
    """Hashlenemeyen değerlere karşı dayanıklı tekil değer sayımı yapar."""
    try:
        return int(series.nunique(dropna=True))
    except TypeError:
        return int(series.astype(str).nunique(dropna=True))


def get_series(df: pd.DataFrame, column: Any) -> pd.Series:
    """Sütunu her zaman ``Series`` olarak döndürür.

    Tekrarlı sütun adları olan bir DataFrame'de ``df[column]`` bir DataFrame
    döndürür ve sütun bazında işlem yapan kodu bozar. Bu yardımcı ilk eşleşmeyi
    alarak bu durumu güvenli hale getirir.
    """
    value = df[column]
    if isinstance(value, pd.DataFrame):
        return value.iloc[:, 0]
    return value


def deduplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Tekrarlı sütun adlarını teklendirir (``ad``, ``ad_2`` ...).

    Yükleyici bunu zaten yapar; bu fonksiyon son savunma hattıdır: uygulama
    içinde hiçbir zaman tekrarlı etiketli bir DataFrame ile çalışılmaz.
    """
    names = [str(column) for column in df.columns]
    if len(set(names)) == len(names):
        return df
    unique_names: list[str] = []
    for name in names:
        unique_names.append(unique_name(name, unique_names))
    result = df.copy()
    result.columns = unique_names
    return result


def _sample_value(series: pd.Series) -> Any:
    """Sütundan boş olmayan ilk örnek değeri döndürür."""
    non_null = series.dropna()
    if non_null.empty:
        return None
    return non_null.iloc[0]


def infer_column_role(series: pd.Series, total_rows: int | None = None) -> str:
    """Sütunun analizdeki olası rolünü kural tabanlı tahmin eder.

    Bu yalnızca bir **tahmindir**; kesin bir sınıflandırma değildir.

    Args:
        series: İncelenecek sütun.
        total_rows: Veri setinin satır sayısı (None ise serinin uzunluğu).

    Returns:
        "Boş", "Sabit", "Kimlik (ID?)", "Tarih", "Mantıksal",
        "Ölçüm (sayısal)", "Kategorik" veya "Serbest metin".
    """
    rows = int(total_rows if total_rows is not None else len(series))
    if rows == 0 or bool(series.isna().all()):
        return "Boş"

    nunique = safe_nunique(series)
    if nunique <= 1:
        return "Sabit"

    filled = int(series.notna().sum())
    unique_ratio = nunique / filled if filled else 0.0
    kind = column_kind(series)

    if kind == "datetime":
        return "Tarih"
    if kind == "boolean":
        return "Mantıksal"
    if unique_ratio >= DEFAULT_ID_RATIO and nunique >= 20:
        return "Kimlik (ID?)"
    if kind == "numeric":
        # Az sayıda tekil değer taşıyan sayısal sütunlar genelde kodlanmış kategoridir.
        if nunique <= 10 and unique_ratio < 0.05:
            return "Kategorik (kodlanmış)"
        return "Ölçüm (sayısal)"
    if nunique >= 50 and unique_ratio >= DEFAULT_HIGH_CARDINALITY_RATIO:
        return "Serbest metin"
    return "Kategorik"


def column_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Her sütun için profil tablosu üretir.

    Returns:
        Sütun, Veri Tipi, Rol (tahmin), Boş, Boş %, Dolu, Tekil, Tekil %,
        Örnek Değer kolonlarından oluşan DataFrame.
    """
    rows = int(len(df))
    records: list[dict[str, Any]] = []
    for col in df.columns:
        series = get_series(df, col)
        missing = int(series.isna().sum())
        nunique = safe_nunique(series)
        records.append(
            {
                "Sütun": str(col),
                "Veri Tipi": describe_dtype(series),
                "Rol (tahmin)": infer_column_role(series, rows),
                "Boş": missing,
                "Boş %": round(missing / rows * 100, 2) if rows else 0.0,
                "Dolu": rows - missing,
                "Tekil": nunique,
                "Tekil %": round(nunique / rows * 100, 2) if rows else 0.0,
                "Örnek Değer": _sample_value(series),
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=[
            "Sütun",
            "Veri Tipi",
            "Rol (tahmin)",
            "Boş",
            "Boş %",
            "Dolu",
            "Tekil",
            "Tekil %",
            "Örnek Değer",
        ],
    )


def safe_display_df(df: pd.DataFrame, max_rows: int | None = None) -> pd.DataFrame:
    """Streamlit/Arrow uyumlu güvenli bir görüntüleme kopyası üretir.

    Karışık tipli object sütunları metne çevrilir, sonsuz değerler NaN olur.
    Çok büyük veri setlerinde gereksiz kopya oluşmaması için gösterim
    ``DISPLAY_ROW_LIMIT`` satırla sınırlanır (tabloda zaten tamamı okunamaz).

    Args:
        df: Gösterilecek DataFrame.
        max_rows: Gösterilecek maksimum satır sayısı (None ise üst sınır uygulanır).
    """
    limit = DISPLAY_ROW_LIMIT if max_rows is None else min(int(max_rows), DISPLAY_ROW_LIMIT)
    view = df.head(limit) if len(df) > limit else df
    # Tekrarlı sütun adları Arrow dönüşümünü ve sütun atamalarını bozar.
    view = deduplicate_columns(view).copy()
    for index, col in enumerate(view.columns):
        series = view.iloc[:, index]
        if ptypes.is_numeric_dtype(series) and not ptypes.is_bool_dtype(series):
            view[col] = series.replace([np.inf, -np.inf], np.nan)
        elif isinstance(series.dtype, pd.CategoricalDtype):
            view[col] = series.astype(str)
        elif series.dtype == object:
            view[col] = series.map(_stringify_cell)
    view.columns = [str(c) for c in view.columns]
    return view


def _stringify_cell(value: Any) -> Any:
    """Arrow'un serileştiremeyeceği hücreleri metne çevirir."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def looks_numeric_object(series: pd.Series, threshold: float = 0.80) -> bool:
    """Object tipli bir sütunun aslında sayısal olup olmadığını tahmin eder.

    Args:
        series: İncelenecek sütun.
        threshold: Sayıya çevrilebilen dolu değerlerin minimum oranı.
    """
    if ptypes.is_numeric_dtype(series) or ptypes.is_datetime64_any_dtype(series):
        return False
    non_null = series.dropna()
    if non_null.empty:
        return False
    normalized = non_null.astype(str).str.strip().str.replace(" ", "", regex=False)
    converted = pd.to_numeric(normalized.str.replace(",", ".", regex=False), errors="coerce")
    return float(converted.notna().mean()) >= threshold


def coerce_numeric(series: pd.Series) -> pd.Series:
    """Metin sütununu mümkün olduğunca sayıya çevirir (çevrilemeyenler NaN)."""
    if ptypes.is_numeric_dtype(series):
        return series
    text = series.astype(str).str.strip().str.replace(" ", "", regex=False)
    return pd.to_numeric(text.str.replace(",", ".", regex=False), errors="coerce")


def constant_columns(df: pd.DataFrame) -> list[str]:
    """Tek bir tekil değere sahip (bilgi taşımayan) sütunları döndürür."""
    return [str(c) for c in df.columns if safe_nunique(get_series(df, c)) <= 1]


def empty_columns(df: pd.DataFrame) -> list[str]:
    """Tamamen boş sütunları döndürür."""
    return [str(c) for c in df.columns if bool(get_series(df, c).isna().all())]


def potential_id_columns(
    df: pd.DataFrame, ratio: float = DEFAULT_ID_RATIO
) -> list[tuple[str, float]]:
    """Olası kimlik (ID) sütunlarını tekillik oranına göre tahmin eder.

    Returns:
        (sütun_adı, tekillik_oranı) çiftlerinden oluşan liste.
    """
    rows = int(len(df))
    if rows == 0:
        return []
    found: list[tuple[str, float]] = []
    for col in df.columns:
        series = get_series(df, col).dropna()
        if series.empty:
            continue
        r = safe_nunique(series) / rows
        if r >= ratio:
            found.append((str(col), r))
    return sorted(found, key=lambda item: item[1], reverse=True)


def high_cardinality_columns(
    df: pd.DataFrame,
    ratio: float = DEFAULT_HIGH_CARDINALITY_RATIO,
    min_unique: int = 20,
) -> list[tuple[str, int, float]]:
    """Kategorik sütunlarda aşırı tekil değer taşıyanları bulur.

    Returns:
        (sütun, tekil_sayısı, tekillik_oranı) listesi.
    """
    rows = int(len(df))
    if rows == 0:
        return []
    result: list[tuple[str, int, float]] = []
    for col in categorical_columns(df):
        series = get_series(df, col).dropna()
        if series.empty:
            continue
        nunique = safe_nunique(series)
        r = nunique / rows
        if nunique >= min_unique and r >= ratio:
            result.append((str(col), nunique, r))
    return sorted(result, key=lambda item: item[2], reverse=True)
