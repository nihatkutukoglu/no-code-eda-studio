"""Veri temizleme dönüşümleri.

Bu modüldeki tüm fonksiyonlar **saf**tır: verilen DataFrame'i değiştirmez,
yeni bir DataFrame döndürür. Böylece operation replay (undo) deterministik
çalışır ve aynı fonksiyonlar testlerde doğrudan kullanılabilir.

Hatalar her zaman ``CleaningError`` olarak yükseltilir; UI katmanı bunları
yakalayıp kullanıcı dostu mesaj gösterir ve uygulama çökmez.
"""

from __future__ import annotations

import warnings
from typing import Any

import pandas as pd
from pandas.api import types as ptypes

from core import filters as filters_mod
from core import outliers as outliers_mod
from utils import dataframe_utils as dfu
from utils.validators import CleaningError, EdaError

__all__ = [
    "FILL_METHODS",
    "TYPE_TARGETS",
    "drop_missing_rows",
    "fill_missing",
    "drop_duplicates",
    "rename_column",
    "drop_columns",
    "convert_dtype",
    "drop_outlier_rows",
    "filter_rows",
    "missing_row_impact",
    "fill_impact",
    "duplicate_impact",
    "outlier_impact",
    "conversion_preview",
    "coerce_fill_value",
]

#: Boş değer doldurma yöntemleri: ``kod -> etiket``.
FILL_METHODS: dict[str, str] = {
    "mean": "Ortalama (mean)",
    "median": "Medyan (median)",
    "mode": "En sık değer (mode)",
    "value": "Özel değer",
    "ffill": "Önceki değerle doldur (ffill)",
    "bfill": "Sonraki değerle doldur (bfill)",
}

#: Desteklenen veri tipi dönüşümleri: ``kod -> etiket``.
TYPE_TARGETS: dict[str, str] = {
    "integer": "Tam sayı (integer)",
    "float": "Ondalık (float)",
    "string": "Metin (string)",
    "category": "Kategori (category)",
    "datetime": "Tarih/Saat (datetime)",
    "boolean": "Mantıksal (boolean)",
}

_TRUE_TOKENS = {"true", "1", "yes", "y", "evet", "e", "doğru", "dogru", "t"}
_FALSE_TOKENS = {"false", "0", "no", "n", "hayır", "hayir", "h", "yanlış", "yanlis", "f"}


def _require_column(df: pd.DataFrame, column: Any) -> str:
    """Sütunun varlığını doğrular, aksi halde CleaningError yükseltir."""
    if column is None or column not in df.columns:
        raise CleaningError(
            f"'{column}' sütunu veri setinde bulunamadı. "
            "Sütun silinmiş veya adı değiştirilmiş olabilir."
        )
    return str(column)


# ---------------------------------------------------------------------------
# Boş değerler
# ---------------------------------------------------------------------------


def drop_missing_rows(
    df: pd.DataFrame, columns: list[str] | None = None, how: str = "any"
) -> pd.DataFrame:
    """Boş değer içeren satırları siler.

    Args:
        df: Kaynak DataFrame.
        columns: Kontrol edilecek sütunlar. None ise tüm sütunlar.
        how: ``any`` (bir sütun bile boşsa sil) veya ``all`` (tümü boşsa sil).

    Returns:
        Satırları silinmiş yeni DataFrame.
    """
    if how not in ("any", "all"):
        raise CleaningError("'how' parametresi yalnızca 'any' veya 'all' olabilir.")
    subset = None
    if columns:
        subset = [_require_column(df, c) for c in columns]
    result = df.dropna(subset=subset, how=how)
    if result.empty and not df.empty:
        raise CleaningError(
            "Bu işlem tüm satırları silecekti ve uygulanmadı. "
            "Daha dar bir sütun seçimiyle deneyin."
        )
    return result.reset_index(drop=True)


def coerce_fill_value(df: pd.DataFrame, column: str, value: Any) -> Any:
    """Kullanıcının girdiği özel doldurma değerini sütun tipine uyarlar.

    Sayısal sütunda metin girilirse sayıya çevrilmeye çalışılır; çevrilemezse
    anlaşılır bir hata verilir.
    """
    column = _require_column(df, column)
    series = df[column]
    if value is None:
        raise CleaningError("Doldurmak için bir değer girmelisiniz.")
    if isinstance(value, str) and not value.strip():
        raise CleaningError("Doldurmak için bir değer girmelisiniz.")

    kind = dfu.column_kind(series)
    if kind == "numeric":
        try:
            number = float(str(value).replace(",", "."))
        except (TypeError, ValueError) as exc:
            raise CleaningError(
                f"'{value}' sayıya çevrilemedi. '{column}' sayısal bir sütun olduğu "
                "için sayısal bir değer girmelisiniz."
            ) from exc
        if ptypes.is_integer_dtype(series) and float(number).is_integer():
            return int(number)
        return number
    if kind == "datetime":
        try:
            ts = pd.Timestamp(value)
        except (TypeError, ValueError) as exc:
            raise CleaningError(
                f"'{value}' tarihe çevrilemedi. Örnek biçim: 2024-01-31"
            ) from exc
        if pd.isna(ts):
            raise CleaningError(f"'{value}' geçerli bir tarih değil.")
        return ts
    if kind == "boolean":
        text = str(value).strip().lower()
        if text in _TRUE_TOKENS:
            return True
        if text in _FALSE_TOKENS:
            return False
        raise CleaningError(
            f"'{value}' mantıksal değere çevrilemedi. True/False kullanın."
        )
    return str(value)


def fill_missing(
    df: pd.DataFrame, column: str, method: str = "median", value: Any = None
) -> pd.DataFrame:
    """Bir sütundaki boş değerleri doldurur.

    Args:
        df: Kaynak DataFrame.
        column: Doldurulacak sütun.
        method: ``FILL_METHODS`` içindeki bir anahtar.
        value: ``method='value'`` için kullanılacak değer.

    Returns:
        Boş değerleri doldurulmuş yeni DataFrame.

    Raises:
        CleaningError: Yöntem sütun tipiyle uyumsuzsa veya hesaplanamazsa.
    """
    column = _require_column(df, column)
    if method not in FILL_METHODS:
        raise CleaningError(f"Desteklenmeyen doldurma yöntemi: {method}")

    series = df[column]
    result = df.copy()

    if method in ("mean", "median"):
        if column not in dfu.numeric_columns(df):
            raise CleaningError(
                f"'{method}' yöntemi yalnızca sayısal sütunlarda kullanılabilir. "
                f"'{column}' sütunu {series.dtype} tipinde. "
                "Mode veya özel değer ile doldurmayı deneyin."
            )
        stat = series.mean() if method == "mean" else series.median()
        if pd.isna(stat):
            raise CleaningError(
                f"'{column}' sütununda {method} hesaplanamadı (sütun tamamen boş olabilir)."
            )
        result[column] = series.fillna(stat)
        return result

    if method == "mode":
        modes = series.mode(dropna=True)
        if modes.empty:
            raise CleaningError(
                f"'{column}' sütununda en sık değer bulunamadı (sütun tamamen boş)."
            )
        result[column] = series.fillna(modes.iloc[0])
        return result

    if method == "value":
        result[column] = series.fillna(coerce_fill_value(df, column, value))
        return result

    # ffill / bfill
    result[column] = series.ffill() if method == "ffill" else series.bfill()
    return result


# ---------------------------------------------------------------------------
# Tekrarlı satırlar
# ---------------------------------------------------------------------------


def drop_duplicates(
    df: pd.DataFrame, subset: list[str] | None = None, keep: str = "first"
) -> pd.DataFrame:
    """Tekrarlı satırları siler.

    Args:
        df: Kaynak DataFrame.
        subset: Karşılaştırmada kullanılacak sütunlar (None ise tümü).
        keep: ``first`` veya ``last``.

    Returns:
        Tekrarsız yeni DataFrame.
    """
    if keep not in ("first", "last"):
        raise CleaningError("'keep' parametresi 'first' veya 'last' olmalıdır.")
    cols = [_require_column(df, c) for c in subset] if subset else None
    try:
        result = df.drop_duplicates(subset=cols, keep=keep)
    except TypeError as exc:
        raise CleaningError(
            "Bazı sütunlar karşılaştırılabilir tipte olmadığı için tekrar analizi "
            f"yapılamadı: {exc}"
        ) from exc
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sütun işlemleri
# ---------------------------------------------------------------------------


def rename_column(df: pd.DataFrame, column: str, new_name: str) -> pd.DataFrame:
    """Sütun adını değiştirir."""
    column = _require_column(df, column)
    cleaned = str(new_name or "").strip()
    if not cleaned:
        raise CleaningError("Yeni sütun adı boş olamaz.")
    if cleaned == column:
        raise CleaningError("Yeni ad mevcut adla aynı.")
    if cleaned in [str(c) for c in df.columns]:
        raise CleaningError(f"'{cleaned}' adında bir sütun zaten var.")
    return df.rename(columns={column: cleaned})


def drop_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Bir veya birden fazla sütunu siler."""
    if not columns:
        raise CleaningError("Silmek için en az bir sütun seçmelisiniz.")
    targets = [_require_column(df, c) for c in columns]
    if len(targets) >= df.shape[1]:
        raise CleaningError(
            "Tüm sütunları silemezsiniz. En az bir sütun kalmalıdır."
        )
    return df.drop(columns=targets)


def _to_boolean(series: pd.Series) -> pd.Series:
    """Seriyi mantıksal (nullable boolean) tipe çevirir."""
    if ptypes.is_bool_dtype(series):
        return series
    if ptypes.is_numeric_dtype(series):
        return series.map(lambda v: pd.NA if pd.isna(v) else bool(v)).astype("boolean")

    def convert(value: Any) -> Any:
        if pd.isna(value):
            return pd.NA
        text = str(value).strip().lower()
        if text in _TRUE_TOKENS:
            return True
        if text in _FALSE_TOKENS:
            return False
        return pd.NA

    return series.map(convert).astype("boolean")


def convert_dtype(
    df: pd.DataFrame,
    column: str,
    target: str,
    datetime_format: str | None = None,
) -> pd.DataFrame:
    """Sütunun veri tipini dönüştürür.

    Dönüşüm hataları satır bazında NaN'a çevrilir (``errors='coerce'``), böylece
    uygulama çökmez. Tüm değerler kaybolacaksa işlem iptal edilir.

    Args:
        df: Kaynak DataFrame.
        column: Dönüştürülecek sütun.
        target: ``TYPE_TARGETS`` içindeki bir anahtar.
        datetime_format: Tarih dönüşümü için opsiyonel biçim (``%d/%m/%Y`` gibi).

    Returns:
        Tipi dönüştürülmüş yeni DataFrame.

    Raises:
        CleaningError: Hedef tip bilinmiyorsa veya dönüşüm tüm veriyi kaybediyorsa.
    """
    column = _require_column(df, column)
    if target not in TYPE_TARGETS:
        raise CleaningError(f"Desteklenmeyen hedef tip: {target}")

    series = df[column]
    filled_before = int(series.notna().sum())
    result = df.copy()

    try:
        if target == "integer":
            numeric = dfu.coerce_numeric(series)
            converted = numeric.round().astype("Int64")
        elif target == "float":
            converted = dfu.coerce_numeric(series).astype("Float64")
        elif target == "string":
            converted = series.astype("string")
        elif target == "category":
            converted = series.astype("category")
        elif target == "boolean":
            converted = _to_boolean(series)
        else:  # datetime
            with warnings.catch_warnings():
                # Biçim verilmediğinde pandas satır bazında ayrıştırma uyarısı
                # verir; dönüşemeyen değerler zaten NaN'a çevrildiği için bu
                # uyarı kullanıcıyı ilgilendirmiyor.
                warnings.simplefilter("ignore", UserWarning)
                converted = pd.to_datetime(
                    series, errors="coerce", format=datetime_format or None
                )
    except (TypeError, ValueError, OverflowError) as exc:
        raise CleaningError(
            f"'{column}' sütunu {TYPE_TARGETS[target]} tipine dönüştürülemedi: {exc}"
        ) from exc

    filled_after = int(converted.notna().sum())
    if filled_before > 0 and filled_after == 0:
        raise CleaningError(
            f"'{column}' sütunu {TYPE_TARGETS[target]} tipine dönüştürülemedi: "
            "hiçbir değer çevrilemedi, işlem uygulanmadı. "
            "Önce sütun içeriğini kontrol edin."
        )
    result[column] = converted
    return result


# ---------------------------------------------------------------------------
# Aykırı değerler ve filtreler
# ---------------------------------------------------------------------------


def drop_outlier_rows(
    df: pd.DataFrame,
    column: str,
    factor: float = outliers_mod.DEFAULT_IQR_FACTOR,
    method: str = "iqr",
) -> pd.DataFrame:
    """Aykırı değer satırlarını siler (IQR veya Z-Score yöntemiyle).

    Args:
        df: Kaynak DataFrame.
        column: Sayısal sütun.
        factor: IQR çarpanı veya |z| eşiği.
        method: ``iqr`` veya ``zscore``.
    """
    column = _require_column(df, column)
    try:
        mask = outliers_mod.outlier_mask(df, column, factor, method)
    except EdaError as exc:
        raise CleaningError(str(exc)) from exc
    result = df.loc[~mask]
    if result.empty and not df.empty:
        raise CleaningError(
            "Bu işlem tüm satırları silecekti ve uygulanmadı. "
            "Eşik değerini büyütmeyi deneyin."
        )
    return result.reset_index(drop=True)


def filter_rows(
    df: pd.DataFrame, filters: list[dict[str, Any]], combine: str = "and"
) -> pd.DataFrame:
    """Filtre listesini kalıcı olarak uygular (satır seçimi).

    Args:
        df: Kaynak DataFrame.
        filters: Filtre sözlükleri.
        combine: ``and`` veya ``or``.
    """
    if not filters:
        raise CleaningError("Uygulanacak filtre yok.")
    try:
        result = filters_mod.apply_filters(df, filters, combine)
    except EdaError as exc:
        raise CleaningError(str(exc)) from exc
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Etki tahmini (işlem öncesi bilgilendirme)
# ---------------------------------------------------------------------------


def missing_row_impact(
    df: pd.DataFrame, columns: list[str] | None = None, how: str = "any"
) -> int:
    """Boş değer silme işleminin kaç satırı etkileyeceğini döndürür."""
    subset = [c for c in (columns or []) if c in df.columns] or None
    target = df[subset] if subset else df
    if how == "all":
        return int(target.isna().all(axis=1).sum())
    return int(target.isna().any(axis=1).sum())


def fill_impact(df: pd.DataFrame, column: str) -> int:
    """Doldurma işleminin kaç hücreyi etkileyeceğini döndürür."""
    if column not in df.columns:
        return 0
    return int(df[column].isna().sum())


def duplicate_impact(
    df: pd.DataFrame, subset: list[str] | None = None, keep: str = "first"
) -> int:
    """Tekrar silme işleminin kaç satırı etkileyeceğini döndürür."""
    cols = [c for c in (subset or []) if c in df.columns] or None
    try:
        return int(df.duplicated(subset=cols, keep=keep).sum())
    except TypeError:
        return int(df.astype(str).duplicated(subset=cols, keep=keep).sum())


def outlier_impact(
    df: pd.DataFrame,
    column: str,
    factor: float = outliers_mod.DEFAULT_IQR_FACTOR,
    method: str = "iqr",
) -> int:
    """Aykırı değer silme işleminin kaç satırı etkileyeceğini döndürür."""
    try:
        return int(outliers_mod.outlier_mask(df, column, factor, method).sum())
    except EdaError:
        return 0


def conversion_preview(
    df: pd.DataFrame, column: str, target: str, datetime_format: str | None = None
) -> dict[str, Any]:
    """Tip dönüşümünün ön izlemesini üretir (uygulamadan önce bilgi).

    Returns:
        ``ok`` (dönüşen dolu değer sayısı), ``lost`` (kaybolacak değer sayısı),
        ``examples`` (çevrilemeyen örnek değerler), ``error`` (varsa mesaj).
    """
    if column not in df.columns:
        return {"ok": 0, "lost": 0, "examples": [], "error": "Sütun bulunamadı."}
    series = df[column]
    try:
        converted = convert_dtype(df, column, target, datetime_format)[column]
    except CleaningError as exc:
        return {"ok": 0, "lost": int(series.notna().sum()), "examples": [], "error": str(exc)}

    lost_mask = series.notna() & converted.isna()
    examples = (
        series[lost_mask].dropna().astype(str).unique().tolist()[:5] if lost_mask.any() else []
    )
    return {
        "ok": int(converted.notna().sum()),
        "lost": int(lost_mask.sum()),
        "examples": examples,
        "error": None,
    }
