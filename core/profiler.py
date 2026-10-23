"""Veri kalitesi profilleme fonksiyonları.

Bu modül **hiçbir veriyi değiştirmez**; yalnızca analiz üretir:
boş değerler, tekrarlar, sabit sütunlar, tekillik, olası ID sütunları ve
veri tipi tutarsızlıkları.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from utils import dataframe_utils as dfu

__all__ = [
    "overview_metrics",
    "missing_value_table",
    "duplicate_report",
    "constant_column_table",
    "near_constant_table",
    "unique_value_table",
    "potential_id_table",
    "high_cardinality_table",
    "dtype_check_table",
    "value_problem_table",
    "duplicate_column_names",
    "value_counts_table",
]


def overview_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """Dataset Overview ekranının metriklerini döndürür."""
    return dfu.dataset_shape_info(df)


def missing_value_table(df: pd.DataFrame, only_missing: bool = False) -> pd.DataFrame:
    """Sütun bazında boş değer tablosu üretir.

    Args:
        df: Kaynak DataFrame.
        only_missing: True ise yalnızca boş değer içeren sütunlar döner.

    Returns:
        ``Sütun``, ``Boş Değer``, ``Boş %``, ``Dolu Değer`` kolonları.
    """
    rows = int(len(df))
    records: list[dict[str, Any]] = []
    for col in df.columns:
        missing = int(dfu.get_series(df, col).isna().sum())
        if only_missing and missing == 0:
            continue
        records.append(
            {
                "Sütun": str(col),
                "Boş Değer": missing,
                "Boş %": round(missing / rows * 100, 2) if rows else 0.0,
                "Dolu Değer": rows - missing,
            }
        )
    table = pd.DataFrame.from_records(
        records, columns=["Sütun", "Boş Değer", "Boş %", "Dolu Değer"]
    )
    if not table.empty:
        table = table.sort_values("Boş %", ascending=False).reset_index(drop=True)
    return table


def duplicate_report(df: pd.DataFrame, sample_size: int = 10) -> dict[str, Any]:
    """Tekrarlı satır analizini üretir.

    Returns:
        ``count``, ``ratio``, ``unique_rows``, ``sample`` (örnek tekrar kayıtları)
        anahtarlarını içeren sözlük.
    """
    rows = int(len(df))
    if rows == 0 or df.shape[1] == 0:
        # Sütunsuz veya satırsız veride tekrar analizi tanımsızdır.
        return {
            "count": 0,
            "ratio": 0.0,
            "affected_rows": 0,
            "unique_rows": rows,
            "sample": df.head(0),
        }
    try:
        dup_mask = df.duplicated(keep=False)
        drop_mask = df.duplicated(keep="first")
    except TypeError:
        as_text = df.astype(str)
        dup_mask = as_text.duplicated(keep=False)
        drop_mask = as_text.duplicated(keep="first")

    removable = int(drop_mask.sum())
    sample = df.loc[dup_mask]
    if not sample.empty:
        # Tekrar gruplarını bir arada göstermek için sırala.
        try:
            sample = sample.sort_values(by=list(df.columns))
        except TypeError:
            pass
    return {
        "count": removable,
        "ratio": (removable / rows * 100) if rows else 0.0,
        "affected_rows": int(dup_mask.sum()),
        "unique_rows": rows - removable,
        "sample": sample.head(sample_size),
    }


def constant_column_table(df: pd.DataFrame) -> pd.DataFrame:
    """Tek tekil değere sahip sütunları listeler.

    Returns:
        ``Sütun``, ``Tek Değer``, ``Boş Değer`` kolonları.
    """
    records: list[dict[str, Any]] = []
    for col in dfu.constant_columns(df):
        series = dfu.get_series(df, col)
        non_null = series.dropna()
        records.append(
            {
                "Sütun": str(col),
                "Tek Değer": non_null.iloc[0] if not non_null.empty else "(tamamen boş)",
                "Boş Değer": int(series.isna().sum()),
            }
        )
    return pd.DataFrame.from_records(records, columns=["Sütun", "Tek Değer", "Boş Değer"])


def unique_value_table(df: pd.DataFrame) -> pd.DataFrame:
    """Sütun bazında tekillik tablosu üretir.

    Returns:
        ``Sütun``, ``Veri Tipi``, ``Tekil``, ``Tekil %``, ``En Sık Değer``,
        ``En Sık Frekans`` kolonları.
    """
    rows = int(len(df))
    records: list[dict[str, Any]] = []
    for col in df.columns:
        series = dfu.get_series(df, col)
        nunique = dfu.safe_nunique(series)
        try:
            counts = series.value_counts(dropna=True)
        except TypeError:
            counts = series.astype(str).value_counts(dropna=True)
        top_value = counts.index[0] if not counts.empty else None
        top_freq = int(counts.iloc[0]) if not counts.empty else 0
        records.append(
            {
                "Sütun": str(col),
                "Veri Tipi": dfu.describe_dtype(series),
                "Tekil": nunique,
                "Tekil %": round(nunique / rows * 100, 2) if rows else 0.0,
                "En Sık Değer": top_value,
                "En Sık Frekans": top_freq,
            }
        )
    table = pd.DataFrame.from_records(
        records,
        columns=["Sütun", "Veri Tipi", "Tekil", "Tekil %", "En Sık Değer", "En Sık Frekans"],
    )
    if not table.empty:
        table = table.sort_values("Tekil %", ascending=False).reset_index(drop=True)
    return table


def potential_id_table(df: pd.DataFrame, ratio: float = dfu.DEFAULT_ID_RATIO) -> pd.DataFrame:
    """Olası ID sütunlarını listeler (yalnızca tahmin).

    Returns:
        ``Sütun``, ``Tekil``, ``Tekillik Oranı %``, ``Veri Tipi`` kolonları.
    """
    records: list[dict[str, Any]] = []
    for col, r in dfu.potential_id_columns(df, ratio):
        records.append(
            {
                "Sütun": col,
                "Tekil": dfu.safe_nunique(dfu.get_series(df, col)),
                "Tekillik Oranı %": round(r * 100, 2),
                "Veri Tipi": dfu.describe_dtype(dfu.get_series(df, col)),
            }
        )
    return pd.DataFrame.from_records(
        records, columns=["Sütun", "Tekil", "Tekillik Oranı %", "Veri Tipi"]
    )


def _looks_like_date(series: pd.Series, threshold: float = 0.80) -> bool:
    """Metin sütununun tarih içerip içermediğini basit kurallarla tahmin eder."""
    non_null = series.dropna().astype(str).str.strip()
    if non_null.empty:
        return False
    sample = non_null.head(200)
    # Rakam ve tarih ayracı içermeyen değerleri hızlıca ele.
    plausible = sample.str.contains(r"\d", regex=True) & sample.str.contains(
        r"[-/.:]", regex=True
    )
    if float(plausible.mean()) < threshold:
        return False
    converted = pd.to_datetime(sample, errors="coerce", format="mixed", dayfirst=False)
    return float(converted.notna().mean()) >= threshold


def _mixed_types(series: pd.Series) -> bool:
    """Object sütunda birden fazla Python tipi bulunup bulunmadığını kontrol eder."""
    if series.dtype != object:
        return False
    types = {type(v).__name__ for v in series.dropna().head(500)}
    return len(types) > 1


def dtype_check_table(df: pd.DataFrame) -> pd.DataFrame:
    """Veri tipi tutarsızlıklarını kural tabanlı olarak tespit eder.

    Tespit edilen durumlar:
        * Sayısal görünen ama metin olarak saklanan sütunlar.
        * Tarih görünen ama metin olarak saklanan sütunlar.
        * 0/1 veya Evet/Hayır gibi mantıksal görünen sütunlar.
        * Karışık Python tipleri içeren sütunlar.
        * Tamamen boş sütunlar.

    Returns:
        ``Sütun``, ``Mevcut Tip``, ``Tespit``, ``Öneri`` kolonları.
    """
    records: list[dict[str, Any]] = []
    for col in df.columns:
        series = dfu.get_series(df, col)
        current = str(series.dtype)
        findings: list[tuple[str, str]] = []

        if bool(series.isna().all()):
            findings.append(("Sütun tamamen boş", "Sütunu silmeyi değerlendirin"))
        elif ptypes.is_numeric_dtype(series) or ptypes.is_datetime64_any_dtype(series):
            if ptypes.is_float_dtype(series):
                non_null = series.dropna()
                if not non_null.empty and bool((non_null % 1 == 0).all()):
                    findings.append(
                        ("Tüm değerler tam sayı ama float saklanıyor", "integer'a dönüştürün")
                    )
        else:
            if dfu.looks_numeric_object(series):
                findings.append(
                    ("Sayısal görünüyor ama metin olarak saklanıyor", "float/integer'a dönüştürün")
                )
            elif _looks_like_date(series):
                findings.append(
                    ("Tarih görünüyor ama metin olarak saklanıyor", "datetime'a dönüştürün")
                )
            else:
                values = {
                    str(v).strip().lower() for v in series.dropna().unique()[:10]
                }
                if values and values <= {"0", "1", "true", "false", "yes", "no", "evet", "hayır"}:
                    findings.append(("Mantıksal (boolean) görünüyor", "boolean'a dönüştürün"))
                nunique = dfu.safe_nunique(series)
                if nunique and len(df) and nunique / len(df) < 0.05 and nunique <= 50:
                    findings.append(
                        ("Az sayıda tekil değer taşıyor", "category tipine dönüştürerek bellek kazanın")
                    )
            if _mixed_types(series):
                findings.append(("Karışık veri tipleri içeriyor", "Tek bir tipe dönüştürün"))

        for detection, suggestion in findings:
            records.append(
                {
                    "Sütun": str(col),
                    "Mevcut Tip": current,
                    "Tespit": detection,
                    "Öneri": suggestion,
                }
            )
    return pd.DataFrame.from_records(
        records, columns=["Sütun", "Mevcut Tip", "Tespit", "Öneri"]
    )


def near_constant_table(df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """Neredeyse sabit sütunları listeler (baskın değer oranı yüksek).

    Args:
        df: Kaynak DataFrame.
        threshold: Baskın değerin minimum oranı (0-1).

    Returns:
        ``Sütun``, ``Baskın Değer``, ``Baskın Oran %``, ``Tekil`` kolonları.
    """
    rows = int(len(df))
    constants = set(dfu.constant_columns(df))
    records: list[dict[str, Any]] = []
    if rows == 0:
        return pd.DataFrame(
            records, columns=["Sütun", "Baskın Değer", "Baskın Oran %", "Tekil"]
        )
    for col in df.columns:
        if str(col) in constants:
            continue  # tamamen sabit sütunlar ayrı bildiriliyor
        series = dfu.get_series(df, col).dropna()
        if series.empty:
            continue
        try:
            counts = series.value_counts()
        except TypeError:
            counts = series.astype(str).value_counts()
        ratio = float(counts.iloc[0]) / rows
        if ratio >= threshold:
            records.append(
                {
                    "Sütun": str(col),
                    "Baskın Değer": counts.index[0],
                    "Baskın Oran %": round(ratio * 100, 2),
                    "Tekil": dfu.safe_nunique(series),
                }
            )
    table = pd.DataFrame.from_records(
        records, columns=["Sütun", "Baskın Değer", "Baskın Oran %", "Tekil"]
    )
    if not table.empty:
        table = table.sort_values("Baskın Oran %", ascending=False).reset_index(drop=True)
    return table


def high_cardinality_table(
    df: pd.DataFrame, ratio: float = dfu.DEFAULT_HIGH_CARDINALITY_RATIO, min_unique: int = 20
) -> pd.DataFrame:
    """Yüksek kardinaliteli kategorik sütunları listeler.

    Returns:
        ``Sütun``, ``Tekil``, ``Tekillik Oranı %``, ``Örnek Değerler`` kolonları.
    """
    records: list[dict[str, Any]] = []
    for col, nunique, unique_ratio in dfu.high_cardinality_columns(df, ratio, min_unique):
        samples = [str(v) for v in dfu.get_series(df, col).dropna().unique()[:3]]
        records.append(
            {
                "Sütun": col,
                "Tekil": nunique,
                "Tekillik Oranı %": round(unique_ratio * 100, 2),
                "Örnek Değerler": ", ".join(samples),
            }
        )
    return pd.DataFrame.from_records(
        records, columns=["Sütun", "Tekil", "Tekillik Oranı %", "Örnek Değerler"]
    )


def value_problem_table(df: pd.DataFrame) -> pd.DataFrame:
    """Değer düzeyindeki sorunları tespit eder.

    Tespit edilenler:
        * Baştaki/sondaki boşluk içeren metinler
        * Çoklu boşluk içeren metinler
        * Boş görünen ama dolu sayılan değerler ("", " ", "-", "NA", "null")
        * Sonsuz (``inf``) sayısal değerler
        * Negatif değerler (bilgi amaçlı)

    Returns:
        ``Sütun``, ``Sorun``, ``Adet``, ``Öneri`` kolonları.
    """
    placeholders = {"", "-", "na", "n/a", "null", "none", "nan", "bilinmiyor", "?"}
    records: list[dict[str, Any]] = []

    for col in df.columns:
        series = dfu.get_series(df, col)
        if ptypes.is_numeric_dtype(series) and not ptypes.is_bool_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce")
            infinite = int(np.isinf(numeric).sum())
            if infinite:
                records.append(
                    {
                        "Sütun": str(col),
                        "Sorun": "Sonsuz (inf) değer",
                        "Adet": infinite,
                        "Öneri": "Bu satırları filtreleyin veya boş değere çevirin",
                    }
                )
            continue
        if ptypes.is_datetime64_any_dtype(series) or ptypes.is_bool_dtype(series):
            continue

        text = series.dropna().astype(str)
        if text.empty:
            continue
        trailing = int((text != text.str.strip()).sum())
        if trailing:
            records.append(
                {
                    "Sütun": str(col),
                    "Sorun": "Baştaki/sondaki boşluk",
                    "Adet": trailing,
                    "Öneri": "Dönüşümler → Metin → boşlukları kırp",
                }
            )
        multi_space = int(text.str.contains(r"\s{2,}", regex=True).sum())
        if multi_space:
            records.append(
                {
                    "Sütun": str(col),
                    "Sorun": "Çoklu boşluk",
                    "Adet": multi_space,
                    "Öneri": "Dönüşümler → Metin → çoklu boşlukları tek boşluğa indir",
                }
            )
        placeholder_count = int(text.str.strip().str.lower().isin(placeholders).sum())
        if placeholder_count:
            records.append(
                {
                    "Sütun": str(col),
                    "Sorun": "Boş değer yerine kullanılan metin ('-', 'NA' gibi)",
                    "Adet": placeholder_count,
                    "Öneri": "Bu değerleri gerçek boş (NaN) değere çevirin",
                }
            )
        case_variants = int(text.nunique() - text.str.lower().nunique())
        if case_variants > 0:
            records.append(
                {
                    "Sütun": str(col),
                    "Sorun": "Yalnızca büyük/küçük harf farkı olan değerler",
                    "Adet": case_variants,
                    "Öneri": "Dönüşümler → Metin → küçük harfe çevir",
                }
            )
    return pd.DataFrame.from_records(records, columns=["Sütun", "Sorun", "Adet", "Öneri"])


def duplicate_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Tekrarlı veya normalize edildiğinde çakışan sütun adlarını bulur.

    Returns:
        ``Sütun Adı``, ``Tekrar``, ``Sorun`` kolonları.
    """
    names = [str(column) for column in df.columns]
    records: list[dict[str, Any]] = []

    exact: dict[str, int] = {}
    for name in names:
        exact[name] = exact.get(name, 0) + 1
    for name, count in exact.items():
        if count > 1:
            records.append(
                {"Sütun Adı": name, "Tekrar": count, "Sorun": "Birebir aynı sütun adı"}
            )

    normalized: dict[str, list[str]] = {}
    for name in names:
        key = name.strip().lower().replace(" ", "_")
        normalized.setdefault(key, []).append(name)
    for key, group in normalized.items():
        if len(set(group)) > 1:
            records.append(
                {
                    "Sütun Adı": ", ".join(sorted(set(group))),
                    "Tekrar": len(group),
                    "Sorun": f"Benzer adlar ('{key}' olarak normalleşiyor)",
                }
            )
    return pd.DataFrame.from_records(records, columns=["Sütun Adı", "Tekrar", "Sorun"])


def value_counts_table(df: pd.DataFrame, column: str, top_n: int = 10) -> pd.DataFrame:
    """Bir sütunun değer frekans tablosunu üretir.

    Returns:
        ``Değer``, ``Frekans``, ``Yüzde %`` kolonları.
    """
    if column not in df.columns:
        return pd.DataFrame(columns=["Değer", "Frekans", "Yüzde %"])
    series = df[column]
    try:
        counts = series.value_counts(dropna=False)
    except TypeError:
        counts = series.astype(str).value_counts(dropna=False)
    total = int(counts.sum()) or 1
    head = counts.head(top_n)
    return pd.DataFrame(
        {
            "Değer": [("(boş)" if pd.isna(v) else v) for v in head.index],
            "Frekans": head.to_numpy(),
            "Yüzde %": [round(float(c) / total * 100, 2) for c in head.to_numpy()],
        }
    )
