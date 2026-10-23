"""Sayısal, kategorik ve tarih sütunları için istatistiksel özetler.

Çarpıklık (skewness) ve basıklık (kurtosis) hesabı SciPy varsa SciPy ile,
yoksa pandas ile yapılır; böylece modül SciPy olmadan da çalışır.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from utils import dataframe_utils as dfu
from utils.validators import ValidationError, require_numeric_column

try:  # SciPy opsiyonel bağımlılık gibi ele alınır
    from scipy import stats as scipy_stats

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover - SciPy kurulu olmayan ortam
    scipy_stats = None
    _HAS_SCIPY = False

__all__ = [
    "STAT_EXPLANATIONS",
    "numeric_summary",
    "numeric_summary_table",
    "categorical_summary",
    "frequency_table",
    "datetime_summary",
    "describe_all_numeric",
    "shapiro_normality",
]

#: İstatistik kavramlarının kısa, teknik olmayan açıklamaları.
STAT_EXPLANATIONS: dict[str, str] = {
    "mean": "Ortalama, tüm değerlerin toplamının adede bölümüdür; aykırı değerlerden etkilenir.",
    "median": "Medyan, sıralı verinin ortasındaki değerdir; aykırı değerlere karşı daha dayanıklıdır.",
    "std": "Standart sapma, değerlerin ortalama etrafında ne kadar yayıldığını gösterir.",
    "variance": "Varyans, standart sapmanın karesidir; yayılımın bir başka ölçüsüdür.",
    "iqr": "IQR (Q3 - Q1), ortadaki %50'lik dilimin genişliğidir; aykırı değer tespitinde kullanılır.",
    "cv": "Varyasyon katsayısı (std / ortalama), yayılımı ortalamaya göre ölçer; birimden bağımsızdır.",
    "entropy": "Entropi, kategorilerin ne kadar dengeli dağıldığını gösterir (0 = tek kategori).",
    "range": "Aralık, en büyük ve en küçük değer arasındaki farktır.",
    "skewness": "Çarpıklık değeri dağılımın simetrik olup olmadığı hakkında fikir verir (0'a yakın = simetrik).",
    "kurtosis": "Basıklık, dağılımın uçlarının ne kadar kalın olduğunu özetler (0'a yakın = normale benzer).",
    "mode": "Mod, en sık tekrarlanan değerdir.",
    "unique": "Tekil değer sayısı, sütunun kaç farklı değer taşıdığını gösterir.",
}


def _skew_kurt(clean: pd.Series) -> tuple[float, float]:
    """Çarpıklık ve basıklık (excess) değerlerini hesaplar."""
    if len(clean) < 3:
        return float("nan"), float("nan")
    if _HAS_SCIPY:
        return (
            float(scipy_stats.skew(clean, bias=False)),
            float(scipy_stats.kurtosis(clean, bias=False)),
        )
    return float(clean.skew()), float(clean.kurt())


def _clean_numeric(df: pd.DataFrame, column: str) -> pd.Series:
    """Sütunu sayısala çevirip NaN/inf değerlerini atar.

    Raises:
        ValidationError: Sütun yok, sayısal değil veya hiç dolu değer içermiyorsa.
    """
    column = require_numeric_column(df, column, "Sayısal istatistikler")
    series = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    clean = series.dropna()
    if clean.empty:
        raise ValidationError(
            f"'{column}' sütununda istatistik hesaplanacak dolu değer yok."
        )
    return clean


def numeric_summary(df: pd.DataFrame, column: str) -> dict[str, Any]:
    """Sayısal bir sütun için tam istatistik özeti üretir.

    Returns:
        count, missing, mean, median, std, variance, min, q1, q2, q3, max,
        range, iqr, skewness, kurtosis, zeros, negatives anahtarları.
    """
    clean = _clean_numeric(df, column)
    q1, q2, q3 = (float(clean.quantile(q)) for q in (0.25, 0.50, 0.75))
    skewness, kurtosis = _skew_kurt(clean)
    mean = float(clean.mean())
    std = float(clean.std(ddof=1)) if len(clean) > 1 else 0.0
    modes = clean.mode(dropna=True)
    return {
        "column": str(column),
        "count": int(clean.count()),
        "missing": int(df[column].isna().sum()),
        "mean": mean,
        "median": q2,
        "mode": float(modes.iloc[0]) if not modes.empty else float("nan"),
        "std": std,
        "variance": float(clean.var(ddof=1)) if len(clean) > 1 else 0.0,
        "cv": (std / abs(mean)) if mean not in (0, 0.0) else float("nan"),
        "min": float(clean.min()),
        "q1": q1,
        "q2": q2,
        "q3": q3,
        "max": float(clean.max()),
        "range": float(clean.max() - clean.min()),
        "iqr": q3 - q1,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "zeros": int((clean == 0).sum()),
        "negatives": int((clean < 0).sum()),
    }


def numeric_summary_table(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Sayısal özeti ``Metrik``/``Değer`` tablosu olarak döndürür."""
    summary = numeric_summary(df, column)
    order = [
        ("count", "Dolu değer (count)"),
        ("missing", "Boş değer (missing)"),
        ("mean", "Ortalama (mean)"),
        ("median", "Medyan (median)"),
        ("mode", "Mod (en sık değer)"),
        ("std", "Standart sapma (std)"),
        ("variance", "Varyans (variance)"),
        ("cv", "Varyasyon katsayısı (std/ortalama)"),
        ("min", "Minimum"),
        ("q1", "Q1 (%25)"),
        ("q2", "Q2 / Medyan (%50)"),
        ("q3", "Q3 (%75)"),
        ("max", "Maksimum"),
        ("range", "Aralık (range)"),
        ("iqr", "IQR (Q3 - Q1)"),
        ("skewness", "Çarpıklık (skewness)"),
        ("kurtosis", "Basıklık (kurtosis)"),
    ]
    return pd.DataFrame(
        {
            "Metrik": [label for _, label in order],
            "Değer": [summary[key] for key, _ in order],
        }
    )


def categorical_summary(df: pd.DataFrame, column: str, top_n: int = 10) -> dict[str, Any]:
    """Kategorik bir sütun için özet üretir.

    Returns:
        unique, missing, mode, mode_frequency, mode_ratio, top_values (DataFrame),
        imbalance_ratio anahtarları.
    """
    if column not in df.columns:
        raise ValidationError(f"'{column}' sütunu bulunamadı.")
    series = df[column]
    non_null = series.dropna()
    if non_null.empty:
        raise ValidationError(f"'{column}' sütunu tamamen boş; özet üretilemez.")

    try:
        counts = non_null.value_counts()
    except TypeError:
        counts = non_null.astype(str).value_counts()

    total = int(counts.sum())
    mode_value = counts.index[0]
    mode_freq = int(counts.iloc[0])
    probabilities = counts.to_numpy(dtype=float) / total if total else np.array([])
    entropy = (
        float(-(probabilities * np.log2(probabilities)).sum())
        if probabilities.size and (probabilities > 0).all()
        else 0.0
    )
    max_entropy = float(np.log2(len(counts))) if len(counts) > 1 else 0.0
    return {
        "column": str(column),
        "count": int(non_null.shape[0]),
        "missing": int(series.isna().sum()),
        "unique": int(counts.shape[0]),
        "mode": mode_value,
        "mode_frequency": mode_freq,
        "mode_ratio": (mode_freq / total * 100) if total else 0.0,
        "imbalance_ratio": (mode_freq / total) if total else 0.0,
        "entropy": entropy,
        "max_entropy": max_entropy,
        "balance": (entropy / max_entropy) if max_entropy else 1.0,
        "top_values": frequency_table(df, column, top_n),
    }


def frequency_table(df: pd.DataFrame, column: str, top_n: int = 10) -> pd.DataFrame:
    """Frekans ve yüzde dağılımı tablosu üretir.

    Returns:
        ``Değer``, ``Frekans``, ``Yüzde %``, ``Kümülatif %`` kolonları.
    """
    if column not in df.columns:
        raise ValidationError(f"'{column}' sütunu bulunamadı.")
    series = df[column].dropna()
    try:
        counts = series.value_counts()
    except TypeError:
        counts = series.astype(str).value_counts()
    total = int(counts.sum()) or 1
    head = counts.head(max(1, int(top_n)))
    percentages = [float(c) / total * 100 for c in head.to_numpy()]
    cumulative = np.cumsum(percentages)
    return pd.DataFrame(
        {
            "Değer": [str(v) for v in head.index],
            "Frekans": head.to_numpy(),
            "Yüzde %": [round(p, 2) for p in percentages],
            "Kümülatif %": [round(float(c), 2) for c in cumulative],
        }
    )


def datetime_summary(df: pd.DataFrame, column: str) -> dict[str, Any]:
    """Tarih sütunu için özet üretir.

    Returns:
        min_date, max_date, range_days, count, missing, unique_days,
        by_year / by_month / by_weekday (DataFrame) anahtarları.
    """
    if column not in df.columns:
        raise ValidationError(f"'{column}' sütunu bulunamadı.")
    series = df[column]
    if column not in dfu.datetime_columns(df):
        converted = pd.to_datetime(series, errors="coerce")
        if converted.notna().sum() == 0:
            raise ValidationError(
                f"'{column}' sütunu tarih tipine çevrilemedi. "
                "Veri Temizleme sayfasından dönüştürmeyi deneyin."
            )
        series = converted

    clean = series.dropna()
    if clean.empty:
        raise ValidationError(f"'{column}' sütununda geçerli tarih yok.")

    min_date, max_date = clean.min(), clean.max()
    by_year = (
        clean.dt.year.value_counts().sort_index().rename_axis("Yıl").reset_index(name="Adet")
    )
    by_month = (
        clean.dt.month.value_counts().sort_index().rename_axis("Ay").reset_index(name="Adet")
    )
    by_quarter = (
        clean.dt.quarter.value_counts().sort_index().rename_axis("Çeyrek").reset_index(name="Adet")
    )
    weekday_names = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]
    weekday_counts = clean.dt.dayofweek.value_counts().sort_index()
    by_weekday = pd.DataFrame(
        {
            "Gün": [weekday_names[int(i)] for i in weekday_counts.index],
            "Adet": weekday_counts.to_numpy(),
        }
    )
    return {
        "column": str(column),
        "count": int(clean.shape[0]),
        "missing": int(series.isna().sum()),
        "min_date": min_date,
        "max_date": max_date,
        "range_days": int((max_date - min_date).days),
        "unique_days": int(clean.dt.normalize().nunique()),
        "by_year": by_year,
        "by_quarter": by_quarter,
        "by_month": by_month,
        "by_weekday": by_weekday,
    }


def describe_all_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Tüm sayısal sütunlar için özet tablo üretir (karşılaştırma amaçlı)."""
    records: list[dict[str, Any]] = []
    for col in dfu.numeric_columns(df):
        try:
            summary = numeric_summary(df, col)
        except ValidationError:
            continue
        records.append(
            {
                "Sütun": summary["column"],
                "Dolu": summary["count"],
                "Boş": summary["missing"],
                "Ortalama": round(summary["mean"], 4),
                "Medyan": round(summary["median"], 4),
                "Std": round(summary["std"], 4),
                "Min": round(summary["min"], 4),
                "Max": round(summary["max"], 4),
                "Çarpıklık": round(summary["skewness"], 4)
                if not pd.isna(summary["skewness"])
                else None,
                "Basıklık": round(summary["kurtosis"], 4)
                if not pd.isna(summary["kurtosis"])
                else None,
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=[
            "Sütun",
            "Dolu",
            "Boş",
            "Ortalama",
            "Medyan",
            "Std",
            "Min",
            "Max",
            "Çarpıklık",
            "Basıklık",
        ],
    )


def shapiro_normality(df: pd.DataFrame, column: str, max_sample: int = 5000) -> dict[str, Any] | None:
    """Shapiro-Wilk normallik testi (SciPy varsa).

    Args:
        df: Kaynak DataFrame.
        column: Sayısal sütun.
        max_sample: Testin uygulanacağı maksimum örnek büyüklüğü.

    Returns:
        ``statistic``, ``p_value``, ``sample_size``, ``normal`` anahtarlarını
        içeren sözlük; SciPy yoksa veya örnek yetersizse None.
    """
    if not _HAS_SCIPY:
        return None
    try:
        clean = _clean_numeric(df, column)
    except ValidationError:
        return None
    if len(clean) < 3:
        return None
    sample = clean.sample(max_sample, random_state=0) if len(clean) > max_sample else clean
    try:
        statistic, p_value = scipy_stats.shapiro(sample)
    except (ValueError, TypeError):  # pragma: no cover
        return None
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "sample_size": int(len(sample)),
        "normal": bool(p_value >= 0.05),
    }
