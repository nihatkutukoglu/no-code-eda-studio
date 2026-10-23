"""IQR tabanlı aykırı değer (outlier) analizi.

İlk sürümde yalnızca IQR yöntemi desteklenir. Yöntem şeffaftır:
    * Q1 = 1. çeyrek, Q3 = 3. çeyrek, IQR = Q3 - Q1
    * Alt sınır = Q1 - k * IQR, Üst sınır = Q3 + k * IQR  (varsayılan k = 1.5)
Bu sınırların dışındaki değerler aykırı olarak işaretlenir.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from utils import dataframe_utils as dfu
from utils.validators import ValidationError, require_numeric_column

__all__ = [
    "DEFAULT_IQR_FACTOR",
    "DEFAULT_ZSCORE_THRESHOLD",
    "OUTLIER_METHODS",
    "OutlierReport",
    "iqr_bounds",
    "zscore_bounds",
    "detect_outliers",
    "outlier_mask",
    "outlier_summary_table",
]

#: IQR çarpanının varsayılan değeri.
DEFAULT_IQR_FACTOR = 1.5
#: Z-Score yönteminin varsayılan eşiği (|z| > 3).
DEFAULT_ZSCORE_THRESHOLD = 3.0

#: Desteklenen aykırı değer yöntemleri.
OUTLIER_METHODS: dict[str, str] = {
    "iqr": "IQR (çeyrekler açıklığı)",
    "zscore": "Z-Score (standart sapma)",
}


@dataclass
class OutlierReport:
    """Bir sütun için aykırı değer analizi sonucu."""

    column: str
    factor: float
    q1: float
    q3: float
    iqr: float
    lower_bound: float
    upper_bound: float
    count: int
    ratio: float
    total: int
    lower_count: int
    upper_count: int
    mask: pd.Series
    method: str = "iqr"

    @property
    def method_label(self) -> str:
        """Yöntemin kullanıcıya gösterilecek adı."""
        return OUTLIER_METHODS.get(self.method, self.method)

    def as_dict(self) -> dict[str, Any]:
        """Maskeyi dışarıda bırakarak sözlük temsilini döndürür."""
        return {
            "column": self.column,
            "method": self.method,
            "factor": self.factor,
            "q1": self.q1,
            "q3": self.q3,
            "iqr": self.iqr,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "count": self.count,
            "ratio": self.ratio,
            "total": self.total,
            "lower_count": self.lower_count,
            "upper_count": self.upper_count,
        }


def iqr_bounds(
    series: pd.Series, factor: float = DEFAULT_IQR_FACTOR
) -> tuple[float, float, float, float, float]:
    """IQR sınırlarını hesaplar.

    Args:
        series: Sayısal seri.
        factor: IQR çarpanı (1.5 yaygın, 3.0 daha toleranslı).

    Returns:
        ``(q1, q3, iqr, lower_bound, upper_bound)``

    Raises:
        ValidationError: Seri sayısal değilse veya hiç dolu değer yoksa.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    clean = numeric.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        raise ValidationError(
            "Sütunda aykırı değer analizi yapılabilecek dolu sayısal değer yok."
        )
    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    iqr = q3 - q1
    return q1, q3, iqr, q1 - factor * iqr, q3 + factor * iqr


def zscore_bounds(
    series: pd.Series, threshold: float = DEFAULT_ZSCORE_THRESHOLD
) -> tuple[float, float, float, float]:
    """Z-Score yöntemine göre sınırları hesaplar.

    Sınırlar ``ortalama ± eşik × standart sapma`` formülüyle bulunur.

    Args:
        series: Sayısal seri.
        threshold: |z| eşiği (yaygın değer 3).

    Returns:
        ``(mean, std, lower_bound, upper_bound)``

    Raises:
        ValidationError: Dolu değer yoksa veya standart sapma 0 ise.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    clean = numeric.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        raise ValidationError(
            "Sütunda aykırı değer analizi yapılabilecek dolu sayısal değer yok."
        )
    mean = float(clean.mean())
    std = float(clean.std(ddof=0))
    if std == 0:
        raise ValidationError(
            "Sütunun standart sapması 0 (tüm değerler aynı); Z-Score "
            "hesaplanamaz. IQR yöntemini kullanın."
        )
    return mean, std, mean - threshold * std, mean + threshold * std


def outlier_mask(
    df: pd.DataFrame,
    column: str,
    factor: float = DEFAULT_IQR_FACTOR,
    method: str = "iqr",
) -> pd.Series:
    """Aykırı değer satırlarını işaretleyen boolean maske döndürür.

    Args:
        df: Kaynak DataFrame.
        column: Sayısal sütun.
        factor: IQR çarpanı veya Z-Score eşiği.
        method: ``iqr`` veya ``zscore``.
    """
    if column not in df.columns:
        raise ValidationError(f"'{column}' sütunu bulunamadı.")
    if method not in OUTLIER_METHODS:
        raise ValidationError(f"Desteklenmeyen aykırı değer yöntemi: {method}")
    numeric = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if method == "zscore":
        _, _, lower, upper = zscore_bounds(df[column], factor)
    else:
        _, _, _, lower, upper = iqr_bounds(df[column], factor)
    return ((numeric < lower) | (numeric > upper)).fillna(False)


def detect_outliers(
    df: pd.DataFrame,
    column: str,
    factor: float = DEFAULT_IQR_FACTOR,
    method: str = "iqr",
) -> OutlierReport:
    """Bir sütun için tam aykırı değer raporu üretir.

    Args:
        df: Kaynak DataFrame.
        column: Sayısal sütun adı.
        factor: IQR çarpanı (``iqr``) veya |z| eşiği (``zscore``).
        method: ``iqr`` veya ``zscore``.

    Returns:
        İstatistikleri ve boolean maskeyi içeren OutlierReport.

    Raises:
        ValidationError: Sütun yok, sayısal değil veya yöntem geçersizse.
    """
    column = require_numeric_column(df, column, "Aykırı değer analizi")
    if method not in OUTLIER_METHODS:
        raise ValidationError(f"Desteklenmeyen aykırı değer yöntemi: {method}")
    if float(factor) <= 0:
        raise ValidationError(
            "IQR çarpanı / Z-Score eşiği 0'dan büyük olmalıdır."
        )

    q1, q3, iqr, lower, upper = iqr_bounds(df[column], factor)
    if method == "zscore":
        _, _, lower, upper = zscore_bounds(df[column], factor)

    numeric = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    below = (numeric < lower).fillna(False)
    above = (numeric > upper).fillna(False)
    mask = below | above
    total = int(numeric.notna().sum())
    count = int(mask.sum())
    return OutlierReport(
        column=str(column),
        factor=float(factor),
        method=str(method),
        q1=q1,
        q3=q3,
        iqr=iqr,
        lower_bound=lower,
        upper_bound=upper,
        count=count,
        ratio=(count / total * 100) if total else 0.0,
        total=total,
        lower_count=int(below.sum()),
        upper_count=int(above.sum()),
        mask=mask,
    )


def outlier_summary_table(
    df: pd.DataFrame, columns: list[str] | None = None, factor: float = DEFAULT_IQR_FACTOR
) -> pd.DataFrame:
    """Birden fazla sayısal sütun için aykırı değer özet tablosu üretir.

    Args:
        df: Kaynak DataFrame.
        columns: İncelenecek sütunlar (None ise tüm sayısal sütunlar).
        factor: IQR çarpanı.

    Returns:
        Sütun bazında aykırı değer sayısı ve oranını içeren DataFrame.
    """
    targets = columns if columns is not None else dfu.numeric_columns(df)
    records: list[dict[str, Any]] = []
    for col in targets:
        try:
            report = detect_outliers(df, col, factor)
        except ValidationError:
            continue
        records.append(
            {
                "Sütun": report.column,
                "Q1": round(report.q1, 4),
                "Q3": round(report.q3, 4),
                "IQR": round(report.iqr, 4),
                "Alt Sınır": round(report.lower_bound, 4),
                "Üst Sınır": round(report.upper_bound, 4),
                "Aykırı Sayısı": report.count,
                "Aykırı %": round(report.ratio, 2),
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=[
            "Sütun",
            "Q1",
            "Q3",
            "IQR",
            "Alt Sınır",
            "Üst Sınır",
            "Aykırı Sayısı",
            "Aykırı %",
        ],
    )
