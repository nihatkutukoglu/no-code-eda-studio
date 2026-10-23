"""Korelasyon ve ilişki analizi.

İlk sürümde Pearson ve Spearman desteklenir. Korelasyon **nedensellik
göstermez**; UI katmanı bu uyarıyı her zaman gösterir.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from utils import dataframe_utils as dfu
from utils.helpers import py_literal
from utils.validators import ValidationError, require_columns

__all__ = [
    "CORRELATION_METHODS",
    "DEFAULT_THRESHOLD",
    "correlation_matrix",
    "strong_pairs",
    "pair_correlation",
    "correlation_code",
    "interpret_strength",
    "cramers_v",
    "categorical_association",
    "categorical_association_code",
]

#: Desteklenen korelasyon yöntemleri.
CORRELATION_METHODS: dict[str, str] = {
    "pearson": "Pearson (doğrusal ilişki)",
    "spearman": "Spearman (sıralama tabanlı, monoton ilişki)",
    "kendall": "Kendall tau (sıralama uyumu, küçük örneklere dayanıklı)",
}

#: Güçlü ilişki için varsayılan eşik.
DEFAULT_THRESHOLD = 0.70


def correlation_matrix(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    method: str = "pearson",
    min_periods: int = 2,
) -> pd.DataFrame:
    """Seçilen sayısal sütunlar için korelasyon matrisi üretir.

    Args:
        df: Kaynak DataFrame.
        columns: Kullanılacak sayısal sütunlar (None ise tümü).
        method: ``pearson`` veya ``spearman``.
        min_periods: Korelasyon için gereken minimum ortak gözlem sayısı.

    Returns:
        Simetrik korelasyon matrisi (DataFrame).

    Raises:
        ValidationError: Yöntem desteklenmiyorsa veya en az 2 sayısal sütun yoksa.
    """
    if method not in CORRELATION_METHODS:
        raise ValidationError(f"Desteklenmeyen korelasyon yöntemi: {method}")

    available = dfu.numeric_columns(df)
    targets = [c for c in (columns or available) if c in available]
    missing = [c for c in (columns or []) if c not in available]
    if missing:
        raise ValidationError(
            "Korelasyon yalnızca sayısal sütunlarla hesaplanır. "
            "Sayısal olmayan seçimler: " + ", ".join(map(str, missing))
        )
    if len(targets) < 2:
        raise ValidationError(
            "Korelasyon için en az 2 sayısal sütun seçmelisiniz. "
            f"Şu an {len(targets)} sütun seçili."
        )

    subset = df[targets].apply(pd.to_numeric, errors="coerce")
    subset = subset.replace([np.inf, -np.inf], np.nan)
    # Sabit (tek değerli) sütunlarda korelasyon tanımsızdır; pandas NaN döndürür.
    return subset.corr(method=method, min_periods=min_periods)


def strong_pairs(
    matrix: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD
) -> pd.DataFrame:
    """Eşik üstündeki güçlü ilişkileri tablolar.

    Aynı sütunun kendisiyle korelasyonu (köşegen) ve tekrarlı çiftler
    (A-B / B-A) listeye dahil edilmez.

    Args:
        matrix: ``correlation_matrix`` çıktısı.
        threshold: ``abs(r)`` için alt sınır.

    Returns:
        ``Sütun 1``, ``Sütun 2``, ``Korelasyon (r)``, ``|r|``, ``Yön``, ``Güç``
        kolonlarını içeren DataFrame (|r| azalan sırada).
    """
    limit = abs(float(threshold))
    records: list[dict[str, Any]] = []
    columns = list(matrix.columns)
    for i, first in enumerate(columns):
        for second in columns[i + 1 :]:
            value = matrix.loc[first, second]
            if pd.isna(value):
                continue
            r = float(value)
            if abs(r) < limit:
                continue
            records.append(
                {
                    "Sütun 1": str(first),
                    "Sütun 2": str(second),
                    "Korelasyon (r)": round(r, 4),
                    "|r|": round(abs(r), 4),
                    "Yön": "Pozitif" if r > 0 else "Negatif",
                    "Güç": interpret_strength(r),
                }
            )
    table = pd.DataFrame.from_records(
        records,
        columns=["Sütun 1", "Sütun 2", "Korelasyon (r)", "|r|", "Yön", "Güç"],
    )
    if not table.empty:
        table = table.sort_values("|r|", ascending=False).reset_index(drop=True)
    return table


def interpret_strength(r: float) -> str:
    """Korelasyon katsayısını sözel güç etiketine çevirir."""
    value = abs(float(r))
    if value >= 0.90:
        return "Çok güçlü"
    if value >= 0.70:
        return "Güçlü"
    if value >= 0.50:
        return "Orta"
    if value >= 0.30:
        return "Zayıf"
    return "Çok zayıf"


def pair_correlation(
    df: pd.DataFrame, x: str, y: str, method: str = "pearson"
) -> dict[str, Any]:
    """İki sütun arasındaki korelasyonu hesaplar.

    Returns:
        ``r``, ``method``, ``n`` (ortak gözlem sayısı), ``strength`` anahtarları.
    """
    require_columns(df, [x, y], "İlişki analizi")
    if method not in CORRELATION_METHODS:
        raise ValidationError(f"Desteklenmeyen korelasyon yöntemi: {method}")

    numeric = dfu.numeric_columns(df)
    non_numeric = [c for c in (x, y) if c not in numeric]
    if non_numeric:
        raise ValidationError(
            "İlişki analizi sayısal sütun gerektirir. Sayısal olmayan: "
            + ", ".join(non_numeric)
        )

    pair = df[[x, y]].apply(pd.to_numeric, errors="coerce")
    pair = pair.replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 2:
        raise ValidationError(
            "İki sütunda birlikte dolu olan en az 2 satır bulunamadı."
        )
    r = float(pair[x].corr(pair[y], method=method))
    return {
        "r": r,
        "method": method,
        "n": int(len(pair)),
        "strength": interpret_strength(r) if not pd.isna(r) else "Hesaplanamadı",
    }


def cramers_v(df: pd.DataFrame, first: str, second: str) -> dict[str, Any]:
    """İki kategorik sütun arasındaki ilişkiyi Chi-Square ve Cramér's V ile ölçer.

    Cramér's V 0 ile 1 arasındadır: 0 ilişki yok, 1 tam ilişki.

    Args:
        df: Kaynak DataFrame.
        first: Birinci kategorik sütun.
        second: İkinci kategorik sütun.

    Returns:
        ``v``, ``chi2``, ``p_value``, ``dof``, ``n``, ``strength``,
        ``significant`` anahtarlarını içeren sözlük. SciPy yoksa ``chi2`` ve
        ``p_value`` None döner.

    Raises:
        ValidationError: Sütunlar yoksa, aynıysa veya tablo kurulamıyorsa.
    """
    for column in (first, second):
        if column not in df.columns:
            raise ValidationError(f"'{column}' sütunu bulunamadı.")
    if first == second:
        raise ValidationError("İlişki analizi için farklı iki sütun seçmelisiniz.")

    pair = df[[first, second]].dropna()
    if pair.empty:
        raise ValidationError(
            "İki sütunda birlikte dolu olan satır bulunamadı."
        )
    table = pd.crosstab(pair[first].astype("string"), pair[second].astype("string"))
    if table.shape[0] < 2 or table.shape[1] < 2:
        raise ValidationError(
            "Her iki sütunda da en az 2 farklı değer olmalıdır."
        )

    n = int(table.to_numpy().sum())
    chi2: float | None = None
    p_value: float | None = None
    dof: int | None = None
    try:  # SciPy opsiyonel
        from scipy import stats as scipy_stats

        chi2, p_value, dof, _expected = scipy_stats.chi2_contingency(table.to_numpy())
        chi2, p_value, dof = float(chi2), float(p_value), int(dof)
    except Exception:  # pragma: no cover - SciPy yok veya tablo uygun değil
        observed = table.to_numpy(dtype=float)
        row_totals = observed.sum(axis=1, keepdims=True)
        column_totals = observed.sum(axis=0, keepdims=True)
        expected = row_totals @ column_totals / n
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(expected > 0, (observed - expected) ** 2 / expected, 0.0)
        chi2 = float(terms.sum())

    smaller_dimension = min(table.shape) - 1
    v = float(np.sqrt((chi2 / n) / smaller_dimension)) if chi2 and n and smaller_dimension else 0.0
    v = min(max(v, 0.0), 1.0)
    return {
        "v": v,
        "chi2": chi2,
        "p_value": p_value,
        "dof": dof,
        "n": n,
        "strength": interpret_strength(v),
        "significant": bool(p_value is not None and p_value < 0.05),
        "table": table,
    }


def categorical_association(
    df: pd.DataFrame, columns: list[str] | None = None, threshold: float = 0.30
) -> pd.DataFrame:
    """Kategorik sütun çiftleri için Cramér's V tablosu üretir.

    Args:
        df: Kaynak DataFrame.
        columns: İncelenecek kategorik sütunlar (None ise uygun olanlar seçilir).
        threshold: Listeye alınacak minimum Cramér's V değeri.

    Returns:
        ``Sütun 1``, ``Sütun 2``, ``Cramér's V``, ``p-değeri``, ``Güç``
        kolonlarını içeren DataFrame.
    """
    candidates = [
        column
        for column in (columns or dfu.categorical_columns(df))
        if column in df.columns and 2 <= dfu.safe_nunique(df[column]) <= 50
    ]
    records: list[dict[str, Any]] = []
    for index, first in enumerate(candidates):
        for second in candidates[index + 1 :]:
            try:
                result = cramers_v(df, first, second)
            except ValidationError:
                continue
            if result["v"] < threshold:
                continue
            records.append(
                {
                    "Sütun 1": str(first),
                    "Sütun 2": str(second),
                    "Cramér's V": round(result["v"], 4),
                    "p-değeri": round(result["p_value"], 6)
                    if result["p_value"] is not None
                    else None,
                    "Güç": result["strength"],
                }
            )
    table = pd.DataFrame.from_records(
        records, columns=["Sütun 1", "Sütun 2", "Cramér's V", "p-değeri", "Güç"]
    )
    if not table.empty:
        table = table.sort_values("Cramér's V", ascending=False).reset_index(drop=True)
    return table


def categorical_association_code(first: str, second: str, df_name: str = "df") -> str:
    """Chi-Square / Cramér's V analizinin Python kodunu üretir."""
    return (
        "from scipy import stats" + chr(10) + chr(10) +
        f"table = pd.crosstab({df_name}[{py_literal(first)}], "
        f"{df_name}[{py_literal(second)}])" + chr(10) +
        "chi2, p_value, dof, expected = stats.chi2_contingency(table)" + chr(10) +
        "n = table.to_numpy().sum()" + chr(10) +
        "cramers_v = (chi2 / n / (min(table.shape) - 1)) ** 0.5" + chr(10) +
        "print('Chi-Square:', round(chi2, 3), 'p:', round(p_value, 5))" + chr(10) +
        "print(\"Cramer's V:\", round(cramers_v, 3))"
    )


def correlation_code(
    columns: list[str], method: str = "pearson", df_name: str = "df"
) -> str:
    """Korelasyon analizinin Python kodunu üretir."""
    cols = py_literal(list(columns))
    return (
        f"corr_matrix = {df_name}[{cols}].corr(method={py_literal(method)})\n"
        "print(corr_matrix)\n"
        "fig = px.imshow(corr_matrix, text_auto='.2f', color_continuous_scale='RdBu_r',\n"
        "                zmin=-1, zmax=1, title='Korelasyon Matrisi')\n"
        "fig.show()"
    )
