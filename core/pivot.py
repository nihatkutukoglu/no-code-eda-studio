"""Pivot table ve crosstab üretimi (no-code).

Pivot/crosstab sonuçları ``working_df``'i değiştirmez; analiz çıktısı olarak
tablo, grafik ve CSV export için kullanılır. Her sonucun Python kod karşılığı
üretilir ve Notebook export'una eklenebilir.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from utils import dataframe_utils as dfu
from utils.helpers import py_literal
from utils.validators import ValidationError

__all__ = [
    "PIVOT_AGGREGATIONS",
    "CROSSTAB_NORMALIZE",
    "build_pivot",
    "pivot_code",
    "build_crosstab",
    "crosstab_code",
    "flatten_columns",
]

#: Pivot/GroupBy için desteklenen aggregation fonksiyonları.
PIVOT_AGGREGATIONS: dict[str, str] = {
    "count": "count (adet)",
    "nunique": "nunique (tekil adet)",
    "sum": "sum (toplam)",
    "mean": "mean (ortalama)",
    "median": "median (medyan)",
    "min": "min (en küçük)",
    "max": "max (en büyük)",
    "std": "std (standart sapma)",
}

#: Crosstab yüzde seçenekleri: ``kod -> (etiket, pandas normalize değeri)``
CROSSTAB_NORMALIZE: dict[str, tuple[str, Any]] = {
    "count": ("Adet", False),
    "row": ("Satır yüzdesi (%)", "index"),
    "column": ("Sütun yüzdesi (%)", "columns"),
    "total": ("Toplam yüzdesi (%)", "all"),
}

#: Sayısal olmayan sütunlarda kullanılamayan aggregation'lar.
_NUMERIC_ONLY = ("sum", "mean", "median", "std")


def flatten_columns(df: pd.DataFrame, separator: str = " | ") -> pd.DataFrame:
    """MultiIndex sütunlarını tek seviyeye indirir (gösterim/export için)."""
    result = df.copy()
    if isinstance(result.columns, pd.MultiIndex):
        result.columns = [
            separator.join(str(part) for part in parts if str(part) != "")
            for parts in result.columns
        ]
    else:
        result.columns = [str(column) for column in result.columns]
    return result


def build_pivot(
    df: pd.DataFrame,
    rows: list[str],
    columns: list[str] | None,
    values: str | None,
    aggregation: str = "mean",
    fill_value: float | None = None,
    margins: bool = False,
) -> pd.DataFrame:
    """Pivot table üretir.

    Args:
        df: Kaynak DataFrame.
        rows: Satır (index) sütunları.
        columns: Sütun (column) kırılımı; boş olabilir.
        values: Değer sütunu. ``count``/``nunique`` dışında zorunludur.
        aggregation: ``PIVOT_AGGREGATIONS`` içindeki bir anahtar.
        fill_value: Boş hücreler için doldurma değeri (None ise NaN kalır).
        margins: Satır/sütun toplamlarını ekle.

    Returns:
        Düzleştirilmiş sütun adlarıyla pivot sonucu.

    Raises:
        ValidationError: Seçimler geçersizse veya sonuç üretilemezse.
    """
    if not rows:
        raise ValidationError("Pivot için en az bir satır (Rows) sütunu seçmelisiniz.")
    if aggregation not in PIVOT_AGGREGATIONS:
        raise ValidationError(f"Desteklenmeyen aggregation: {aggregation}")

    columns = [c for c in (columns or []) if c]
    selected = [*rows, *columns] + ([values] if values else [])
    missing = [c for c in selected if c not in df.columns]
    if missing:
        raise ValidationError("Şu sütunlar bulunamadı: " + ", ".join(map(str, missing)))

    overlap = set(rows) & set(columns)
    if overlap:
        raise ValidationError(
            "Aynı sütun hem satır hem sütun kırılımı olarak kullanılamaz: "
            + ", ".join(sorted(overlap))
        )

    if values is None:
        if aggregation not in ("count", "nunique"):
            raise ValidationError(
                f"'{aggregation}' için bir değer (Values) sütunu seçmelisiniz."
            )
        # Değer sütunu yoksa satır sayısını sayarız.
        working = df.assign(_adet=1)
        values, aggregation_fn = "_adet", "count"
    else:
        working = df
        aggregation_fn = aggregation
        if aggregation in _NUMERIC_ONLY and values not in dfu.numeric_columns(df):
            raise ValidationError(
                f"'{aggregation}' yalnızca sayısal sütunlarla çalışır. "
                f"'{values}' sayısal değil; count veya nunique deneyin."
            )

    try:
        table = pd.pivot_table(
            working,
            index=rows,
            columns=columns or None,
            values=values,
            aggfunc=aggregation_fn,
            fill_value=fill_value,
            margins=bool(margins),
            margins_name="Toplam",
            observed=True,
            dropna=False,
        )
    except (TypeError, ValueError, KeyError) as exc:
        raise ValidationError(
            f"Pivot table oluşturulamadı: {exc}. Sütun tiplerini ve "
            "aggregation seçiminizi kontrol edin."
        ) from exc

    if isinstance(table, pd.Series):
        table = table.to_frame(name=values)
    if table.empty:
        raise ValidationError(
            "Pivot sonucu boş. Filtreleri gevşetmeyi veya farklı sütunlar "
            "seçmeyi deneyin."
        )
    result = table.reset_index()
    result = flatten_columns(result)
    if "_adet" in result.columns:
        result = result.rename(columns={"_adet": "adet"})
    return result


def pivot_code(
    rows: list[str],
    columns: list[str] | None,
    values: str | None,
    aggregation: str = "mean",
    fill_value: float | None = None,
    margins: bool = False,
    df_name: str = "df",
    target: str = "pivot_df",
) -> str:
    """Pivot table işleminin Python kodunu üretir."""
    columns = [c for c in (columns or []) if c]
    if values is None:
        lines = [f"pivot_source = {df_name}.assign(_adet=1)"]
        source, value_column, aggregation_fn = "pivot_source", "_adet", "count"
    else:
        lines = []
        source, value_column, aggregation_fn = df_name, values, aggregation

    args = [
        source,
        f"index={py_literal(list(rows))}",
        f"columns={py_literal(columns)}" if columns else None,
        f"values={py_literal(value_column)}",
        f"aggfunc={py_literal(aggregation_fn)}",
        f"fill_value={py_literal(fill_value)}" if fill_value is not None else None,
        "margins=True, margins_name='Toplam'" if margins else None,
        "observed=True",
    ]
    call = "pd.pivot_table(" + ", ".join(arg for arg in args if arg) + ")"
    lines.append(f"{target} = {call}.reset_index()")
    return "\n".join(lines)


def build_crosstab(
    df: pd.DataFrame,
    row_column: str,
    column_column: str,
    normalize: str = "count",
    margins: bool = True,
    decimals: int = 1,
) -> pd.DataFrame:
    """İki kategorik sütun arasında crosstab üretir.

    Args:
        df: Kaynak DataFrame.
        row_column: Satır sütunu.
        column_column: Sütun sütunu.
        normalize: ``count``, ``row``, ``column`` veya ``total``.
        margins: Toplam satırı/sütunu ekle.
        decimals: Yüzde gösteriminde ondalık basamak.

    Returns:
        Crosstab sonucu (yüzde seçildiyse 0-100 aralığında).

    Raises:
        ValidationError: Seçimler geçersizse.
    """
    if normalize not in CROSSTAB_NORMALIZE:
        raise ValidationError(f"Desteklenmeyen yüzde seçeneği: {normalize}")
    for column in (row_column, column_column):
        if column not in df.columns:
            raise ValidationError(f"'{column}' sütunu bulunamadı.")
    if row_column == column_column:
        raise ValidationError("Satır ve sütun için farklı sütunlar seçmelisiniz.")

    for column in (row_column, column_column):
        nunique = dfu.safe_nunique(df[column])
        if nunique > 100:
            raise ValidationError(
                f"'{column}' sütununda {nunique} farklı değer var. Crosstab "
                "okunamaz hale gelir; daha az kategorili bir sütun seçin veya "
                "önce binning uygulayın."
            )

    _, normalize_value = CROSSTAB_NORMALIZE[normalize]
    try:
        table = pd.crosstab(
            df[row_column].astype("string"),
            df[column_column].astype("string"),
            normalize=normalize_value,
            margins=bool(margins),
            margins_name="Toplam",
            dropna=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"Crosstab oluşturulamadı: {exc}") from exc

    if normalize != "count":
        table = (table * 100).round(int(decimals))
    result = table.reset_index()
    return flatten_columns(result)


def crosstab_code(
    row_column: str,
    column_column: str,
    normalize: str = "count",
    margins: bool = True,
    df_name: str = "df",
    target: str = "crosstab_df",
) -> str:
    """Crosstab işleminin Python kodunu üretir."""
    _, normalize_value = CROSSTAB_NORMALIZE.get(normalize, ("", False))
    args = [
        f"{df_name}[{py_literal(row_column)}]",
        f"{df_name}[{py_literal(column_column)}]",
        f"normalize={py_literal(normalize_value)}" if normalize_value else None,
        "margins=True, margins_name='Toplam'" if margins else None,
    ]
    call = "pd.crosstab(" + ", ".join(arg for arg in args if arg) + ")"
    if normalize != "count":
        return f"{target} = ({call} * 100).round(1).reset_index()"
    return f"{target} = {call}.reset_index()"


def chart_ready_pivot(table: pd.DataFrame, max_columns: int = 12) -> pd.DataFrame:
    """Pivot sonucunu grafiğe uygun hale getirir (uzun formata çevirir).

    Args:
        table: ``build_pivot`` çıktısı.
        max_columns: Grafiğe dönüştürülecek maksimum değer sütunu sayısı.

    Returns:
        ``kategori``, ``seri``, ``deger`` kolonlarından oluşan uzun format tablo.
    """
    if table.empty or table.shape[1] < 2:
        raise ValidationError("Grafiğe dönüştürülecek pivot sonucu yok.")
    index_column = table.columns[0]
    value_columns = [
        column
        for column in table.columns[1:]
        if pd.api.types.is_numeric_dtype(table[column])
    ][:max_columns]
    if not value_columns:
        raise ValidationError(
            "Pivot sonucunda grafiğe uygun sayısal sütun bulunamadı."
        )
    melted = table.melt(
        id_vars=[index_column],
        value_vars=value_columns,
        var_name="seri",
        value_name="deger",
    )
    melted = melted.rename(columns={index_column: "kategori"})
    melted["kategori"] = melted["kategori"].astype(str)
    melted["deger"] = pd.to_numeric(melted["deger"], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    return melted
