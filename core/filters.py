"""No-code filtreleme ve group by / aggregation motoru.

Filtreler ``{"column", "operator", "value", "value2"}`` biçiminde sözlüklerle
tanımlanır ve AND mantığıyla birleştirilir. Her filtrenin Python kod karşılığı
üretilebilir; böylece Code Generator ve Notebook Export aynı mantığı yeniden
kullanır.

Semantik kural: değer karşılaştırması yapan tüm operatörler (``!=`` ve
``içermez`` dahil) boş (NaN) değerleri **sonuca dahil etmez**. Boş değerlerle
çalışmak için ``boş`` / ``boş değil`` operatörleri kullanılır. Böylece
"gelir != 2000" filtresi, geliri bilinmeyen satırları getirmez.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from utils import dataframe_utils as dfu
from utils.helpers import col_literal, col_ref, format_number, py_literal
from utils.validators import FilterError, ValidationError

__all__ = [
    "NUMERIC_OPERATORS",
    "TEXT_OPERATORS",
    "DATETIME_OPERATORS",
    "COMBINE_MODES",
    "AGGREGATIONS",
    "operators_for_column",
    "needs_second_value",
    "needs_value",
    "build_mask",
    "apply_filters",
    "describe_filter",
    "describe_filters",
    "filter_code",
    "filters_to_code",
    "apply_groupby",
    "groupby_code",
    "filter_impact",
]

#: Sayısal sütunlar için operatörler: ``kod -> etiket``.
NUMERIC_OPERATORS: dict[str, str] = {
    "eq": "= (eşit)",
    "ne": "!= (eşit değil)",
    "gt": "> (büyük)",
    "gte": ">= (büyük veya eşit)",
    "lt": "< (küçük)",
    "lte": "<= (küçük veya eşit)",
    "between": "arasında (between)",
    "is_null": "boş (NaN)",
    "not_null": "boş değil",
}

#: Metin / kategorik sütunlar için operatörler.
TEXT_OPERATORS: dict[str, str] = {
    "equals": "eşittir",
    "not_equals": "eşit değildir",
    "contains": "içerir",
    "not_contains": "içermez",
    "starts_with": "ile başlar",
    "ends_with": "ile biter",
    "is_in": "şunlardan biri (is in)",
    "not_in": "şunlardan biri değil",
    "is_null": "boş",
    "not_null": "boş değil",
}

#: Filtrelerin birleştirme mantığı.
COMBINE_MODES: dict[str, str] = {
    "and": "VE (tüm koşullar sağlanmalı)",
    "or": "VEYA (en az bir koşul sağlanmalı)",
}

#: Tarih sütunları için operatörler.
DATETIME_OPERATORS: dict[str, str] = {
    "before": "öncesinde (before)",
    "after": "sonrasında (after)",
    "between": "arasında (between)",
    "is_null": "boş",
    "not_null": "boş değil",
}

#: Desteklenen aggregation fonksiyonları: ``kod -> etiket``.
AGGREGATIONS: dict[str, str] = {
    "count": "count (adet)",
    "sum": "sum (toplam)",
    "mean": "mean (ortalama)",
    "median": "median (medyan)",
    "min": "min (en küçük)",
    "max": "max (en büyük)",
}

_TWO_VALUE_OPERATORS = {"between"}
_NO_VALUE_OPERATORS = {"is_null", "not_null"}
_MULTI_VALUE_OPERATORS = {"is_in", "not_in"}


def operators_for_column(df: pd.DataFrame, column: str) -> dict[str, str]:
    """Sütunun tipine uygun operatör sözlüğünü döndürür."""
    if column not in df.columns:
        raise FilterError(f"'{column}' sütunu bulunamadı.")
    kind = dfu.column_kind(df[column])
    if kind == "numeric":
        return dict(NUMERIC_OPERATORS)
    if kind == "datetime":
        return dict(DATETIME_OPERATORS)
    return dict(TEXT_OPERATORS)


def needs_value(operator: str) -> bool:
    """Operatörün değer girişi gerektirip gerektirmediğini söyler."""
    return operator not in _NO_VALUE_OPERATORS


def needs_second_value(operator: str) -> bool:
    """Operatörün ikinci bir değer (between üst sınırı) gerektirip gerektirmediği."""
    return operator in _TWO_VALUE_OPERATORS


def _as_number(value: Any, label: str) -> float:
    """Değeri sayıya çevirir."""
    if value is None or (isinstance(value, str) and not value.strip()):
        raise FilterError(f"{label} için bir sayı girmelisiniz.")
    try:
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError) as exc:
        raise FilterError(f"{label} sayıya çevrilemedi: {value!r}") from exc


def _as_timestamp(value: Any, label: str) -> pd.Timestamp:
    """Değeri Timestamp'e çevirir."""
    if value is None or (isinstance(value, str) and not value.strip()):
        raise FilterError(f"{label} için bir tarih seçmelisiniz.")
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise FilterError(f"{label} tarihe çevrilemedi: {value!r}") from exc
    if pd.isna(ts):
        raise FilterError(f"{label} geçerli bir tarih değil: {value!r}")
    return ts


def _as_list(value: Any, label: str) -> list[Any]:
    """Değeri listeye çevirir (virgülle ayrılmış metin de kabul edilir)."""
    if value is None:
        raise FilterError(f"{label} için en az bir değer seçmelisiniz.")
    if isinstance(value, (list, tuple, set, pd.Series, np.ndarray)):
        items = [v for v in list(value)]
    else:
        items = [part.strip() for part in str(value).split(",")]
    items = [v for v in items if not (isinstance(v, str) and v == "")]
    if not items:
        raise FilterError(f"{label} için en az bir değer seçmelisiniz.")
    return items


def build_mask(df: pd.DataFrame, flt: dict[str, Any]) -> pd.Series:
    """Tek bir filtre için boolean maske üretir.

    Args:
        df: Filtrenin uygulanacağı DataFrame.
        flt: ``column``, ``operator``, ``value``, ``value2`` içeren sözlük.

    Returns:
        DataFrame indeksiyle hizalı boolean Series.

    Raises:
        FilterError: Sütun/operatör/değer geçersizse.
    """
    column = flt.get("column")
    operator = str(flt.get("operator") or "")
    if column not in df.columns:
        raise FilterError(
            f"'{column}' sütunu veri setinde yok. Filtre güncel olmayabilir."
        )
    series = df[column]
    kind = dfu.column_kind(series)
    label = f"'{column}' filtresi"

    if operator == "is_null":
        return series.isna()
    if operator == "not_null":
        return series.notna()

    allowed = operators_for_column(df, column)
    if operator not in allowed:
        raise FilterError(
            f"'{operator}' operatörü '{column}' sütunu ({kind}) için uygun değil."
        )

    # Boş değerler hiçbir değer karşılaştırmasına dahil edilmez.
    not_null = series.notna()

    if kind == "numeric":
        if operator == "between":
            low = _as_number(flt.get("value"), f"{label} alt sınırı")
            high = _as_number(flt.get("value2"), f"{label} üst sınırı")
            if low > high:
                low, high = high, low
            return series.between(low, high) & not_null
        number = _as_number(flt.get("value"), label)
        comparison = {
            "eq": lambda: series == number,
            "ne": lambda: series != number,
            "gt": lambda: series > number,
            "gte": lambda: series >= number,
            "lt": lambda: series < number,
            "lte": lambda: series <= number,
        }[operator]()
        return comparison & not_null

    if kind == "datetime":
        if operator == "between":
            start = _as_timestamp(flt.get("value"), f"{label} başlangıcı")
            end = _as_timestamp(flt.get("value2"), f"{label} bitişi")
            if start > end:
                start, end = end, start
            return series.between(start, end) & not_null
        ts = _as_timestamp(flt.get("value"), label)
        comparison = series < ts if operator == "before" else series > ts
        return comparison & not_null

    # Metin / kategorik / mantıksal
    text = series.astype(str)
    if operator in _MULTI_VALUE_OPERATORS:
        values = [str(v) for v in _as_list(flt.get("value"), label)]
        mask = text.isin(values)
        mask = ~mask if operator == "not_in" else mask
        return mask & not_null
    needle = flt.get("value")
    if needle is None or (isinstance(needle, str) and needle == ""):
        raise FilterError(f"{label} için bir değer girmelisiniz.")
    needle = str(needle)
    if operator == "equals":
        return (text == needle) & not_null
    if operator == "not_equals":
        return (text != needle) & not_null
    if operator == "contains":
        return text.str.contains(needle, case=False, na=False, regex=False) & not_null
    if operator == "starts_with":
        return text.str.lower().str.startswith(needle.lower(), na=False) & not_null
    if operator == "ends_with":
        return text.str.lower().str.endswith(needle.lower(), na=False) & not_null
    return ~text.str.contains(needle, case=False, na=False, regex=False) & not_null


def apply_filters(
    df: pd.DataFrame, filters: list[dict[str, Any]], combine: str = "and"
) -> pd.DataFrame:
    """Filtreleri AND veya OR mantığıyla uygular ve yeni bir DataFrame döndürür.

    Args:
        df: Kaynak DataFrame (değiştirilmez).
        filters: Filtre sözlüklerinin listesi.
        combine: ``and`` (tümü) veya ``or`` (en az biri).

    Returns:
        Filtrelenmiş yeni DataFrame.

    Raises:
        FilterError: Herhangi bir filtre geçersizse veya mantık bilinmiyorsa.
    """
    if not filters:
        return df
    if combine not in COMBINE_MODES:
        raise FilterError(f"Desteklenmeyen birleştirme mantığı: {combine}")

    use_and = combine == "and"
    mask = pd.Series(use_and, index=df.index)
    for flt in filters:
        current = build_mask(df, flt).fillna(False)
        mask = (mask & current) if use_and else (mask | current)
    return df.loc[mask].copy()


def describe_filter(flt: dict[str, Any]) -> str:
    """Filtreyi insan tarafından okunabilir Türkçe metne çevirir."""
    column = flt.get("column")
    operator = str(flt.get("operator") or "")
    labels = {**NUMERIC_OPERATORS, **TEXT_OPERATORS, **DATETIME_OPERATORS}
    op_label = labels.get(operator, operator)
    if operator in _NO_VALUE_OPERATORS:
        return f"{column} {op_label}"
    if operator in _TWO_VALUE_OPERATORS:
        return f"{column} {op_label}: {flt.get('value')} - {flt.get('value2')}"
    value = flt.get("value")
    if isinstance(value, (list, tuple, set)):
        value = ", ".join(str(v) for v in value)
    return f"{column} {op_label} {value}"


def describe_filters(filters: list[dict[str, Any]], combine: str = "and") -> str:
    """Tüm filtrelerin özetini tek satırda döndürür."""
    if not filters:
        return "Filtre yok"
    joiner = " VE " if combine == "and" else " VEYA "
    return joiner.join(describe_filter(f) for f in filters)


def filter_code(flt: dict[str, Any], df_name: str = "df") -> str:
    """Tek filtre için boolean maske üreten Python kodu döndürür."""
    column = flt.get("column")
    operator = str(flt.get("operator") or "")
    ref = col_ref(column, df_name)

    if operator == "is_null":
        return f"{ref}.isna()"
    if operator == "not_null":
        return f"{ref}.notna()"

    # Olumsuz operatörlerde boş değerlerin dışlanması koda da yansıtılır.
    not_null = f"{ref}.notna()"

    symbols = {"eq": "==", "ne": "!=", "gt": ">", "gte": ">=", "lt": "<", "lte": "<="}
    if operator in symbols:
        comparison = f"({ref} {symbols[operator]} {py_literal(flt.get('value'))})"
        return f"({comparison} & {not_null})" if operator == "ne" else comparison
    if operator == "between":
        value, value2 = flt.get("value"), flt.get("value2")
        return f"{ref}.between({py_literal(value)}, {py_literal(value2)})"
    if operator == "before":
        return f"({ref} < pd.Timestamp({col_literal(flt.get('value'))}))"
    if operator == "after":
        return f"({ref} > pd.Timestamp({col_literal(flt.get('value'))}))"
    if operator == "equals":
        return f"({ref}.astype(str) == {py_literal(str(flt.get('value')))})"
    if operator == "not_equals":
        return (
            f"(({ref}.astype(str) != {py_literal(str(flt.get('value')))}) & {not_null})"
        )
    if operator == "contains":
        return (
            f"{ref}.astype(str).str.contains({py_literal(str(flt.get('value')))}, "
            "case=False, na=False, regex=False)"
        )
    if operator == "starts_with":
        needle = py_literal(str(flt.get("value")).lower())
        return (
            f"({ref}.astype(str).str.lower().str.startswith({needle}, na=False)"
            f" & {not_null})"
        )
    if operator == "ends_with":
        needle = py_literal(str(flt.get("value")).lower())
        return (
            f"({ref}.astype(str).str.lower().str.endswith({needle}, na=False)"
            f" & {not_null})"
        )
    if operator == "not_contains":
        return (
            f"(~{ref}.astype(str).str.contains({py_literal(str(flt.get('value')))}, "
            f"case=False, na=False, regex=False) & {not_null})"
        )
    if operator in _MULTI_VALUE_OPERATORS:
        values = [str(v) for v in _as_list(flt.get("value"), "Filtre")]
        code = f"{ref}.astype(str).isin({py_literal(values)})"
        return f"(~{code} & {not_null})" if operator == "not_in" else code
    raise FilterError(f"'{operator}' operatörü için kod üretilemedi.")


def filters_to_code(
    filters: list[dict[str, Any]],
    df_name: str = "df",
    target: str | None = None,
    combine: str = "and",
) -> str:
    """Filtre listesi için tam Python kodu üretir.

    Args:
        filters: Filtre sözlükleri.
        df_name: Kaynak DataFrame değişken adı.
        target: Sonucun atanacağı değişken adı (None ise ``df_name``).
        combine: ``and`` veya ``or``.

    Returns:
        ``df = df[mask]`` biçiminde kod; filtre yoksa boş metin.
    """
    if not filters:
        return ""
    out = target or df_name
    operator = "&" if combine == "and" else "|"
    masks = [filter_code(f, df_name) for f in filters]
    if len(masks) == 1:
        return f"{out} = {df_name}[{masks[0]}]"
    joined = f" {operator} ".join(f"({m})" for m in masks)
    if len(joined) <= 88:
        return f"{out} = {df_name}[{joined}]"
    lines = [
        "mask = (",
        *[
            f"    {m}" if index == 0 else f"    {operator} {m}"
            for index, m in enumerate(masks)
        ],
        ")",
    ]
    return "\n".join(lines) + f"\n{out} = {df_name}[mask]"


def apply_groupby(
    df: pd.DataFrame,
    group_by: list[str],
    value_columns: list[str],
    aggregation: str,
) -> pd.DataFrame:
    """Group by / aggregation uygular.

    Args:
        df: Kaynak DataFrame.
        group_by: Gruplanacak sütun(lar).
        value_columns: Toplanacak değer sütunları. ``count`` için boş olabilir.
        aggregation: ``AGGREGATIONS`` içindeki bir anahtar.

    Returns:
        Düz (reset_index yapılmış) sonuç DataFrame'i.

    Raises:
        ValidationError: Seçimler geçersizse.
    """
    if not group_by:
        raise ValidationError("Group by için en az bir sütun seçmelisiniz.")
    if aggregation not in AGGREGATIONS:
        raise ValidationError(f"Desteklenmeyen aggregation: {aggregation}")
    missing = [c for c in list(group_by) + list(value_columns) if c not in df.columns]
    if missing:
        raise ValidationError("Şu sütunlar bulunamadı: " + ", ".join(map(str, missing)))

    if aggregation == "count" and not value_columns:
        result = df.groupby(group_by, dropna=False).size().reset_index(name="count")
        return result

    if not value_columns:
        raise ValidationError(
            f"'{aggregation}' için en az bir değer sütunu seçmelisiniz."
        )

    if aggregation in ("sum", "mean", "median"):
        non_numeric = [c for c in value_columns if c not in dfu.numeric_columns(df)]
        if non_numeric:
            raise ValidationError(
                f"'{aggregation}' yalnızca sayısal sütunlarla çalışır. "
                "Sayısal olmayan seçimler: " + ", ".join(non_numeric)
            )

    try:
        grouped = df.groupby(group_by, dropna=False)[value_columns].agg(aggregation)
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValidationError(
            f"Group by işlemi uygulanamadı: {exc}. Sütun tiplerini kontrol edin."
        ) from exc
    return grouped.reset_index()


def groupby_code(
    group_by: list[str],
    value_columns: list[str],
    aggregation: str,
    df_name: str = "df",
    target: str = "grouped_df",
) -> str:
    """Group by işleminin Python kodunu üretir."""
    if aggregation == "count" and not value_columns:
        return (
            f"{target} = {df_name}.groupby({py_literal(list(group_by))}, dropna=False)"
            f".size().reset_index(name='count')"
        )
    return (
        f"{target} = ({df_name}.groupby({py_literal(list(group_by))}, dropna=False)"
        f"[{py_literal(list(value_columns))}]"
        f".agg({py_literal(aggregation)}).reset_index())"
    )


def filter_impact(
    df: pd.DataFrame, filters: list[dict[str, Any]], combine: str = "and"
) -> str:
    """Filtrenin kaç satır bırakacağını özetleyen metin döndürür."""
    try:
        kept = len(apply_filters(df, filters, combine))
    except FilterError as exc:
        return str(exc)
    total = len(df)
    ratio = (kept / total * 100) if total else 0
    return (
        f"{format_number(kept)} / {format_number(total)} satır kalır "
        f"({format_number(ratio, 1)}%)"
    )
