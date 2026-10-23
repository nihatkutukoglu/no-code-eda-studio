"""Genel yardımcı fonksiyonlar: kod üretimi için güvenli literal gösterimi,
sayı/bellek biçimlendirme ve küçük ortak araçlar.

Bu modül Streamlit'e bağımlı değildir; saf Python yardımcıları içerir.
"""

from __future__ import annotations

import math
import uuid
from typing import Any, Iterable

import numpy as np
import pandas as pd

__all__ = [
    "new_id",
    "py_literal",
    "col_ref",
    "col_literal",
    "format_number",
    "format_percent",
    "format_bytes",
    "safe_filename",
    "unique_name",
]


def new_id() -> str:
    """Kısa, çakışma riski düşük bir kimlik üretir."""
    return uuid.uuid4().hex[:12]


def py_literal(value: Any) -> str:
    """Bir Python değerini, üretilen kod içine gömülebilecek literal'e çevirir.

    NumPy ve Pandas tiplerini saf Python karşılıklarına indirger; NaN/Inf
    değerlerini çalıştırılabilir ifadelere (``float('nan')``) dönüştürür.

    Args:
        value: Literal'e çevrilecek değer.

    Returns:
        ``eval`` edilebilir Python kaynak kodu parçası.
    """
    if value is None:
        return "None"
    if isinstance(value, (np.bool_, bool)):
        return "True" if bool(value) else "False"
    if isinstance(value, (np.integer,)):
        return repr(int(value))
    if isinstance(value, (np.floating, float)):
        f = float(value)
        if math.isnan(f):
            return "float('nan')"
        if math.isinf(f):
            return "float('inf')" if f > 0 else "float('-inf')"
        return repr(f)
    if isinstance(value, int):
        return repr(value)
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, pd.Timestamp):
        return f"pd.Timestamp({value.isoformat()!r})"
    if isinstance(value, (list, tuple, set)):
        items = ", ".join(py_literal(v) for v in value)
        if isinstance(value, tuple):
            return f"({items},)" if len(value) == 1 else f"({items})"
        if isinstance(value, set):
            return "set()" if not value else "{" + items + "}"
        return f"[{items}]"
    if isinstance(value, dict):
        inner = ", ".join(f"{py_literal(k)}: {py_literal(v)}" for k, v in value.items())
        return "{" + inner + "}"
    return repr(str(value))


def col_literal(column: Any) -> str:
    """Sütun adını güvenli bir Python string literal'ine çevirir.

    Boşluk, tırnak, ters bölü gibi karakterler ``repr`` sayesinde doğru
    şekilde kaçışlanır.
    """
    return repr(str(column))


def col_ref(column: Any, df_name: str = "df") -> str:
    """``df["Sütun Adı"]`` biçiminde güvenli bir sütun erişimi üretir."""
    return f"{df_name}[{col_literal(column)}]"


def format_number(value: Any, decimals: int = 2) -> str:
    """Sayıları binlik ayracı ile okunabilir biçimde döndürür."""
    if value is None:
        return "-"
    try:
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            return f"{int(value):,}".replace(",", ".")
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(f):
        return "-"
    if math.isinf(f):
        return "∞" if f > 0 else "-∞"
    if float(f).is_integer() and abs(f) < 1e15:
        return f"{int(f):,}".replace(",", ".")
    text = f"{f:,.{decimals}f}"
    # Türkçe biçim: binlik "." ve ondalık ","
    return text.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def format_percent(value: Any, decimals: int = 1) -> str:
    """0-100 aralığındaki bir oranı yüzde metnine çevirir."""
    if value is None:
        return "-"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(f):
        return "-"
    return f"{format_number(round(f, decimals), decimals)}%"


def format_bytes(num_bytes: Any) -> str:
    """Bayt değerini KB/MB/GB olarak biçimlendirir."""
    try:
        size = float(num_bytes)
    except (TypeError, ValueError):
        return "-"
    if math.isnan(size):
        return "-"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{format_number(size, 1)} {unit}"
        size /= 1024
    return f"{format_number(size, 1)} TB"


def safe_filename(name: str, default: str = "dataset") -> str:
    """Dosya adından güvensiz karakterleri temizler."""
    base = str(name or default).strip()
    for sep in ("/", "\\"):
        base = base.split(sep)[-1]
    if "." in base:
        base = base.rsplit(".", 1)[0]
    cleaned = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in base)
    cleaned = cleaned.strip("_") or default
    return cleaned[:60]


def unique_name(candidate: str, existing: Iterable[str]) -> str:
    """Var olan isimlerle çakışmayan bir ad üretir (``ad_2`` gibi)."""
    taken = {str(e) for e in existing}
    if candidate not in taken:
        return candidate
    i = 2
    while f"{candidate}_{i}" in taken:
        i += 1
    return f"{candidate}_{i}"
