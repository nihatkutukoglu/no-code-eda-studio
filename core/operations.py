"""Operasyon kaydı, uygulanması, geri alınması (undo) ve replay altyapısı.

Mimari kararlar:
    * ``original_df`` asla değiştirilmez.
    * Her kullanıcı işlemi bir ``Operation`` olarak ``operation_history``
      içine yazılır; kayıt ``id``, ``action``, ``params``, ``description`` ve
      ``python_code`` alanlarını içerir.
    * ``working_df`` istenildiği an ``replay(original_df, history)`` ile
      yeniden üretilebilir. Böylece her adım için DataFrame snapshot'ı
      saklanmaz (RAM dostu) ve undo deterministik olur.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

import pandas as pd

from core import cleaner
from core import transformations as tf
from core.code_generator import build_operation_code
from core.filters import describe_filters
from utils.helpers import format_number, new_id
from utils.validators import CleaningError, ValidationError

__all__ = [
    "ACTION_LABELS",
    "Operation",
    "apply_operation",
    "create_operation",
    "describe_operation",
    "history_table",
    "history_to_dicts",
    "normalize_params",
    "replay",
    "reset_dataframe",
    "undo_last",
]

#: Operasyon adlarının kullanıcıya gösterilecek etiketleri.
ACTION_LABELS: dict[str, str] = {
    "drop_missing_rows": "Boş değerli satırları sil",
    "fill_missing": "Boş değerleri doldur",
    "drop_duplicates": "Tekrarlı satırları sil",
    "rename_column": "Sütunu yeniden adlandır",
    "drop_columns": "Sütun sil",
    "convert_dtype": "Veri tipi dönüştür",
    "drop_outlier_rows": "Aykırı değerleri sil",
    "filter_rows": "Filtre uygula",
    "text_transform": "Metin temizleme",
    "numeric_transform": "Sayısal dönüşüm",
    "encode_categorical": "Kategorik encoding",
    "datetime_features": "Tarih özellikleri üret",
    "arithmetic_feature": "Yeni sütun (aritmetik)",
    "bin_column": "Sayısal sütunu grupla (binning)",
    "sort_rows": "Satırları sırala",
    "duplicate_column": "Sütunu kopyala",
    "reorder_columns": "Sütun sırasını değiştir",
}

#: Her operasyonu uygulayan saf fonksiyonlar.
_APPLIERS: dict[str, Callable[[pd.DataFrame, dict[str, Any]], pd.DataFrame]] = {
    "drop_missing_rows": lambda df, p: cleaner.drop_missing_rows(
        df, p.get("columns"), p.get("how", "any")
    ),
    "fill_missing": lambda df, p: cleaner.fill_missing(
        df, p["column"], p.get("method", "median"), p.get("value")
    ),
    "drop_duplicates": lambda df, p: cleaner.drop_duplicates(
        df, p.get("subset"), p.get("keep", "first")
    ),
    "rename_column": lambda df, p: cleaner.rename_column(df, p["column"], p["new_name"]),
    "drop_columns": lambda df, p: cleaner.drop_columns(df, p["columns"]),
    "convert_dtype": lambda df, p: cleaner.convert_dtype(
        df, p["column"], p["target"], p.get("datetime_format")
    ),
    "drop_outlier_rows": lambda df, p: cleaner.drop_outlier_rows(
        df, p["column"], float(p.get("factor", 1.5)), p.get("method", "iqr")
    ),
    "filter_rows": lambda df, p: cleaner.filter_rows(
        df, p.get("filters") or [], p.get("combine", "and")
    ),
    "text_transform": lambda df, p: tf.text_transform(
        df,
        p["column"],
        p.get("method", "trim"),
        find=p.get("find"),
        replace_with=p.get("replace_with", ""),
        characters=p.get("characters"),
        new_column=p.get("new_column"),
    ),
    "numeric_transform": lambda df, p: tf.numeric_transform(
        df,
        p["column"],
        p.get("method", "standard"),
        decimals=int(p.get("decimals", 2)),
        new_column=p.get("new_column"),
    ),
    "encode_categorical": lambda df, p: tf.encode_categorical(
        df,
        p["column"],
        p.get("method", "label"),
        categories=p.get("categories"),
        drop_first=bool(p.get("drop_first")),
        new_column=p.get("new_column"),
    ),
    "datetime_features": lambda df, p: tf.datetime_features(
        df, p["column"], list(p.get("features") or []), prefix=p.get("prefix")
    ),
    "arithmetic_feature": lambda df, p: tf.arithmetic_feature(
        df, p["left"], p["right"], p.get("operation", "add"), p["new_column"]
    ),
    "bin_column": lambda df, p: tf.bin_column(
        df,
        p["column"],
        p.get("method", "equal_width"),
        bins=int(p.get("bins", 4)),
        edges=p.get("edges"),
        labels=p.get("labels"),
        new_column=p.get("new_column"),
    ),
    "sort_rows": lambda df, p: tf.sort_rows(
        df, list(p.get("columns") or []), p.get("ascending", True)
    ),
    "duplicate_column": lambda df, p: tf.duplicate_column(
        df, p["column"], p.get("new_column")
    ),
    "reorder_columns": lambda df, p: tf.reorder_columns(df, list(p.get("columns") or [])),
}


@dataclass
class Operation:
    """Tek bir veri dönüşümünü temsil eder."""

    action: str
    params: dict[str, Any]
    description: str
    python_code: str
    id: str = field(default_factory=new_id)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    rows_before: int | None = None
    rows_after: int | None = None
    cols_before: int | None = None
    cols_after: int | None = None

    @property
    def label(self) -> str:
        """Operasyonun tipine ait kullanıcı dostu etiket."""
        return ACTION_LABELS.get(self.action, self.action)

    @property
    def row_delta(self) -> int:
        """İşlem sonucu değişen satır sayısı (negatif = silinen)."""
        if self.rows_before is None or self.rows_after is None:
            return 0
        return int(self.rows_after - self.rows_before)

    @property
    def col_delta(self) -> int:
        """İşlem sonucu değişen sütun sayısı."""
        if self.cols_before is None or self.cols_after is None:
            return 0
        return int(self.cols_after - self.cols_before)

    def to_dict(self) -> dict[str, Any]:
        """Session state / export için sözlük temsili."""
        return {
            "id": self.id,
            "action": self.action,
            "params": self.params,
            "description": self.description,
            "python_code": self.python_code,
            "created_at": self.created_at,
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "cols_before": self.cols_before,
            "cols_after": self.cols_after,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Operation":
        """Sözlükten Operation üretir (geri yükleme için)."""
        return cls(
            action=str(data.get("action", "")),
            params=dict(data.get("params") or {}),
            description=str(data.get("description", "")),
            python_code=str(data.get("python_code", "")),
            id=str(data.get("id") or new_id()),
            created_at=str(data.get("created_at") or ""),
            rows_before=data.get("rows_before"),
            rows_after=data.get("rows_after"),
            cols_before=data.get("cols_before"),
            cols_after=data.get("cols_after"),
        )


def normalize_params(
    df: pd.DataFrame, action: str, params: dict[str, Any]
) -> dict[str, Any]:
    """Parametreleri replay sırasında deterministik olacak biçimde normalize eder.

    Örneğin özel değerle doldurmada kullanıcının girdiği metin, sütun tipine
    uygun bir Python değerine çevrilir ve bu değer geçmişe yazılır.

    Args:
        df: Operasyonun uygulanacağı güncel DataFrame.
        action: Operasyon adı.
        params: Ham parametreler.

    Returns:
        Normalize edilmiş yeni parametre sözlüğü.
    """
    clean = {k: v for k, v in (params or {}).items() if v is not None}

    if action == "fill_missing" and clean.get("method") == "value":
        clean["value"] = cleaner.coerce_fill_value(df, clean.get("column"), clean.get("value"))
    if action == "drop_columns":
        clean["columns"] = [str(c) for c in (clean.get("columns") or [])]
    if action == "drop_missing_rows" and clean.get("columns"):
        clean["columns"] = [str(c) for c in clean["columns"]]
    if action == "drop_duplicates" and clean.get("subset"):
        clean["subset"] = [str(c) for c in clean["subset"]]
    if action == "drop_outlier_rows":
        clean["factor"] = float(clean.get("factor", 1.5))
        clean["method"] = str(clean.get("method", "iqr"))
    if action in ("reorder_columns", "sort_rows") and clean.get("columns"):
        clean["columns"] = [str(c) for c in clean["columns"]]
    if action == "datetime_features" and clean.get("features"):
        clean["features"] = [str(f) for f in clean["features"]]
    if action == "bin_column":
        if clean.get("edges"):
            clean["edges"] = sorted(float(edge) for edge in clean["edges"])
        if clean.get("labels"):
            clean["labels"] = [str(label) for label in clean["labels"]]
        if clean.get("bins") is not None:
            clean["bins"] = int(clean["bins"])
    if action == "encode_categorical" and clean.get("categories"):
        clean["categories"] = [str(value) for value in clean["categories"]]
    if action == "numeric_transform" and clean.get("decimals") is not None:
        clean["decimals"] = int(clean["decimals"])
    return clean


def describe_operation(action: str, params: dict[str, Any], df: pd.DataFrame | None = None) -> str:
    """Operasyon için Türkçe açıklama metni üretir.

    Args:
        action: Operasyon adı.
        params: Normalize edilmiş parametreler.
        df: Etkilenen satır sayısını hesaplamak için opsiyonel DataFrame.

    Returns:
        İşlem geçmişinde gösterilecek tek satırlık açıklama.
    """
    params = params or {}
    column = params.get("column")

    if action == "drop_missing_rows":
        columns = params.get("columns") or []
        scope = ", ".join(str(c) for c in columns) if columns else "Tüm sütunlar"
        count = cleaner.missing_row_impact(df, columns, params.get("how", "any")) if df is not None else None
        suffix = f" ({format_number(count)} satır)" if count is not None else ""
        return f"{scope} için boş değer içeren satırlar silindi{suffix}."

    if action == "fill_missing":
        labels = {
            "mean": "ortalama",
            "median": "medyan",
            "mode": "en sık değer (mode)",
            "value": f"özel değer ({params.get('value')!r})",
            "ffill": "önceki değer (ffill)",
            "bfill": "sonraki değer (bfill)",
        }
        method = labels.get(str(params.get("method")), str(params.get("method")))
        count = cleaner.fill_impact(df, column) if df is not None else None
        suffix = f" ({format_number(count)} hücre)" if count is not None else ""
        return f"{column} sütunundaki boş değerler {method} ile dolduruldu{suffix}."

    if action == "drop_duplicates":
        subset = params.get("subset") or []
        scope = ", ".join(str(c) for c in subset) if subset else "Tüm sütunlar"
        count = cleaner.duplicate_impact(df, subset, params.get("keep", "first")) if df is not None else None
        suffix = f" ({format_number(count)} satır)" if count is not None else ""
        return f"{scope} bazında tekrarlı satırlar silindi{suffix}."

    if action == "rename_column":
        return f"{column} sütununun adı '{params.get('new_name')}' olarak değiştirildi."

    if action == "drop_columns":
        cols = ", ".join(str(c) for c in (params.get("columns") or []))
        return f"Sütun(lar) silindi: {cols}."

    if action == "convert_dtype":
        label = cleaner.TYPE_TARGETS.get(str(params.get("target")), str(params.get("target")))
        return f"{column} sütunu {label} tipine dönüştürüldü."

    if action == "drop_outlier_rows":
        factor = params.get("factor", 1.5)
        method = str(params.get("method", "iqr"))
        count = (
            cleaner.outlier_impact(df, column, float(factor), method)
            if df is not None
            else None
        )
        suffix = f" ({format_number(count)} satır)" if count is not None else ""
        label = "Z-Score (|z| >" if method == "zscore" else "IQR (k="
        closing = ")" if method == "zscore" else ")"
        return (
            f"{column} sütununda {label} {factor}{closing} yöntemiyle bulunan "
            f"aykırı değerler silindi{suffix}."
        )

    if action == "filter_rows":
        return "Filtre veri setine uygulandı: " + describe_filters(
            params.get("filters") or [], str(params.get("combine", "and"))
        )

    target = params.get("new_column")

    if action == "text_transform":
        label = tf.TEXT_METHODS.get(str(params.get("method")), str(params.get("method")))
        where = f"'{target}' sütununa yazıldı" if target and target != column else "yerinde güncellendi"
        return f"{column} sütununda metin temizleme ({label}); {where}."

    if action == "numeric_transform":
        label = tf.NUMERIC_METHODS.get(str(params.get("method")), str(params.get("method")))
        where = f"'{target}' sütununa yazıldı" if target and target != column else "yerinde güncellendi"
        return f"{column} sütununa {label} uygulandı; {where}."

    if action == "encode_categorical":
        label = tf.ENCODING_METHODS.get(str(params.get("method")), str(params.get("method")))
        return f"{column} sütunu {label} ile sayısala çevrildi."

    if action == "datetime_features":
        features = ", ".join(
            tf.DATETIME_FEATURES.get(str(f), str(f)) for f in (params.get("features") or [])
        )
        return f"{column} tarih sütunundan yeni özellikler üretildi: {features}."

    if action == "arithmetic_feature":
        label = tf.ARITHMETIC_OPERATIONS.get(
            str(params.get("operation")), str(params.get("operation"))
        )
        return (
            f"'{params.get('new_column')}' sütunu oluşturuldu: "
            f"{params.get('left')} {label} {params.get('right')}."
        )

    if action == "bin_column":
        label = tf.BINNING_METHODS.get(str(params.get("method")), str(params.get("method")))
        name = params.get("new_column") or f"{column}_bin"
        return f"{column} sütunu '{name}' olarak gruplandı ({label})."

    if action == "sort_rows":
        columns = ", ".join(str(c) for c in (params.get("columns") or []))
        ascending = params.get("ascending", True)
        if isinstance(ascending, bool):
            direction = "artan" if ascending else "azalan"
        else:
            direction = ", ".join("artan" if value else "azalan" for value in ascending)
        return f"Satırlar şu sütunlara göre sıralandı: {columns} ({direction})."

    if action == "duplicate_column":
        name = params.get("new_column") or f"{column}_kopya"
        return f"{column} sütununun kopyası '{name}' olarak oluşturuldu."

    if action == "reorder_columns":
        return "Sütun sırası değiştirildi."

    return ACTION_LABELS.get(action, action)


def create_operation(
    df: pd.DataFrame, action: str, params: dict[str, Any]
) -> Operation:
    """Operasyon kaydını (henüz uygulamadan) oluşturur.

    Raises:
        ValidationError: Operasyon tanımlı değilse.
    """
    if action not in _APPLIERS:
        raise ValidationError(f"Bilinmeyen operasyon: {action}")
    normalized = normalize_params(df, action, params)
    return Operation(
        action=action,
        params=normalized,
        description=describe_operation(action, normalized, df),
        python_code=build_operation_code(action, normalized),
    )


def apply_operation(
    df: pd.DataFrame, action: str, params: dict[str, Any]
) -> tuple[pd.DataFrame, Operation]:
    """Operasyonu uygular ve kaydını döndürür.

    Args:
        df: Güncel working DataFrame.
        action: Operasyon adı.
        params: Ham parametreler.

    Returns:
        ``(yeni_df, operation)`` ikilisi.

    Raises:
        ValidationError: Operasyon tanımlı değilse.
        CleaningError: Dönüşüm uygulanamazsa (veri değişmez).
    """
    operation = create_operation(df, action, params)
    applier = _APPLIERS[action]
    try:
        result = applier(df, operation.params)
    except (CleaningError, ValidationError):
        raise
    except Exception as exc:  # beklenmeyen pandas hataları
        raise CleaningError(
            f"İşlem uygulanamadı: {exc}. Veri setiniz değiştirilmedi."
        ) from exc

    if result is None:
        raise CleaningError("İşlem sonuç üretmedi, veri setiniz değiştirilmedi.")

    operation.rows_before, operation.cols_before = int(df.shape[0]), int(df.shape[1])
    operation.rows_after, operation.cols_after = int(result.shape[0]), int(result.shape[1])
    return result, operation


def replay(
    original_df: pd.DataFrame, history: list[dict[str, Any] | Operation]
) -> tuple[pd.DataFrame, list[str]]:
    """Operasyon geçmişini orijinal veri üzerinde sırayla yeniden uygular.

    Undo ve reset bu fonksiyon üzerine kuruludur: snapshot saklanmaz, geçmiş
    yeniden oynatılır.

    Args:
        original_df: Hiç değiştirilmemiş kaynak DataFrame.
        history: Operasyon sözlükleri veya Operation nesneleri.

    Returns:
        ``(working_df, hatalar)``. Hata listesi, uygulanamayan operasyonların
        açıklamalarını içerir; bu operasyonlar atlanır ve kalanlar uygulanır.
    """
    df = original_df.copy(deep=True)
    errors: list[str] = []
    for item in history or []:
        op = item if isinstance(item, Operation) else Operation.from_dict(item)
        applier = _APPLIERS.get(op.action)
        if applier is None:
            errors.append(f"Bilinmeyen operasyon atlandı: {op.action}")
            continue
        try:
            df = applier(df, op.params)
        except Exception as exc:
            errors.append(f"'{op.description or op.action}' yeniden uygulanamadı: {exc}")
    return df, errors


def undo_last(
    original_df: pd.DataFrame, history: list[dict[str, Any]]
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any] | None, list[str]]:
    """Son operasyonu geçmişten çıkarır ve kalanları yeniden oynatır.

    Returns:
        ``(yeni_df, yeni_history, çıkarılan_operasyon, hatalar)``
    """
    if not history:
        return original_df.copy(deep=True), [], None, []
    remaining = list(history[:-1])
    removed = history[-1]
    df, errors = replay(original_df, remaining)
    return df, remaining, removed, errors


def reset_dataframe(original_df: pd.DataFrame) -> pd.DataFrame:
    """Veri setini orijinal haline döndürür."""
    return original_df.copy(deep=True)


def history_to_dicts(history: list[Operation | dict[str, Any]]) -> list[dict[str, Any]]:
    """Operasyon listesini sözlük listesine çevirir."""
    return [op.to_dict() if isinstance(op, Operation) else dict(op) for op in history or []]


def history_table(history: list[dict[str, Any]]) -> pd.DataFrame:
    """İşlem geçmişini tablo olarak döndürür (UI gösterimi için).

    Returns:
        ``#``, ``İşlem``, ``Açıklama``, ``Satır``, ``Sütun``, ``Zaman``
        kolonlarını içeren DataFrame.
    """
    records: list[dict[str, Any]] = []
    for index, item in enumerate(history or [], start=1):
        op = Operation.from_dict(item) if isinstance(item, dict) else item
        records.append(
            {
                "#": index,
                "İşlem": op.label,
                "Açıklama": op.description,
                "Satır": f"{op.rows_before} → {op.rows_after}"
                if op.rows_before is not None
                else "-",
                "Sütun": f"{op.cols_before} → {op.cols_after}"
                if op.cols_before is not None
                else "-",
                "Zaman": (op.created_at or "").replace("T", " "),
            }
        )
    return pd.DataFrame.from_records(
        records, columns=["#", "İşlem", "Açıklama", "Satır", "Sütun", "Zaman"]
    )
