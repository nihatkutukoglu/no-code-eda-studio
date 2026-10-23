"""No-code işlemlerin Python/Pandas kod karşılığını üreten modül.

İki sorumluluğu var:
    1. Tek bir operasyonun kod satır(lar)ını üretmek (``build_operation_code``).
       Bu kod, operasyon kaydedilirken ``operation_history`` içine yazılır.
    2. Tüm oturumu çalıştırılabilir tek bir script haline getirmek
       (``generate_full_script``) — Code Viewer ve ``.py`` / ``.ipynb`` export
       aynı fonksiyonu kullanır.

Üretilen kod, veri değerlerini gömmez; yalnızca yapılan işlemleri tarif eder.
Sütun adları ``repr`` ile kaçışlandığı için boşluk/tırnak içeren adlar güvenlidir.
"""

from __future__ import annotations

from typing import Any

from core.filters import filters_to_code
from core.loader import loading_code
from utils.helpers import col_literal, col_ref, py_literal
from utils.validators import ValidationError

__all__ = [
    "ACTION_CATEGORIES",
    "CODE_SECTIONS",
    "action_category",
    "filter_history",
    "build_operation_code",
    "generate_imports",
    "generate_cleaning_section",
    "generate_charts_section",
    "generate_full_script",
    "generate_notes_section",
]

#: Kod/notebook bölümlerinin başlıkları.
CODE_SECTIONS: dict[str, str] = {
    "cleaning": "Veri Temizleme",
    "transformation": "Dönüşümler",
    "feature": "Feature Engineering",
    "arrangement": "Sıralama ve Düzen",
}

#: Her operasyonun hangi bölüme ait olduğu.
ACTION_CATEGORIES: dict[str, str] = {
    "drop_missing_rows": "cleaning",
    "fill_missing": "cleaning",
    "drop_duplicates": "cleaning",
    "rename_column": "cleaning",
    "drop_columns": "cleaning",
    "convert_dtype": "cleaning",
    "drop_outlier_rows": "cleaning",
    "filter_rows": "cleaning",
    "text_transform": "cleaning",
    "numeric_transform": "transformation",
    "encode_categorical": "transformation",
    "bin_column": "transformation",
    "datetime_features": "feature",
    "arithmetic_feature": "feature",
    "duplicate_column": "feature",
    "sort_rows": "arrangement",
    "reorder_columns": "arrangement",
}


def action_category(action: str) -> str:
    """Operasyonun kod bölümü kategorisini döndürür."""
    return ACTION_CATEGORIES.get(str(action), "cleaning")


def filter_history(
    history: list[dict[str, Any]] | None, categories: list[str] | None
) -> list[dict[str, Any]]:
    """Operasyon geçmişini kategoriye göre filtreler (sırayı korur).

    Args:
        history: Operasyon sözlükleri.
        categories: İstenen kategoriler (None ise tümü).
    """
    if not history:
        return []
    if not categories:
        return list(history)
    allowed = set(categories)
    return [
        operation
        for operation in history
        if action_category(operation.get("action", "")) in allowed
    ]


_BOOL_MAP_LITERAL = (
    "{'true': True, '1': True, 'yes': True, 'evet': True, 'y': True, "
    "'false': False, '0': False, 'no': False, 'hayir': False, 'n': False}"
)


def _drop_missing_code(params: dict[str, Any]) -> str:
    """``drop_missing_rows`` operasyonunun kodunu üretir."""
    columns = params.get("columns") or []
    how = params.get("how", "any")
    if columns:
        args = f"subset={py_literal(list(columns))}, how={py_literal(how)}"
    else:
        args = f"how={py_literal(how)}" if how != "any" else ""
    return f"df = df.dropna({args}).reset_index(drop=True)"


def _fill_missing_code(params: dict[str, Any]) -> str:
    """``fill_missing`` operasyonunun kodunu üretir."""
    column = params["column"]
    method = params.get("method", "median")
    ref = col_ref(column)
    if method in ("mean", "median"):
        return f"{ref} = {ref}.fillna({ref}.{method}())"
    if method == "mode":
        return f"{ref} = {ref}.fillna({ref}.mode(dropna=True).iloc[0])"
    if method == "value":
        return f"{ref} = {ref}.fillna({py_literal(params.get('value'))})"
    if method in ("ffill", "bfill"):
        return f"{ref} = {ref}.{method}()"
    raise ValidationError(f"Kod üretilemeyen doldurma yöntemi: {method}")


def _drop_duplicates_code(params: dict[str, Any]) -> str:
    """``drop_duplicates`` operasyonunun kodunu üretir."""
    subset = params.get("subset") or []
    keep = params.get("keep", "first")
    args = []
    if subset:
        args.append(f"subset={py_literal(list(subset))}")
    if keep != "first":
        args.append(f"keep={py_literal(keep)}")
    return f"df = df.drop_duplicates({', '.join(args)}).reset_index(drop=True)"


def _convert_dtype_code(params: dict[str, Any]) -> str:
    """``convert_dtype`` operasyonunun kodunu üretir."""
    column = params["column"]
    target = params.get("target")
    ref = col_ref(column)
    if target == "integer":
        return f"{ref} = pd.to_numeric({ref}, errors='coerce').round().astype('Int64')"
    if target == "float":
        return f"{ref} = pd.to_numeric({ref}, errors='coerce').astype('Float64')"
    if target == "string":
        return f"{ref} = {ref}.astype('string')"
    if target == "category":
        return f"{ref} = {ref}.astype('category')"
    if target == "boolean":
        return (
            f"{ref} = ({ref}.astype(str).str.strip().str.lower()"
            f".map({_BOOL_MAP_LITERAL}).astype('boolean'))"
        )
    if target == "datetime":
        fmt = params.get("datetime_format")
        extra = f", format={py_literal(fmt)}" if fmt else ""
        return f"{ref} = pd.to_datetime({ref}, errors='coerce'{extra})"
    raise ValidationError(f"Kod üretilemeyen tip dönüşümü: {target}")


def _drop_outliers_code(params: dict[str, Any]) -> str:
    """``drop_outlier_rows`` operasyonunun kodunu üretir (IQR veya Z-Score)."""
    column = params["column"]
    factor = float(params.get("factor", 1.5))
    method = str(params.get("method", "iqr"))
    ref = col_ref(column)
    lit = col_literal(column)

    if method == "zscore":
        lines = [
            f"col_mean = df[{lit}].mean()",
            f"col_std = df[{lit}].std(ddof=0)",
            f"lower_bound = col_mean - {factor} * col_std",
            f"upper_bound = col_mean + {factor} * col_std",
        ]
    else:
        lines = [
            f"q1 = df[{lit}].quantile(0.25)",
            f"q3 = df[{lit}].quantile(0.75)",
            "iqr = q3 - q1",
            f"lower_bound = q1 - {factor} * iqr",
            f"upper_bound = q3 + {factor} * iqr",
        ]
    lines.append(
        f"df = df[~(({ref} < lower_bound) | ({ref} > upper_bound))]"
        ".reset_index(drop=True)"
    )
    return "\n".join(lines)


def _filter_rows_code(params: dict[str, Any]) -> str:
    """``filter_rows`` operasyonunun kodunu üretir."""
    code = filters_to_code(
        params.get("filters") or [], combine=str(params.get("combine", "and"))
    )
    if not code:
        raise ValidationError("Filtre tanımı boş olduğu için kod üretilemedi.")
    return code + "\ndf = df.reset_index(drop=True)"



# ---------------------------------------------------------------------------
# Dönüşüm operasyonları (transformations.py karşılıkları)
# ---------------------------------------------------------------------------


def _text_transform_code(params: dict[str, Any]) -> str:
    """``text_transform`` operasyonunun kodunu üretir."""
    column = params["column"]
    method = params.get("method")
    target = params.get("new_column") or column
    source = f"{col_ref(column)}.astype('string')"
    out = col_ref(target)

    simple = {
        "trim": ".str.strip()",
        "lower": ".str.lower()",
        "upper": ".str.upper()",
        "title": ".str.title()",
    }
    if method in simple:
        return f"{out} = {source}{simple[method]}"
    if method == "normalize_space":
        return (
            f"{out} = {source}.str.replace(r'\\s+', ' ', regex=True).str.strip()"
        )
    if method == "replace":
        find = py_literal(str(params.get("find", "")))
        replace_with = py_literal(str(params.get("replace_with") or ""))
        return f"{out} = {source}.str.replace({find}, {replace_with}, regex=False)"
    if method == "remove_chars":
        chars = py_literal(list(str(params.get("characters") or "")))
        return (
            f"{out} = {source}" + chr(10) +
            f"for char in {chars}:" + chr(10) +
            f"    {out} = {out}.str.replace(char, '', regex=False)"
        )
    raise ValidationError(f"Kod üretilemeyen metin dönüşümü: {method}")


def _numeric_transform_code(params: dict[str, Any]) -> str:
    """``numeric_transform`` operasyonunun kodunu üretir."""
    column = params["column"]
    method = params.get("method")
    target = params.get("new_column") or column
    ref = col_ref(column)
    out = col_ref(target)

    if method == "log":
        return f"{out} = np.log({ref}.where({ref} > 0))"
    if method == "log1p":
        return f"{out} = np.log1p({ref}.where({ref} > -1))"
    if method == "sqrt":
        return f"{out} = np.sqrt({ref}.where({ref} >= 0))"
    if method == "abs":
        return f"{out} = {ref}.abs()"
    if method == "round":
        return f"{out} = {ref}.round({int(params.get('decimals', 2))})"
    if method == "minmax":
        return (
            f"col_min = {ref}.min()" + chr(10) +
            f"col_max = {ref}.max()" + chr(10) +
            f"{out} = ({ref} - col_min) / (col_max - col_min)"
        )
    if method == "standard":
        return (
            f"col_mean = {ref}.mean()" + chr(10) +
            f"col_std = {ref}.std(ddof=0)" + chr(10) +
            f"{out} = ({ref} - col_mean) / col_std"
        )
    if method == "robust":
        return (
            f"col_median = {ref}.median()" + chr(10) +
            f"col_iqr = {ref}.quantile(0.75) - {ref}.quantile(0.25)" + chr(10) +
            f"{out} = ({ref} - col_median) / col_iqr"
        )
    raise ValidationError(f"Kod üretilemeyen sayısal dönüşüm: {method}")


def _encode_categorical_code(params: dict[str, Any]) -> str:
    """``encode_categorical`` operasyonunun kodunu üretir."""
    column = params["column"]
    method = params.get("method", "label")
    ref = col_ref(column)

    if method == "onehot":
        drop_first = bool(params.get("drop_first"))
        return (
            f"dummies = pd.get_dummies({ref}.astype('string'), "
            f"prefix={col_literal(column)}, drop_first={drop_first}).astype('int8')"
            + chr(10) +
            f"df = pd.concat([df.drop(columns=[{col_literal(column)}]), dummies], axis=1)"
        )

    target = params.get("new_column") or column
    out = col_ref(target)
    if method == "label":
        return (
            f"{out} = ({ref}.astype('string').astype('category').cat.codes"
            f".where({ref}.notna()).astype('Int64'))"
        )
    if method == "ordinal":
        categories = [str(value) for value in (params.get("categories") or [])]
        mapping = {value: index for index, value in enumerate(categories)}
        return f"{out} = {ref}.astype('string').map({py_literal(mapping)}).astype('Int64')"
    raise ValidationError(f"Kod üretilemeyen encoding yöntemi: {method}")


_MONTH_NAMES_LITERAL = (
    "['Ocak', 'Şubat', 'Mart', 'Nisan', 'Mayıs', 'Haziran', "
    "'Temmuz', 'Ağustos', 'Eylül', 'Ekim', 'Kasım', 'Aralık']"
)
_WEEKDAY_NAMES_LITERAL = (
    "['Pazartesi', 'Salı', 'Çarşamba', 'Perşembe', 'Cuma', 'Cumartesi', 'Pazar']"
)


def _datetime_features_code(params: dict[str, Any]) -> str:
    """``datetime_features`` operasyonunun kodunu üretir."""
    column = params["column"]
    features = list(params.get("features") or [])
    if not features:
        raise ValidationError("Tarih özelliği seçilmediği için kod üretilemedi.")
    prefix = str(params.get("prefix") or column)

    lines = [f"dt_series = pd.to_datetime({col_ref(column)}, errors='coerce')"]
    templates = {
        "year": "dt_series.dt.year.astype('Int64')",
        "quarter": "dt_series.dt.quarter.astype('Int64')",
        "month": "dt_series.dt.month.astype('Int64')",
        "day": "dt_series.dt.day.astype('Int64')",
        "weekday": "dt_series.dt.dayofweek.astype('Int64')",
        "hour": "dt_series.dt.hour.astype('Int64')",
        "is_weekend": "(dt_series.dt.dayofweek >= 5).astype('Int64')",
        "week": "dt_series.dt.isocalendar().week.astype('Int64')",
        "dayofyear": "dt_series.dt.dayofyear.astype('Int64')",
    }
    for feature in features:
        name = col_ref(f"{prefix}_{feature}")
        if feature == "month_name":
            lines.append(f"month_names = {_MONTH_NAMES_LITERAL}")
            lines.append(
                f"{name} = dt_series.dt.month.map("
                "lambda m: month_names[int(m) - 1] if pd.notna(m) else pd.NA"
                ").astype('string')"
            )
        elif feature == "weekday_name":
            lines.append(f"weekday_names = {_WEEKDAY_NAMES_LITERAL}")
            lines.append(
                f"{name} = dt_series.dt.dayofweek.map("
                "lambda d: weekday_names[int(d)] if pd.notna(d) else pd.NA"
                ").astype('string')"
            )
        else:
            lines.append(f"{name} = {templates[feature]}")
    return chr(10).join(lines)


def _arithmetic_feature_code(params: dict[str, Any]) -> str:
    """``arithmetic_feature`` operasyonunun kodunu üretir."""
    left = col_ref(params["left"])
    right = col_ref(params["right"])
    out = col_ref(params["new_column"])
    operation = params.get("operation")
    symbols = {"add": "+", "subtract": "-", "multiply": "*"}
    if operation in symbols:
        return f"{out} = {left} {symbols[operation]} {right}"
    if operation == "divide":
        return f"{out} = {left} / {right}.replace(0, np.nan)"
    if operation == "ratio":
        return f"{out} = {left} / ({left} + {right}).replace(0, np.nan)"
    raise ValidationError(f"Kod üretilemeyen aritmetik işlem: {operation}")


def _bin_column_code(params: dict[str, Any]) -> str:
    """``bin_column`` operasyonunun kodunu üretir."""
    column = params["column"]
    method = params.get("method", "equal_width")
    target = params.get("new_column") or f"{column}_bin"
    ref = col_ref(column)
    out = col_ref(target)
    labels = params.get("labels")
    labels_arg = f", labels={py_literal([str(label) for label in labels])}" if labels else ""

    if method == "custom":
        edges = sorted(float(edge) for edge in (params.get("edges") or []))
        return (
            f"{out} = pd.cut({ref}, bins={py_literal(edges)}{labels_arg}, "
            "include_lowest=True).astype('string')"
        )
    bins = int(params.get("bins", 4))
    if method == "quantile":
        return (
            f"{out} = pd.qcut({ref}, q={bins}{labels_arg}, "
            "duplicates='drop').astype('string')"
        )
    return f"{out} = pd.cut({ref}, bins={bins}{labels_arg}).astype('string')"


def _sort_rows_code(params: dict[str, Any]) -> str:
    """``sort_rows`` operasyonunun kodunu üretir."""
    columns = [str(column) for column in (params.get("columns") or [])]
    ascending = params.get("ascending", True)
    if isinstance(ascending, bool):
        order = py_literal(bool(ascending))
    else:
        order = py_literal([bool(value) for value in ascending])
    return (
        f"df = df.sort_values(by={py_literal(columns)}, ascending={order})"
        ".reset_index(drop=True)"
    )

#: Operasyon adından kod üreticisine eşleme.
_CODE_BUILDERS: dict[str, Any] = {
    "drop_missing_rows": _drop_missing_code,
    "fill_missing": _fill_missing_code,
    "drop_duplicates": _drop_duplicates_code,
    "rename_column": lambda p: (
        f"df = df.rename(columns={{{col_literal(p['column'])}: "
        f"{col_literal(p['new_name'])}}})"
    ),
    "drop_columns": lambda p: f"df = df.drop(columns={py_literal(list(p['columns']))})",
    "convert_dtype": _convert_dtype_code,
    "drop_outlier_rows": _drop_outliers_code,
    "filter_rows": _filter_rows_code,
    "text_transform": _text_transform_code,
    "numeric_transform": _numeric_transform_code,
    "encode_categorical": _encode_categorical_code,
    "datetime_features": _datetime_features_code,
    "arithmetic_feature": _arithmetic_feature_code,
    "bin_column": _bin_column_code,
    "sort_rows": _sort_rows_code,
    "duplicate_column": lambda p: (
        f"{col_ref(p.get('new_column') or str(p['column']) + '_kopya')} = "
        f"{col_ref(p['column'])}"
    ),
    "reorder_columns": lambda p: f"df = df[{py_literal([str(c) for c in p['columns']])}]",
}


def build_operation_code(action: str, params: dict[str, Any]) -> str:
    """Bir operasyonun Python kod karşılığını üretir.

    Args:
        action: Operasyon adı (``fill_missing`` gibi).
        params: Operasyon parametreleri.

    Returns:
        Bir veya birden fazla satırdan oluşan Python kodu.

    Raises:
        ValidationError: Operasyon için kod üreticisi tanımlı değilse.
    """
    builder = _CODE_BUILDERS.get(action)
    if builder is None:
        raise ValidationError(f"'{action}' operasyonu için kod üreticisi tanımlı değil.")
    try:
        return builder(params)
    except KeyError as exc:
        raise ValidationError(
            f"'{action}' operasyonu için eksik parametre: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Tam script üretimi
# ---------------------------------------------------------------------------


def generate_imports(include_charts: bool = True) -> str:
    """Script başındaki import bloğunu üretir."""
    lines = ["import pandas as pd", "import numpy as np"]
    if include_charts:
        lines.append("import plotly.express as px")
    return "\n".join(lines)


def generate_loading_section(meta: dict[str, Any]) -> str:
    """Dataset yükleme bölümünü üretir."""
    return loading_code(meta)


def generate_cleaning_section(
    history: list[dict[str, Any]], with_comments: bool = True
) -> str:
    """Operation history'yi sırayla koda çevirir.

    Args:
        history: ``action``, ``params``, ``description``, ``python_code``
            alanlarını içeren operasyon sözlükleri.
        with_comments: Her operasyonun açıklamasını yorum satırı olarak ekler.

    Returns:
        Veri temizleme kod bloğu (işlem yoksa boş metin).
    """
    blocks: list[str] = []
    for index, op in enumerate(history, start=1):
        code = op.get("python_code") or build_operation_code(
            op.get("action", ""), op.get("params", {}) or {}
        )
        if with_comments and op.get("description"):
            blocks.append(f"# {index}. {op['description']}\n{code}")
        else:
            blocks.append(code)
    return "\n\n".join(blocks)


def generate_charts_section(
    charts: list[dict[str, Any]], with_comments: bool = True
) -> str:
    """Oluşturulan grafiklerin Plotly kodlarını üretir."""
    blocks: list[str] = []
    for index, chart in enumerate(charts, start=1):
        code = chart.get("python_code")
        if not code:
            continue
        if with_comments:
            title = chart.get("description") or chart.get("title") or chart.get("type", "")
            blocks.append(f"# Grafik {index}: {title}\n{code}")
        else:
            blocks.append(code)
    return "\n\n".join(blocks)


def generate_notes_section(insights: list[dict[str, Any]]) -> str:
    """Rule-based insight sonuçlarını yorum satırı olarak üretir."""
    if not insights:
        return ""
    lines = ["# Otomatik veri kalitesi notları (kural tabanlı):"]
    for item in insights:
        severity = str(item.get("severity", "info")).upper()
        column = item.get("column")
        prefix = f"[{severity}]" + (f" {column}:" if column else "")
        lines.append(f"# {prefix} {item.get('title', '')} - {item.get('message', '')}")
    return "\n".join(lines)


def generate_sectioned_cleaning(
    history: list[dict[str, Any]], with_comments: bool = True
) -> str:
    """Operasyonları kronolojik sırada, bölüm başlıklarıyla koda çevirir.

    Sıra **değiştirilmez** (replay doğruluğu için); kategori değiştiğinde
    yorum satırı olarak yeni bölüm başlığı eklenir.
    """
    if not history:
        return ""
    blocks: list[str] = []
    current_category: str | None = None
    for index, operation in enumerate(history, start=1):
        category = action_category(operation.get("action", ""))
        if category != current_category:
            current_category = category
            blocks.append(f"# --- {CODE_SECTIONS.get(category, category)} ---")
        code = operation.get("python_code") or build_operation_code(
            operation.get("action", ""), operation.get("params", {}) or {}
        )
        if with_comments and operation.get("description"):
            blocks.append(f"# {index}. {operation['description']}\n{code}")
        else:
            blocks.append(code)
    return "\n\n".join(blocks)


def generate_full_script(
    meta: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
    charts: list[dict[str, Any]] | None = None,
    include_loading: bool = True,
    include_cleaning: bool = True,
    include_charts: bool = True,
    include_header: bool = True,
    analysis_code: str | None = None,
    ml_code: str | None = None,
) -> str:
    """Oturumun tamamını çalıştırılabilir tek bir Python script'ine çevirir.

    Args:
        meta: Dataset bilgisi (``filename``, ``file_type``, ``separator``,
            ``encoding``, ``sheet_name``).
        history: Operasyon geçmişi.
        charts: Oluşturulan grafik tanımları.
        include_loading: Dataset yükleme bölümünü ekle.
        include_cleaning: Veri temizleme bölümünü ekle.
        include_charts: Grafik bölümünü ekle.
        include_header: Üstteki açıklama bloğunu ekle.
        analysis_code: İstatistik/korelasyon gibi analiz kodu (opsiyonel).
        ml_code: ML hazırlık kodu (opsiyonel).

    Returns:
        Tam Python script metni.
    """
    history = history or []
    charts = charts or []
    sections: list[str] = []

    if include_header:
        # Başlık yorum satırı olarak üretilir: dosya adında ters bölü veya
        # tırnak olsa bile docstring kaçış sorunu oluşmaz.
        name = str(meta.get("filename") or "veri seti").replace("\n", " ")
        sections.append(
            "# ==========================================================\n"
            "# Bu script No-Code EDA Studio tarafından otomatik üretildi.\n"
            f"# Kaynak dosya      : {name}\n"
            f"# Veri işleme adımı : {len(history)}\n"
            f"# Grafik sayısı     : {len(charts)}\n"
            "# =========================================================="
        )

    sections.append(generate_imports(include_charts=bool(include_charts and charts)))

    if include_loading:
        sections.append("# --- Dataset Yükleme ---\n" + generate_loading_section(meta))

    if include_cleaning and history:
        sections.append(generate_sectioned_cleaning(history))

    if analysis_code:
        sections.append("# --- Analiz ---\n" + analysis_code.strip())

    if include_charts and charts:
        sections.append("# --- Görselleştirmeler ---\n" + generate_charts_section(charts))

    if ml_code:
        sections.append("# --- ML Hazırlığı ---\n" + ml_code.strip())

    if not history and not charts and not analysis_code and not ml_code:
        sections.append(
            "# Henüz veri dönüşümü veya grafik oluşturulmadı.\n"
            "print(df.head())\nprint(df.describe(include='all'))"
        )

    return "\n\n".join(section for section in sections if section.strip()) + "\n"
