"""Kural tabanlı (rule-based) Smart Insights motoru.

API veya LLM kullanılmaz. Tüm sonuçlar **deterministik** ve açıklanabilirdir:
her insight hangi kuralın hangi metrikle tetiklendiğini söyler.

Insight sözlüğü:
    ``{"severity", "category", "column", "title", "message", "metric", "rule"}``
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core import outliers as outliers_mod
from utils import dataframe_utils as dfu
from utils.helpers import format_number, format_percent

__all__ = [
    "SEVERITIES",
    "CATEGORIES",
    "INSIGHT_RULES",
    "generate_insights",
    "filter_insights",
    "insight_counts",
    "insights_to_dataframe",
]

#: Önem dereceleri ve etiketleri (sıralama bu sözlüğün sırasına göre yapılır).
SEVERITIES: dict[str, str] = {
    "critical": "Kritik",
    "warning": "Uyarı",
    "info": "Bilgi",
}

#: Insight kategorileri.
CATEGORIES: dict[str, str] = {
    "missing": "Boş Değerler",
    "duplicates": "Tekrarlı Satırlar",
    "constant": "Sabit Sütunlar",
    "identifier": "Olası ID Sütunları",
    "imbalance": "Kategori Dengesizliği",
    "cardinality": "Yüksek Kardinalite",
    "distribution": "Dağılım / Çarpıklık",
    "correlation": "Korelasyon",
    "outliers": "Aykırı Değerler",
    "structure": "Veri Yapısı",
}

#: Kuralların eşik değerleri (UI'da gösterilebilir / testlerde kullanılır).
INSIGHT_RULES: dict[str, float] = {
    "missing_low": 5.0,
    "missing_medium": 20.0,
    "id_ratio": 0.95,
    "imbalance_ratio": 0.80,
    "skewness": 1.0,
    "correlation": 0.80,
    "outlier_ratio": 5.0,
    "cardinality_ratio": 0.50,
    "cardinality_min_unique": 20,
}

_SEVERITY_ORDER = {key: index for index, key in enumerate(SEVERITIES)}


def _insight(
    severity: str,
    category: str,
    title: str,
    message: str,
    column: str | None = None,
    metric: Any = None,
    rule: str = "",
) -> dict[str, Any]:
    """Insight sözlüğü üretir."""
    return {
        "severity": severity,
        "category": category,
        "column": column,
        "title": title,
        "message": message,
        "metric": metric,
        "rule": rule,
    }


def _missing_insights(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Boş değer kurallarını uygular."""
    rows = int(len(df))
    found: list[dict[str, Any]] = []
    for col in df.columns:
        missing = int(dfu.get_series(df, col).isna().sum())
        if missing == 0:
            continue
        ratio = missing / rows * 100 if rows else 0.0
        if missing == rows:
            found.append(
                _insight(
                    "critical",
                    "missing",
                    "Sütun tamamen boş",
                    f"'{col}' sütununun tüm değerleri boş. Analize katkısı yok; "
                    "silmeyi değerlendirin.",
                    column=str(col),
                    metric=round(ratio, 2),
                    rule="missing == 100%",
                )
            )
        elif ratio > INSIGHT_RULES["missing_medium"]:
            found.append(
                _insight(
                    "critical",
                    "missing",
                    "Yüksek oranda boş değer",
                    f"'{col}' sütununun %{ratio:.1f}'i boş ({format_number(missing)} hücre). "
                    "Doldurma yöntemi sonuçları ciddi şekilde etkileyebilir.",
                    column=str(col),
                    metric=round(ratio, 2),
                    rule="missing > 20%",
                )
            )
        elif ratio >= INSIGHT_RULES["missing_low"]:
            found.append(
                _insight(
                    "warning",
                    "missing",
                    "Orta düzeyde boş değer",
                    f"'{col}' sütununun %{ratio:.1f}'i boş. Doldurma veya silme "
                    "kararını analiz hedefinize göre verin.",
                    column=str(col),
                    metric=round(ratio, 2),
                    rule="5% <= missing <= 20%",
                )
            )
        else:
            found.append(
                _insight(
                    "info",
                    "missing",
                    "Düşük oranda boş değer",
                    f"'{col}' sütununda %{ratio:.1f} boş değer var "
                    f"({format_number(missing)} hücre).",
                    column=str(col),
                    metric=round(ratio, 2),
                    rule="missing < 5%",
                )
            )
    return found


def _duplicate_insights(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Tekrarlı satır kuralını uygular."""
    rows = int(len(df))
    dups = dfu.duplicate_count(df)
    if dups == 0 or rows == 0:
        return []
    ratio = dups / rows * 100
    severity = "critical" if ratio >= 10 else "warning"
    return [
        _insight(
            severity,
            "duplicates",
            "Tekrarlı satırlar var",
            f"Veri setinde {format_number(dups)} tekrarlı satır bulundu "
            f"(%{ratio:.1f}). Veri Temizleme sayfasından silebilirsiniz.",
            metric=round(ratio, 2),
            rule="duplicate > 0",
        )
    ]


def _constant_insights(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Sabit sütun kuralını uygular."""
    found: list[dict[str, Any]] = []
    for col in dfu.constant_columns(df):
        if bool(dfu.get_series(df, col).isna().all()):
            continue  # boş sütun zaten missing kuralında bildiriliyor
        found.append(
            _insight(
                "warning",
                "constant",
                "Sabit (tek değerli) sütun",
                f"'{col}' sütunu tek bir değer taşıyor; modelleme ve analizde "
                "bilgi taşımaz.",
                column=str(col),
                metric=1,
                rule="unique == 1",
            )
        )
    return found


def _id_insights(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Olası ID sütunu kuralını uygular."""
    found: list[dict[str, Any]] = []
    for col, ratio in dfu.potential_id_columns(df, INSIGHT_RULES["id_ratio"]):
        found.append(
            _insight(
                "info",
                "identifier",
                "Olası ID sütunu",
                f"'{col}' sütununun değerlerinin %{ratio * 100:.1f}'i tekil. "
                "Bu bir kimlik sütunu olabilir; istatistiksel analizlerde "
                "kullanmak genelde anlamsızdır. (Bu yalnızca bir tahmindir.)",
                column=str(col),
                metric=round(ratio * 100, 2),
                rule="unique_ratio >= 0.95",
            )
        )
    return found


def _imbalance_insights(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Kategori dengesizliği kuralını uygular."""
    found: list[dict[str, Any]] = []
    for col in dfu.categorical_columns(df):
        series = dfu.get_series(df, col).dropna()
        if series.empty or dfu.safe_nunique(series) < 2:
            continue
        try:
            counts = series.value_counts()
        except TypeError:
            counts = series.astype(str).value_counts()
        ratio = float(counts.iloc[0]) / float(counts.sum())
        if ratio >= INSIGHT_RULES["imbalance_ratio"]:
            found.append(
                _insight(
                    "warning",
                    "imbalance",
                    "Kategori dengesizliği",
                    f"'{col}' sütununda '{counts.index[0]}' değeri tek başına "
                    f"%{ratio * 100:.1f} pay alıyor. Karşılaştırmalı analizler "
                    "yanıltıcı olabilir.",
                    column=str(col),
                    metric=round(ratio * 100, 2),
                    rule="max_category_ratio >= 0.80",
                )
            )
    return found


def _cardinality_insights(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Yüksek kardinalite kuralını uygular."""
    found: list[dict[str, Any]] = []
    id_columns = {col for col, _ in dfu.potential_id_columns(df, INSIGHT_RULES["id_ratio"])}
    for col, nunique, ratio in dfu.high_cardinality_columns(
        df,
        INSIGHT_RULES["cardinality_ratio"],
        int(INSIGHT_RULES["cardinality_min_unique"]),
    ):
        if col in id_columns:
            continue  # ID olarak zaten bildirildi
        found.append(
            _insight(
                "warning",
                "cardinality",
                "Yüksek kardinaliteli kategorik sütun",
                f"'{col}' sütununda {format_number(nunique)} farklı değer var "
                f"(%{ratio * 100:.1f} tekillik). Grafik ve gruplamalarda "
                "okunabilirlik düşer; gruplama veya sadeleştirme düşünün.",
                column=str(col),
                metric=round(ratio * 100, 2),
                rule="unique_ratio >= 0.50 ve unique >= 20",
            )
        )
    return found


def _distribution_insights(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Çarpıklık kuralını uygular."""
    found: list[dict[str, Any]] = []
    for col in dfu.numeric_columns(df):
        series = pd.to_numeric(dfu.get_series(df, col), errors="coerce").replace([np.inf, -np.inf], np.nan)
        clean = series.dropna()
        if len(clean) < 3 or clean.nunique() < 3:
            continue
        skewness = float(clean.skew())
        if pd.isna(skewness) or abs(skewness) <= INSIGHT_RULES["skewness"]:
            continue
        direction = "sağa" if skewness > 0 else "sola"
        found.append(
            _insight(
                "info",
                "distribution",
                "Çarpık dağılım",
                f"'{col}' sütununun çarpıklığı {skewness:.2f} ({direction} çarpık). "
                "Ortalama yerine medyan daha temsili olabilir.",
                column=str(col),
                metric=round(skewness, 4),
                rule="abs(skewness) > 1",
            )
        )
    return found


def _correlation_insights(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Yüksek korelasyon kuralını uygular."""
    numeric = dfu.numeric_columns(df)
    if len(numeric) < 2:
        return []
    subset = df[numeric].apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    try:
        matrix = subset.corr(method="pearson", min_periods=2)
    except (ValueError, TypeError):  # pragma: no cover
        return []

    found: list[dict[str, Any]] = []
    for i, first in enumerate(matrix.columns):
        for second in matrix.columns[i + 1 :]:
            value = matrix.loc[first, second]
            if pd.isna(value):
                continue
            r = float(value)
            if abs(r) < INSIGHT_RULES["correlation"]:
                continue
            found.append(
                _insight(
                    "warning",
                    "correlation",
                    "Çok yüksek korelasyon",
                    f"'{first}' ve '{second}' sütunları arasında r = {r:.2f} "
                    "korelasyon var. Sütunlar benzer bilgi taşıyor olabilir. "
                    "Korelasyon nedensellik göstermez.",
                    column=f"{first} ~ {second}",
                    metric=round(r, 4),
                    rule="abs(pearson r) >= 0.80",
                )
            )
    return found


def _outlier_insights(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Aykırı değer kuralını uygular."""
    found: list[dict[str, Any]] = []
    for col in dfu.numeric_columns(df):
        try:
            report = outliers_mod.detect_outliers(df, col)
        except Exception:  # sabit/boş sütunlar
            continue
        if report.count == 0 or report.ratio < INSIGHT_RULES["outlier_ratio"]:
            continue
        severity = "warning" if report.ratio >= 10 else "info"
        found.append(
            _insight(
                severity,
                "outliers",
                "Aykırı değer yoğunluğu",
                f"'{col}' sütununda IQR yöntemine göre {format_number(report.count)} "
                f"aykırı değer var ({format_percent(report.ratio)}). Bir değerin "
                "istatistiksel olarak aykırı olması hatalı olduğu anlamına gelmez.",
                column=str(col),
                metric=round(report.ratio, 2),
                rule="IQR outlier oranı >= 5%",
            )
        )
    return found


def _structure_insights(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Veri yapısıyla ilgili genel kuralları uygular."""
    found: list[dict[str, Any]] = []
    rows, cols = df.shape
    if rows <= 1:
        found.append(
            _insight(
                "critical",
                "structure",
                "Veri seti çok küçük",
                f"Veri setinde yalnızca {rows} satır var. İstatistik, korelasyon ve "
                "aykırı değer analizleri anlamlı sonuç vermez.",
                metric=rows,
                rule="rows <= 1",
            )
        )
    elif rows < 30:
        found.append(
            _insight(
                "info",
                "structure",
                "Küçük veri seti",
                f"{rows} satır, istatistiksel çıkarım için düşük bir sayı olabilir.",
                metric=rows,
                rule="rows < 30",
            )
        )
    if cols == 1:
        found.append(
            _insight(
                "info",
                "structure",
                "Tek sütunlu veri seti",
                "Veri setinde tek sütun var; ilişki ve korelasyon analizleri "
                "yapılamaz. Dosya ayracı yanlış algılanmış olabilir.",
                metric=cols,
                rule="columns == 1",
            )
        )
    for col in dfu.numeric_columns(df):
        series = dfu.get_series(df, col)
        if np.isinf(pd.to_numeric(series, errors="coerce")).any():
            found.append(
                _insight(
                    "warning",
                    "structure",
                    "Sonsuz (inf) değer",
                    f"'{col}' sütununda sonsuz değer var. İstatistikler yanıltıcı "
                    "olabilir; bu satırları temizlemeyi değerlendirin.",
                    column=str(col),
                    rule="inf değer mevcut",
                )
            )
    return found


def generate_insights(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Tüm kuralları uygulayıp insight listesi döndürür.

    Args:
        df: İncelenecek (genellikle ``working_df``) DataFrame.

    Returns:
        Önem derecesine (kritik → bilgi) ve mutlak metriğe göre sıralanmış
        insight sözlükleri listesi.
    """
    if df is None or df.empty or df.shape[1] == 0:
        return [
            _insight(
                "critical",
                "structure",
                "Veri yok",
                "Analiz edilecek veri bulunamadı. Lütfen bir dosya yükleyin.",
                rule="df boş",
            )
        ]

    insights: list[dict[str, Any]] = []
    for rule_fn in (
        _structure_insights,
        _missing_insights,
        _duplicate_insights,
        _constant_insights,
        _id_insights,
        _imbalance_insights,
        _cardinality_insights,
        _distribution_insights,
        _correlation_insights,
        _outlier_insights,
    ):
        try:
            insights.extend(rule_fn(df))
        except Exception:  # tek bir kural hatası tüm motoru durdurmasın
            continue

    def sort_key(item: dict[str, Any]) -> tuple[int, float]:
        metric = item.get("metric")
        try:
            magnitude = -abs(float(metric))
        except (TypeError, ValueError):
            magnitude = 0.0
        return _SEVERITY_ORDER.get(item.get("severity", "info"), 99), magnitude

    return sorted(insights, key=sort_key)


def filter_insights(
    insights: list[dict[str, Any]],
    severities: list[str] | None = None,
    categories: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Insight listesini önem derecesi ve kategoriye göre filtreler."""
    result = list(insights or [])
    if severities:
        allowed = set(severities)
        result = [i for i in result if i.get("severity") in allowed]
    if categories:
        allowed_cat = set(categories)
        result = [i for i in result if i.get("category") in allowed_cat]
    return result


def insight_counts(insights: list[dict[str, Any]]) -> dict[str, int]:
    """Önem derecesine göre insight sayılarını döndürür."""
    counts = {key: 0 for key in SEVERITIES}
    for item in insights or []:
        severity = item.get("severity", "info")
        counts[severity] = counts.get(severity, 0) + 1
    return counts


def insights_to_dataframe(insights: list[dict[str, Any]]) -> pd.DataFrame:
    """Insight listesini tablo haline getirir (export ve gösterim için)."""
    records = [
        {
            "Önem": SEVERITIES.get(item.get("severity", "info"), item.get("severity")),
            "Kategori": CATEGORIES.get(item.get("category", ""), item.get("category")),
            "Sütun": item.get("column") or "-",
            "Başlık": item.get("title"),
            "Mesaj": item.get("message"),
            "Metrik": item.get("metric"),
            "Kural": item.get("rule"),
        }
        for item in insights or []
    ]
    return pd.DataFrame.from_records(
        records,
        columns=["Önem", "Kategori", "Sütun", "Başlık", "Mesaj", "Metrik", "Kural"],
    )
