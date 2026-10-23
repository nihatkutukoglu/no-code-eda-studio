"""Dataset Health / Data Quality Score.

ÖNEMLİ: Bu skor **bilimsel veya evrensel bir standart değildir**. Amacı
gamification değil; kullanıcının veri kalitesi sorunlarını hızlıca görmesidir.
Formül bilinçli olarak basit ve açıklanabilir tutulmuştur.

Skor = ağırlıklı alt bileşen ortalaması (0-100):

| Bileşen                  | Ağırlık | Nasıl hesaplanır                                  |
|--------------------------|---------|---------------------------------------------------|
| Completeness             | 35      | 100 - boş hücre yüzdesi                           |
| Duplicate Quality        | 20      | 100 - tekrarlı satır yüzdesi                      |
| Type Consistency         | 15      | Tip tutarsızlığı bulunan sütun oranına göre ceza  |
| Low Information Columns  | 15      | Sabit / tamamen boş sütun oranına göre ceza       |
| Outlier Indicator        | 15      | Sayısal sütunların ortalama IQR aykırı oranı      |
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from core import outliers as outliers_mod
from core.profiler import dtype_check_table
from utils import dataframe_utils as dfu

__all__ = [
    "COMPONENT_WEIGHTS",
    "SCORE_DISCLAIMER",
    "ScoreComponent",
    "QualityScore",
    "compute_quality_score",
    "score_label",
]

#: Alt bileşen ağırlıkları (toplam 100).
COMPONENT_WEIGHTS: dict[str, int] = {
    "completeness": 35,
    "duplicates": 20,
    "type_consistency": 15,
    "low_information": 15,
    "outliers": 15,
}

#: UI'da her zaman gösterilmesi gereken uyarı metni.
SCORE_DISCLAIMER = (
    "Bu skor bilimsel veya evrensel bir standart değildir. Yalnızca veri "
    "kalitesi sorunlarını hızlı görmeniz için tanımlanmış basit bir göstergedir."
)


@dataclass
class ScoreComponent:
    """Tek bir alt skor bileşeni."""

    key: str
    label: str
    score: float
    weight: int
    detail: str

    @property
    def weighted_points(self) -> float:
        """Bileşenin toplam skora katkısı (puan)."""
        return round(self.score * self.weight / 100, 1)

    @property
    def lost_points(self) -> float:
        """Bu bileşen yüzünden kaybedilen puan."""
        return round(self.weight - self.weighted_points, 1)


@dataclass
class QualityScore:
    """Genel skor ve alt bileşenleri."""

    overall: int
    components: list[ScoreComponent]

    @property
    def label(self) -> str:
        """Skorun sözel karşılığı."""
        return score_label(self.overall)

    def to_dataframe(self) -> pd.DataFrame:
        """Alt skorları tablo olarak döndürür."""
        return pd.DataFrame(
            [
                {
                    "Bileşen": component.label,
                    "Alt Skor": round(component.score, 1),
                    "Ağırlık": component.weight,
                    "Katkı (puan)": component.weighted_points,
                    "Kayıp (puan)": component.lost_points,
                    "Açıklama": component.detail,
                }
                for component in self.components
            ],
            columns=[
                "Bileşen",
                "Alt Skor",
                "Ağırlık",
                "Katkı (puan)",
                "Kayıp (puan)",
                "Açıklama",
            ],
        )


def score_label(score: float) -> str:
    """Skoru sözel etikete çevirir."""
    if score >= 90:
        return "Çok iyi"
    if score >= 75:
        return "İyi"
    if score >= 60:
        return "Orta"
    if score >= 40:
        return "Zayıf"
    return "Sorunlu"


def _clamp(value: float) -> float:
    """Skoru 0-100 aralığına sıkıştırır."""
    return float(max(0.0, min(100.0, value)))


def _completeness(df: pd.DataFrame) -> ScoreComponent:
    """Boş hücre oranına dayalı bileşen."""
    cells = int(df.shape[0]) * int(df.shape[1])
    missing = dfu.total_missing(df)
    ratio = (missing / cells * 100) if cells else 0.0
    return ScoreComponent(
        key="completeness",
        label="Completeness (Doluluk)",
        score=_clamp(100 - ratio),
        weight=COMPONENT_WEIGHTS["completeness"],
        detail=f"{missing:,} / {cells:,} hücre boş (%{ratio:.1f}).",
    )


def _duplicates(df: pd.DataFrame) -> ScoreComponent:
    """Tekrarlı satır oranına dayalı bileşen."""
    rows = int(len(df))
    dups = dfu.duplicate_count(df)
    ratio = (dups / rows * 100) if rows else 0.0
    return ScoreComponent(
        key="duplicates",
        label="Duplicate Quality (Tekrarsızlık)",
        score=_clamp(100 - ratio),
        weight=COMPONENT_WEIGHTS["duplicates"],
        detail=f"{dups:,} tekrarlı satır (%{ratio:.1f}).",
    )


def _type_consistency(df: pd.DataFrame) -> ScoreComponent:
    """Tip tutarsızlığı bulunan sütun oranına dayalı bileşen."""
    columns = int(df.shape[1]) or 1
    try:
        findings = dtype_check_table(df)
    except Exception:  # pragma: no cover
        findings = pd.DataFrame(columns=["Sütun"])
    # Yalnızca gerçek tip sorunlarını say (öneri niteliğindeki category ipucunu değil).
    if not findings.empty:
        serious = findings[~findings["Tespit"].str.contains("Az sayıda tekil", na=False)]
    else:
        serious = findings
    problem_columns = int(serious["Sütun"].nunique()) if not serious.empty else 0
    ratio = problem_columns / columns * 100
    return ScoreComponent(
        key="type_consistency",
        label="Type Consistency (Tip Tutarlılığı)",
        score=_clamp(100 - ratio * 1.5),
        weight=COMPONENT_WEIGHTS["type_consistency"],
        detail=f"{problem_columns} / {columns} sütunda tip tutarsızlığı sinyali var.",
    )


def _low_information(df: pd.DataFrame) -> ScoreComponent:
    """Sabit ve tamamen boş sütunlara dayalı bileşen."""
    columns = int(df.shape[1]) or 1
    constants = set(dfu.constant_columns(df))
    empties = set(dfu.empty_columns(df))
    low_info = constants | empties
    ratio = len(low_info) / columns * 100
    detail = f"{len(low_info)} / {columns} sütun bilgi taşımıyor (sabit veya boş)."
    if low_info:
        preview = ", ".join(sorted(low_info)[:4])
        detail += f" Örnek: {preview}."
    return ScoreComponent(
        key="low_information",
        label="Low Information Columns (Bilgi Taşımayan Sütunlar)",
        score=_clamp(100 - ratio * 2),
        weight=COMPONENT_WEIGHTS["low_information"],
        detail=detail,
    )


def _outliers(df: pd.DataFrame) -> ScoreComponent:
    """Sayısal sütunların ortalama aykırı değer oranına dayalı bileşen."""
    numeric = dfu.numeric_columns(df)
    if not numeric:
        return ScoreComponent(
            key="outliers",
            label="Outlier Indicator (Aykırı Değer)",
            score=100.0,
            weight=COMPONENT_WEIGHTS["outliers"],
            detail="Sayısal sütun yok; bu bileşen ceza uygulamadı.",
        )
    ratios: list[float] = []
    for col in numeric:
        try:
            ratios.append(outliers_mod.detect_outliers(df, col).ratio)
        except Exception:
            continue
    average = sum(ratios) / len(ratios) if ratios else 0.0
    return ScoreComponent(
        key="outliers",
        label="Outlier Indicator (Aykırı Değer)",
        # Ortalama %10 aykırı oranı skoru sıfırlar (10 * 10 = 100 ceza).
        score=_clamp(100 - average * 10),
        weight=COMPONENT_WEIGHTS["outliers"],
        detail=(
            f"{len(ratios)} sayısal sütunda ortalama IQR aykırı oranı %{average:.1f}. "
            "Aykırı değer her zaman hata değildir."
        ),
    )


def compute_quality_score(df: pd.DataFrame) -> QualityScore:
    """Veri kalite skorunu hesaplar.

    Args:
        df: Skorun hesaplanacağı DataFrame (yalnızca ``working_df``).

    Returns:
        Genel skor (0-100) ve alt bileşenleri içeren QualityScore.
    """
    if df is None or df.empty or df.shape[1] == 0:
        empty_components = [
            ScoreComponent(key, key.replace("_", " ").title(), 0.0, weight, "Veri yok.")
            for key, weight in COMPONENT_WEIGHTS.items()
        ]
        return QualityScore(overall=0, components=empty_components)

    components = [
        _completeness(df),
        _duplicates(df),
        _type_consistency(df),
        _low_information(df),
        _outliers(df),
    ]
    total_weight = sum(component.weight for component in components) or 1
    overall = sum(component.score * component.weight for component in components) / total_weight
    return QualityScore(overall=int(round(overall)), components=components)


def score_breakdown_dict(score: QualityScore) -> dict[str, Any]:
    """Skoru sözlük olarak döndürür (export / test için)."""
    return {
        "overall": score.overall,
        "label": score.label,
        "components": [
            {
                "key": component.key,
                "label": component.label,
                "score": round(component.score, 2),
                "weight": component.weight,
                "weighted_points": component.weighted_points,
                "lost_points": component.lost_points,
                "detail": component.detail,
            }
            for component in score.components
        ],
    }
