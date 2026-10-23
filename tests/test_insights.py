"""core.insights ve core.data_quality_score testleri."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.data_quality_score import (
    COMPONENT_WEIGHTS,
    compute_quality_score,
    score_breakdown_dict,
    score_label,
)
from core.insights import (
    CATEGORIES,
    SEVERITIES,
    filter_insights,
    generate_insights,
    insight_counts,
    insights_to_dataframe,
)


def _categories(insights: list[dict]) -> set[str]:
    return {str(item["category"]) for item in insights}


def test_detects_missing_levels() -> None:
    frame = pd.DataFrame(
        {
            "low": [1.0] * 99 + [np.nan],  # %1
            "medium": [1.0] * 90 + [np.nan] * 10,  # %10
            "high": [1.0] * 50 + [np.nan] * 50,  # %50
        }
    )
    insights = generate_insights(frame)
    by_column = {item["column"]: item for item in insights if item["category"] == "missing"}
    assert by_column["low"]["severity"] == "info"
    assert by_column["medium"]["severity"] == "warning"
    assert by_column["high"]["severity"] == "critical"


def test_detects_constant_column() -> None:
    frame = pd.DataFrame({"sabit": ["X"] * 50, "degisken": range(50)})
    insights = generate_insights(frame)
    constant = [i for i in insights if i["category"] == "constant"]
    assert len(constant) == 1 and constant[0]["column"] == "sabit"


def test_detects_potential_id() -> None:
    frame = pd.DataFrame({"id": range(100), "grup": ["A", "B"] * 50})
    insights = generate_insights(frame)
    ids = [i for i in insights if i["category"] == "identifier"]
    assert any(item["column"] == "id" for item in ids)
    assert "tahmin" in ids[0]["message"].lower()


def test_detects_category_imbalance() -> None:
    frame = pd.DataFrame({"segment": ["A"] * 90 + ["B"] * 10})
    insights = generate_insights(frame)
    imbalance = [i for i in insights if i["category"] == "imbalance"]
    assert len(imbalance) == 1
    assert imbalance[0]["metric"] >= 80


def test_detects_skewness() -> None:
    values = list(range(100)) + [100_000, 120_000, 150_000]
    frame = pd.DataFrame({"gelir": values})
    insights = generate_insights(frame)
    assert "distribution" in _categories(insights)


def test_detects_high_correlation() -> None:
    base = np.arange(100, dtype=float)
    frame = pd.DataFrame({"x": base, "y": base * 3 + 1})
    insights = generate_insights(frame)
    correlation = [i for i in insights if i["category"] == "correlation"]
    assert correlation and abs(correlation[0]["metric"]) >= 0.80
    assert "nedensellik" in correlation[0]["message"].lower()


def test_detects_duplicates() -> None:
    frame = pd.DataFrame({"a": [1, 1, 2, 2, 3], "b": ["x", "x", "y", "y", "z"]})
    insights = generate_insights(frame)
    assert "duplicates" in _categories(insights)


def test_detects_outlier_density() -> None:
    frame = pd.DataFrame({"v": [10] * 90 + [10_000] * 10})
    insights = generate_insights(frame)
    assert "outliers" in _categories(insights)


def test_detects_high_cardinality() -> None:
    frame = pd.DataFrame(
        {"kod": [f"K{i}" for i in range(60)] + [f"K{i}" for i in range(40)]}
    )
    insights = generate_insights(frame)
    assert {"cardinality", "identifier"} & _categories(insights)


def test_empty_dataframe_returns_critical() -> None:
    insights = generate_insights(pd.DataFrame())
    assert len(insights) == 1 and insights[0]["severity"] == "critical"


def test_insights_are_deterministic() -> None:
    frame = pd.DataFrame({"a": [1, 2, None, 4] * 10, "b": ["x"] * 40})
    first = generate_insights(frame)
    second = generate_insights(frame)
    assert first == second


def test_insight_objects_have_expected_shape() -> None:
    frame = pd.DataFrame({"a": [1, None, 3], "sabit": [1, 1, 1]})
    for item in generate_insights(frame):
        assert set(item) == {"severity", "category", "column", "title", "message", "metric", "rule"}
        assert item["severity"] in SEVERITIES
        assert item["category"] in CATEGORIES


def test_sorted_by_severity() -> None:
    frame = pd.DataFrame(
        {"tam_bos": [None] * 20, "az_bos": [1.0] * 19 + [None], "sabit": ["X"] * 20}
    )
    severities = [item["severity"] for item in generate_insights(frame)]
    order = {"critical": 0, "warning": 1, "info": 2}
    assert severities == sorted(severities, key=lambda s: order[s])


def test_filter_and_count_helpers() -> None:
    frame = pd.DataFrame({"a": [1.0] * 19 + [None], "sabit": ["X"] * 20})
    insights = generate_insights(frame)
    counts = insight_counts(insights)
    assert sum(counts.values()) == len(insights)
    only_warnings = filter_insights(insights, ["warning"])
    assert all(item["severity"] == "warning" for item in only_warnings)
    only_missing = filter_insights(insights, None, ["missing"])
    assert all(item["category"] == "missing" for item in only_missing)


def test_insights_dataframe_columns() -> None:
    table = insights_to_dataframe(generate_insights(pd.DataFrame({"a": [1, None]})))
    assert list(table.columns) == [
        "Önem", "Kategori", "Sütun", "Başlık", "Mesaj", "Metrik", "Kural"
    ]


# ---------------------------------------------------------------------------
# Data Quality Score
# ---------------------------------------------------------------------------


def test_perfect_dataset_scores_high() -> None:
    frame = pd.DataFrame({"a": range(100), "b": [f"v{i % 7}" for i in range(100)]})
    score = compute_quality_score(frame)
    assert score.overall >= 90
    assert score.label in ("Çok iyi", "İyi")


def test_missing_values_reduce_score() -> None:
    clean = pd.DataFrame({"a": [1.0] * 100})
    dirty = pd.DataFrame({"a": [1.0] * 50 + [None] * 50})
    assert compute_quality_score(dirty).overall < compute_quality_score(clean).overall


def test_duplicates_reduce_score() -> None:
    unique = pd.DataFrame({"a": range(100)})
    duplicated = pd.DataFrame({"a": [1] * 100})
    assert compute_quality_score(duplicated).overall < compute_quality_score(unique).overall


def test_component_weights_sum_to_100() -> None:
    assert sum(COMPONENT_WEIGHTS.values()) == 100
    score = compute_quality_score(pd.DataFrame({"a": range(10)}))
    assert sum(component.weight for component in score.components) == 100


def test_score_is_bounded_and_explainable() -> None:
    frame = pd.DataFrame({"bos": [None] * 50, "sabit": ["X"] * 50})
    score = compute_quality_score(frame)
    assert 0 <= score.overall <= 100
    for component in score.components:
        assert 0 <= component.score <= 100
        assert component.detail
    table = score.to_dataframe()
    assert list(table.columns) == [
        "Bileşen", "Alt Skor", "Ağırlık", "Katkı (puan)", "Kayıp (puan)", "Açıklama"
    ]


def test_empty_dataframe_scores_zero() -> None:
    score = compute_quality_score(pd.DataFrame())
    assert score.overall == 0


def test_score_breakdown_dict() -> None:
    breakdown = score_breakdown_dict(compute_quality_score(pd.DataFrame({"a": [1, 2]})))
    assert breakdown["overall"] == int(breakdown["overall"])
    assert len(breakdown["components"]) == len(COMPONENT_WEIGHTS)


def test_score_labels() -> None:
    assert score_label(95) == "Çok iyi"
    assert score_label(80) == "İyi"
    assert score_label(65) == "Orta"
    assert score_label(45) == "Zayıf"
    assert score_label(10) == "Sorunlu"
