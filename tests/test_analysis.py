"""Filtre, group by, istatistik, korelasyon, aykırı değer ve grafik testleri."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core import profiler
from core.chart_builder import CHART_TYPES, build_chart, default_params, suggest_charts
from core.correlations import (
    correlation_matrix,
    interpret_strength,
    pair_correlation,
    strong_pairs,
)
from core.filters import (
    apply_filters,
    apply_groupby,
    describe_filters,
    filters_to_code,
    groupby_code,
    operators_for_column,
)
from core.outliers import detect_outliers, iqr_bounds, outlier_summary_table
from core.statistics import (
    categorical_summary,
    datetime_summary,
    describe_all_numeric,
    frequency_table,
    numeric_summary,
)
from utils.validators import ChartError, FilterError, ValidationError


@pytest.fixture
def shop_df() -> pd.DataFrame:
    """Filtre/groupby testleri için küçük bir satış tablosu."""
    return pd.DataFrame(
        {
            "sehir": ["Ankara", "Izmir", "Ankara", "Bursa", "Izmir"],
            "gelir": [1000.0, 2000.0, 3000.0, 4000.0, np.nan],
            "tarih": pd.to_datetime(
                ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01"]
            ),
            "aktif": [True, False, True, True, False],
        }
    )


# ---------------------------------------------------------------------------
# Filtreler
# ---------------------------------------------------------------------------


def test_numeric_operators(shop_df: pd.DataFrame) -> None:
    # Boş değerli satır hiçbir karşılaştırmaya dahil edilmez (ne dahil).
    cases = {
        "gt": 2, "gte": 3, "lt": 1, "lte": 2, "eq": 1, "ne": 3,
    }
    for operator, expected in cases.items():
        result = apply_filters(shop_df, [{"column": "gelir", "operator": operator, "value": 2000}])
        assert len(result) == expected, operator


def test_between_and_null_operators(shop_df: pd.DataFrame) -> None:
    result = apply_filters(
        shop_df, [{"column": "gelir", "operator": "between", "value": 1500, "value2": 3500}]
    )
    assert result["gelir"].tolist() == [2000.0, 3000.0]
    assert len(apply_filters(shop_df, [{"column": "gelir", "operator": "is_null"}])) == 1
    assert len(apply_filters(shop_df, [{"column": "gelir", "operator": "not_null"}])) == 4


def test_text_operators(shop_df: pd.DataFrame) -> None:
    assert len(apply_filters(shop_df, [{"column": "sehir", "operator": "equals", "value": "Ankara"}])) == 2
    assert len(apply_filters(shop_df, [{"column": "sehir", "operator": "contains", "value": "izm"}])) == 2
    assert len(apply_filters(shop_df, [{"column": "sehir", "operator": "is_in", "value": ["Ankara", "Bursa"]}])) == 3
    assert len(apply_filters(shop_df, [{"column": "sehir", "operator": "not_in", "value": ["Ankara"]}])) == 3


def test_datetime_operators(shop_df: pd.DataFrame) -> None:
    assert len(apply_filters(shop_df, [{"column": "tarih", "operator": "before", "value": "2024-03-01"}])) == 2
    assert len(apply_filters(shop_df, [{"column": "tarih", "operator": "after", "value": "2024-03-01"}])) == 2
    assert len(
        apply_filters(
            shop_df,
            [{"column": "tarih", "operator": "between", "value": "2024-02-01", "value2": "2024-04-01"}],
        )
    ) == 3


def test_missing_values_excluded_from_comparisons(shop_df: pd.DataFrame) -> None:
    """Boş gelir, '!= 2000' veya 'içermez' filtrelerine dahil edilmemeli."""
    result = apply_filters(shop_df, [{"column": "gelir", "operator": "ne", "value": 2000}])
    assert result["gelir"].isna().sum() == 0
    assert len(result) == 3

    text_result = apply_filters(
        shop_df, [{"column": "gelir", "operator": "not_null"}]
    )
    assert len(text_result) == 4


def test_filters_are_combined_with_and(shop_df: pd.DataFrame) -> None:
    result = apply_filters(
        shop_df,
        [
            {"column": "sehir", "operator": "equals", "value": "Ankara"},
            {"column": "gelir", "operator": "gt", "value": 2000},
        ],
    )
    assert result["gelir"].tolist() == [3000.0]


def test_filtering_does_not_mutate_source(shop_df: pd.DataFrame) -> None:
    snapshot = shop_df.copy(deep=True)
    apply_filters(shop_df, [{"column": "gelir", "operator": "gt", "value": 1}])
    pd.testing.assert_frame_equal(shop_df, snapshot)


def test_invalid_filter_messages(shop_df: pd.DataFrame) -> None:
    with pytest.raises(FilterError):
        apply_filters(shop_df, [{"column": "yok", "operator": "gt", "value": 1}])
    with pytest.raises(FilterError):
        apply_filters(shop_df, [{"column": "gelir", "operator": "contains", "value": "x"}])
    with pytest.raises(FilterError):
        apply_filters(shop_df, [{"column": "gelir", "operator": "gt", "value": "abc"}])
    with pytest.raises(FilterError):
        apply_filters(shop_df, [{"column": "tarih", "operator": "before", "value": "tarih-degil"}])


def test_operators_depend_on_column_type(shop_df: pd.DataFrame) -> None:
    assert "between" in operators_for_column(shop_df, "gelir")
    assert "contains" in operators_for_column(shop_df, "sehir")
    assert "before" in operators_for_column(shop_df, "tarih")


@pytest.mark.parametrize(
    "filters",
    [
        [{"column": "sehir", "operator": "is_in", "value": ["Ankara", "Izmir"]},
         {"column": "gelir", "operator": "between", "value": 500, "value2": 3500}],
        [{"column": "gelir", "operator": "ne", "value": 2000}],
        [{"column": "sehir", "operator": "not_in", "value": ["Ankara"]}],
        [{"column": "sehir", "operator": "not_contains", "value": "izm"}],
        [{"column": "gelir", "operator": "is_null"}],
        [{"column": "tarih", "operator": "after", "value": "2024-02-15"}],
    ],
)
def test_filter_code_matches_runtime_result(
    shop_df: pd.DataFrame, filters: list[dict]
) -> None:
    """Üretilen filtre kodu, uygulamadaki sonucun aynısını vermelidir."""
    code = filters_to_code(filters)
    namespace = {"df": shop_df.copy(), "pd": pd}
    exec(compile(code, "<filter>", "exec"), namespace)  # noqa: S102 - test amaçlı
    pd.testing.assert_frame_equal(
        namespace["df"].reset_index(drop=True),
        apply_filters(shop_df, filters).reset_index(drop=True),
    )


def test_filter_code_is_executable(shop_df: pd.DataFrame) -> None:
    filters = [
        {"column": "sehir", "operator": "is_in", "value": ["Ankara", "Izmir"]},
        {"column": "gelir", "operator": "between", "value": 500, "value2": 3500},
    ]
    code = filters_to_code(filters)
    namespace = {"df": shop_df.copy(), "pd": pd}
    exec(compile(code, "<filter>", "exec"), namespace)  # noqa: S102 - test amaçlı
    pd.testing.assert_frame_equal(
        namespace["df"].reset_index(drop=True),
        apply_filters(shop_df, filters).reset_index(drop=True),
    )


def test_describe_filters_text(shop_df: pd.DataFrame) -> None:
    text = describe_filters([{"column": "gelir", "operator": "gt", "value": 100}])
    assert "gelir" in text
    assert describe_filters([]) == "Filtre yok"


# ---------------------------------------------------------------------------
# Group by
# ---------------------------------------------------------------------------


def test_groupby_mean(shop_df: pd.DataFrame) -> None:
    result = apply_groupby(shop_df, ["sehir"], ["gelir"], "mean")
    values = dict(zip(result["sehir"], result["gelir"]))
    assert values["Ankara"] == pytest.approx(2000.0)


def test_groupby_count_without_value_column(shop_df: pd.DataFrame) -> None:
    result = apply_groupby(shop_df, ["sehir"], [], "count")
    assert set(result.columns) == {"sehir", "count"}
    assert int(result.loc[result["sehir"] == "Ankara", "count"].iloc[0]) == 2


def test_groupby_rejects_non_numeric_mean(shop_df: pd.DataFrame) -> None:
    with pytest.raises(ValidationError):
        apply_groupby(shop_df, ["sehir"], ["tarih"], "mean")


def test_groupby_requires_group_column(shop_df: pd.DataFrame) -> None:
    with pytest.raises(ValidationError):
        apply_groupby(shop_df, [], ["gelir"], "mean")


def test_groupby_code_is_executable(shop_df: pd.DataFrame) -> None:
    code = groupby_code(["sehir"], ["gelir"], "mean")
    namespace = {"df": shop_df.copy()}
    exec(compile(code, "<groupby>", "exec"), namespace)  # noqa: S102 - test amaçlı
    pd.testing.assert_frame_equal(
        namespace["grouped_df"], apply_groupby(shop_df, ["sehir"], ["gelir"], "mean")
    )


# ---------------------------------------------------------------------------
# İstatistikler
# ---------------------------------------------------------------------------


def test_numeric_summary_values() -> None:
    frame = pd.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, np.nan]})
    summary = numeric_summary(frame, "v")
    assert summary["count"] == 4 and summary["missing"] == 1
    assert summary["mean"] == pytest.approx(2.5)
    assert summary["median"] == pytest.approx(2.5)
    assert summary["iqr"] == pytest.approx(1.5)
    assert summary["range"] == pytest.approx(3.0)


def test_numeric_summary_rejects_text(shop_df: pd.DataFrame) -> None:
    with pytest.raises(ValidationError):
        numeric_summary(shop_df, "sehir")


def test_numeric_summary_on_empty_column() -> None:
    with pytest.raises(ValidationError):
        numeric_summary(pd.DataFrame({"v": [np.nan, np.nan]}), "v")


def test_numeric_summary_handles_inf() -> None:
    frame = pd.DataFrame({"v": [1.0, 2.0, np.inf, -np.inf]})
    summary = numeric_summary(frame, "v")
    assert summary["count"] == 2
    assert np.isfinite(summary["mean"])


def test_categorical_summary(shop_df: pd.DataFrame) -> None:
    summary = categorical_summary(shop_df, "sehir", top_n=2)
    assert summary["unique"] == 3
    assert summary["mode"] in ("Ankara", "Izmir")
    assert len(summary["top_values"]) == 2


def test_frequency_table_percentages(shop_df: pd.DataFrame) -> None:
    table = frequency_table(shop_df, "sehir", 10)
    assert table["Yüzde %"].sum() == pytest.approx(100.0, abs=0.01)
    assert table["Kümülatif %"].iloc[-1] == pytest.approx(100.0, abs=0.01)


def test_datetime_summary(shop_df: pd.DataFrame) -> None:
    summary = datetime_summary(shop_df, "tarih")
    assert summary["count"] == 5
    assert summary["range_days"] == 121
    assert not summary["by_year"].empty
    assert len(summary["by_weekday"]) >= 1


def test_datetime_summary_on_text_column_converts() -> None:
    frame = pd.DataFrame({"t": ["2024-01-01", "2024-06-01"]})
    summary = datetime_summary(frame, "t")
    assert summary["count"] == 2


def test_describe_all_numeric(shop_df: pd.DataFrame) -> None:
    table = describe_all_numeric(shop_df)
    assert "gelir" in table["Sütun"].tolist()


# ---------------------------------------------------------------------------
# Korelasyon
# ---------------------------------------------------------------------------


def test_correlation_matrix_shape(numeric_df: pd.DataFrame) -> None:
    matrix = correlation_matrix(numeric_df, ["x", "y", "z"], "pearson")
    assert matrix.shape == (3, 3)
    assert matrix.loc["x", "x"] == pytest.approx(1.0)
    assert matrix.loc["x", "y"] == pytest.approx(matrix.loc["y", "x"])


def test_spearman_supported(numeric_df: pd.DataFrame) -> None:
    matrix = correlation_matrix(numeric_df, ["x", "y"], "spearman")
    assert abs(matrix.loc["x", "y"]) > 0.9


def test_correlation_requires_two_numeric_columns(shop_df: pd.DataFrame) -> None:
    with pytest.raises(ValidationError):
        correlation_matrix(shop_df, ["gelir"], "pearson")
    with pytest.raises(ValidationError):
        correlation_matrix(shop_df, ["gelir", "sehir"], "pearson")
    with pytest.raises(ValidationError):
        correlation_matrix(shop_df, ["gelir"], "kendall")


def test_strong_pairs_excludes_self_and_duplicates(numeric_df: pd.DataFrame) -> None:
    matrix = correlation_matrix(numeric_df, ["x", "y", "z"], "pearson")
    pairs = strong_pairs(matrix, 0.7)
    assert not pairs.empty
    for _, row in pairs.iterrows():
        assert row["Sütun 1"] != row["Sütun 2"]
    seen = {frozenset((row["Sütun 1"], row["Sütun 2"])) for _, row in pairs.iterrows()}
    assert len(seen) == len(pairs)


def test_strong_pairs_threshold(numeric_df: pd.DataFrame) -> None:
    matrix = correlation_matrix(numeric_df, ["x", "y", "z"], "pearson")
    assert strong_pairs(matrix, 0.99999).empty


def test_pair_correlation(numeric_df: pd.DataFrame) -> None:
    result = pair_correlation(numeric_df, "x", "y")
    assert result["n"] == len(numeric_df)
    assert result["strength"] in ("Çok güçlü", "Güçlü")


def test_interpret_strength_labels() -> None:
    assert interpret_strength(0.95) == "Çok güçlü"
    assert interpret_strength(-0.75) == "Güçlü"
    assert interpret_strength(0.55) == "Orta"
    assert interpret_strength(0.35) == "Zayıf"
    assert interpret_strength(0.05) == "Çok zayıf"


# ---------------------------------------------------------------------------
# Aykırı değerler
# ---------------------------------------------------------------------------


def test_iqr_bounds_known_values() -> None:
    series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    q1, q3, iqr, lower, upper = iqr_bounds(series, 1.5)
    assert q1 == pytest.approx(3.25)
    assert q3 == pytest.approx(7.75)
    assert iqr == pytest.approx(4.5)
    assert lower == pytest.approx(-3.5)
    assert upper == pytest.approx(14.5)


def test_detect_outliers_counts() -> None:
    frame = pd.DataFrame({"v": [10, 11, 12, 13, 14, 15, 1000, -900]})
    report = detect_outliers(frame, "v", 1.5)
    assert report.count == 2
    assert report.lower_count == 1 and report.upper_count == 1
    assert report.ratio == pytest.approx(25.0)
    assert report.mask.sum() == 2


def test_detect_outliers_rejects_non_numeric(shop_df: pd.DataFrame) -> None:
    with pytest.raises(ValidationError):
        detect_outliers(shop_df, "sehir")


def test_detect_outliers_on_constant_column() -> None:
    frame = pd.DataFrame({"v": [5] * 10})
    report = detect_outliers(frame, "v")
    assert report.count == 0 and report.iqr == 0


def test_outlier_factor_affects_result() -> None:
    frame = pd.DataFrame({"v": [10, 11, 12, 13, 14, 15, 40]})
    assert detect_outliers(frame, "v", 1.5).count >= detect_outliers(frame, "v", 3.0).count


def test_outlier_summary_table(numeric_df: pd.DataFrame) -> None:
    table = outlier_summary_table(numeric_df, None, 1.5)
    assert set(table["Sütun"]) <= {"x", "y", "z"}
    assert "Aykırı %" in table.columns


# ---------------------------------------------------------------------------
# Grafikler
# ---------------------------------------------------------------------------


def test_all_chart_types_build(numeric_df: pd.DataFrame) -> None:
    """Desteklenen her grafik tipi çizilmeli ve kodu derlenebilmeli."""
    frame = numeric_df.assign(
        grup=["A", "B"] * (len(numeric_df) // 2),
        boyut=numeric_df["z"].abs(),
    )
    params = {
        "bar": {"x": "kategori", "aggregation": "count"},
        "line": {"x": "x", "y": "y"},
        "area": {"x": "x", "y": "y"},
        "scatter": {"x": "x", "y": "y", "color": "kategori"},
        "bubble": {"x": "x", "y": "y", "size": "boyut"},
        "histogram": {"column": "x", "bins": 20},
        "box": {"y": "x", "x": "kategori"},
        "violin": {"y": "x", "x": "kategori"},
        "pie": {"names": "kategori", "aggregation": "count"},
        "donut": {"names": "kategori", "aggregation": "count"},
        "heatmap": {"columns": ["x", "y", "z"], "method": "pearson"},
        "category_heatmap": {"x": "kategori", "y": "grup"},
        "density_heatmap": {"x": "x", "y": "y", "bins": 20},
    }
    assert set(params) == set(CHART_TYPES)
    for chart_type, chart_params in params.items():
        result = build_chart(frame, chart_type, chart_params)
        assert result.figure is not None
        assert result.spec["type"] == chart_type
        assert result.spec["python_code"]
        compile(result.spec["python_code"].replace("fig.show()", "pass"), "<chart>", "exec")


def test_bar_modes_and_orientation(numeric_df: pd.DataFrame) -> None:
    """Gruplu/yığılmış ve yatay çubuk seçenekleri çalışmalı."""
    for mode in ("group", "stack", "overlay"):
        result = build_chart(
            numeric_df,
            "bar",
            {"x": "kategori", "y": "x", "aggregation": "mean", "color": "kategori",
             "barmode": mode},
        )
        assert result.figure.layout.barmode == mode

    horizontal = build_chart(
        numeric_df,
        "bar",
        {"x": "kategori", "y": "x", "aggregation": "mean", "horizontal": True},
    )
    assert horizontal.figure.data[0].orientation == "h"


def test_chart_customization_applied(numeric_df: pd.DataFrame) -> None:
    """Başlık, eksen etiketleri ve legend ayarları uygulanmalı."""
    result = build_chart(
        numeric_df,
        "scatter",
        {
            "x": "x",
            "y": "y",
            "title": "Özel Başlık",
            "x_label": "X Ekseni",
            "y_label": "Y Ekseni",
            "show_legend": False,
        },
    )
    layout = result.figure.layout
    assert layout.title.text == "Özel Başlık"
    assert layout.xaxis.title.text == "X Ekseni"
    assert layout.yaxis.title.text == "Y Ekseni"
    assert layout.showlegend is False
    assert "update_layout" in result.spec["python_code"]


def test_bar_sorting_and_top_n() -> None:
    """Değere göre sıralama ve Top N uygulanmalı."""
    frame = pd.DataFrame({"k": list("aaabbcccd"), "v": [1] * 9})
    result = build_chart(frame, "bar", {"x": "k", "aggregation": "count", "top_n": 2})
    assert len(result.figure.data[0].x) == 2

    ascending = build_chart(
        frame, "bar", {"x": "k", "aggregation": "count", "sort": "value_asc"}
    )
    values = list(ascending.figure.data[0].y)
    assert values == sorted(values)


def test_chart_suggestions_match_column_types() -> None:
    """Öneriler sütun tiplerine uygun olmalı."""
    frame = pd.DataFrame(
        {
            "tarih": pd.date_range("2024-01-01", periods=30, freq="D"),
            "sayi1": range(30),
            "sayi2": range(30, 60),
            "kategori": ["A", "B", "C"] * 10,
        }
    )
    suggestions = suggest_charts(frame)
    assert suggestions
    types = {item["chart_type"] for item in suggestions}
    assert "line" in types  # datetime + numeric
    assert "scatter" in types  # numeric + numeric
    assert {"bar", "box"} & types  # categorical + numeric
    assert "histogram" in types  # single numeric
    for item in suggestions:
        assert item["reason"]
        build_chart(frame, item["chart_type"], item["params"])


def test_default_params_are_usable() -> None:
    """Smart defaults ile her grafik tipi doğrudan çizilebilmeli."""
    frame = pd.DataFrame(
        {
            "tarih": pd.date_range("2024-01-01", periods=20, freq="D"),
            "a": range(20),
            "b": range(20, 40),
            "k1": ["x", "y"] * 10,
            "k2": ["p", "q", "r", "s"] * 5,
        }
    )
    for chart_type in CHART_TYPES:
        params = default_params(frame, chart_type)
        assert params, chart_type
        build_chart(frame, chart_type, params)


def test_suggestions_empty_for_empty_frame() -> None:
    assert suggest_charts(pd.DataFrame()) == []


def test_chart_rejects_wrong_column_type(numeric_df: pd.DataFrame) -> None:
    with pytest.raises(ChartError):
        build_chart(numeric_df, "histogram", {"column": "kategori"})
    with pytest.raises(ChartError):
        build_chart(numeric_df, "scatter", {"x": "kategori", "y": "y"})


def test_chart_rejects_missing_params(numeric_df: pd.DataFrame) -> None:
    with pytest.raises(ChartError):
        build_chart(numeric_df, "scatter", {"x": "x"})
    with pytest.raises(ChartError):
        build_chart(numeric_df, "bar", {})


def test_chart_rejects_unknown_type_and_empty_data(numeric_df: pd.DataFrame) -> None:
    with pytest.raises(ChartError):
        build_chart(numeric_df, "radar", {"x": "x"})
    with pytest.raises(ChartError):
        build_chart(numeric_df.head(0), "histogram", {"column": "x"})


def test_chart_code_uses_custom_data_name(numeric_df: pd.DataFrame) -> None:
    result = build_chart(
        numeric_df, "scatter", {"x": "x", "y": "y"}, data_name="grouped_df",
        code_prefix="grouped_df = df.copy()",
    )
    assert "grouped_df" in result.spec["python_code"]
    assert result.spec["python_code"].startswith("grouped_df = df.copy()")


def test_bar_aggregation_code_matches_result(numeric_df: pd.DataFrame) -> None:
    result = build_chart(numeric_df, "bar", {"x": "kategori", "y": "x", "aggregation": "mean"})
    namespace = {"df": numeric_df.copy(), "px": __import__("plotly.express", fromlist=["x"])}
    exec(  # noqa: S102 - test amaçlı
        compile(result.spec["python_code"].replace("fig.show()", "pass"), "<chart>", "exec"),
        namespace,
    )
    assert len(namespace["plot_df"]) == numeric_df["kategori"].nunique()


def test_large_dataset_is_sampled() -> None:
    frame = pd.DataFrame({"x": np.arange(30_000), "y": np.arange(30_000)})
    result = build_chart(frame, "scatter", {"x": "x", "y": "y"})
    assert result.warnings
    assert "örnekleme" in result.warnings[0].lower()


def test_profiler_tables(messy_df: pd.DataFrame) -> None:
    assert not profiler.missing_value_table(messy_df).empty
    assert profiler.duplicate_report(messy_df)["count"] == 1
    assert "sabit" in profiler.constant_column_table(messy_df)["Sütun"].tolist()
    assert not profiler.unique_value_table(messy_df).empty
    findings = profiler.dtype_check_table(messy_df)
    assert "sayi_metin" in findings["Sütun"].tolist()
    assert "tarih_metin" in findings["Sütun"].tolist()
