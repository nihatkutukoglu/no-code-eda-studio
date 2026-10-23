"""core.pivot, core.ml_preparation, core.report ve yeni export testleri."""

from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import pytest

from core.exporter import (
    DATA_FORMATS,
    dataframe_to_bytes,
    dataframe_to_json_bytes,
    dataframe_to_parquet_bytes,
    insights_to_json_bytes,
    report_filename,
    report_to_bytes,
)
from core.insights import generate_insights
from core.ml_preparation import (
    MAX_CLASSES,
    analyze_readiness,
    detect_task_type,
    full_ml_code,
    pipeline_code,
    train_test_code,
)
from core.pivot import (
    CROSSTAB_NORMALIZE,
    PIVOT_AGGREGATIONS,
    build_crosstab,
    build_pivot,
    chart_ready_pivot,
    crosstab_code,
    pivot_code,
)
from core.report import build_html_report
from utils.validators import EdaError, ValidationError


@pytest.fixture
def sales_df() -> pd.DataFrame:
    """Pivot/crosstab testleri için satış tablosu."""
    rng = np.random.default_rng(11)
    size = 120
    return pd.DataFrame(
        {
            "sehir": rng.choice(["Ankara", "Izmir", "Bursa"], size),
            "cinsiyet": rng.choice(["K", "E"], size),
            "segment": rng.choice(["A", "B", "C"], size),
            "satis": rng.integers(100, 1000, size).astype(float),
            "adet": rng.integers(1, 10, size),
            "hedef": rng.choice([0, 1], size, p=[0.65, 0.35]),
        }
    )


# ---------------------------------------------------------------------------
# Pivot
# ---------------------------------------------------------------------------


def test_pivot_mean_values(sales_df: pd.DataFrame) -> None:
    table = build_pivot(sales_df, ["sehir"], ["cinsiyet"], "satis", "mean")
    assert "sehir" in table.columns
    assert table.shape[0] == sales_df["sehir"].nunique()
    expected = sales_df[(sales_df["sehir"] == "Ankara") & (sales_df["cinsiyet"] == "K")][
        "satis"
    ].mean()
    assert float(table.loc[table["sehir"] == "Ankara", "K"].iloc[0]) == pytest.approx(expected)


def test_pivot_count_without_values(sales_df: pd.DataFrame) -> None:
    table = build_pivot(sales_df, ["sehir"], None, None, "count")
    assert int(table["adet"].sum()) == len(sales_df)


def test_pivot_all_aggregations(sales_df: pd.DataFrame) -> None:
    for aggregation in PIVOT_AGGREGATIONS:
        values = None if aggregation in ("count", "nunique") else "satis"
        table = build_pivot(sales_df, ["sehir"], None, values, aggregation)
        assert not table.empty


def test_pivot_margins_and_fill(sales_df: pd.DataFrame) -> None:
    table = build_pivot(
        sales_df, ["sehir"], ["segment"], "satis", "sum", fill_value=0, margins=True
    )
    assert (table["sehir"] == "Toplam").any()
    assert not table.isna().any().any()


def test_pivot_validation(sales_df: pd.DataFrame) -> None:
    with pytest.raises(ValidationError):
        build_pivot(sales_df, [], None, "satis", "mean")
    with pytest.raises(ValidationError):
        build_pivot(sales_df, ["sehir"], None, None, "mean")  # values gerekli
    with pytest.raises(ValidationError):
        build_pivot(sales_df, ["sehir"], ["sehir"], "satis", "mean")  # çakışma
    with pytest.raises(ValidationError):
        build_pivot(sales_df, ["yok"], None, None, "count")
    with pytest.raises(ValidationError):
        build_pivot(sales_df, ["sehir"], None, "cinsiyet", "mean")  # sayısal değil


def test_pivot_code_matches_result(sales_df: pd.DataFrame) -> None:
    code = pivot_code(["sehir"], ["cinsiyet"], "satis", "mean")
    namespace = {"df": sales_df.copy(), "pd": pd}
    exec(compile(code, "<pivot>", "exec"), namespace)  # noqa: S102 - test amaçlı
    produced = namespace["pivot_df"]
    expected = build_pivot(sales_df, ["sehir"], ["cinsiyet"], "satis", "mean")
    assert produced.shape == expected.shape


def test_pivot_count_code_runs(sales_df: pd.DataFrame) -> None:
    code = pivot_code(["sehir"], None, None, "count")
    namespace = {"df": sales_df.copy(), "pd": pd}
    exec(compile(code, "<pivot>", "exec"), namespace)  # noqa: S102 - test amaçlı
    assert not namespace["pivot_df"].empty


def test_crosstab_counts_and_percentages(sales_df: pd.DataFrame) -> None:
    counts = build_crosstab(sales_df, "sehir", "cinsiyet", "count", margins=False)
    numeric_columns = [c for c in counts.columns if c != "sehir"]
    assert counts[numeric_columns].to_numpy().sum() == len(sales_df)

    rows = build_crosstab(sales_df, "sehir", "cinsiyet", "row", margins=False)
    row_values = rows[[c for c in rows.columns if c != "sehir"]].sum(axis=1)
    assert np.allclose(row_values.to_numpy(), 100.0, atol=0.2)


def test_crosstab_all_normalizations(sales_df: pd.DataFrame) -> None:
    for normalize in CROSSTAB_NORMALIZE:
        table = build_crosstab(sales_df, "sehir", "segment", normalize)
        assert not table.empty


def test_crosstab_validation(sales_df: pd.DataFrame) -> None:
    with pytest.raises(ValidationError):
        build_crosstab(sales_df, "sehir", "sehir")
    with pytest.raises(ValidationError):
        build_crosstab(sales_df, "sehir", "cinsiyet", "yanlis")
    high_card = sales_df.assign(kod=[f"K{i}" for i in range(len(sales_df))])
    with pytest.raises(ValidationError):
        build_crosstab(high_card, "kod", "cinsiyet")


def test_crosstab_code_runs(sales_df: pd.DataFrame) -> None:
    for normalize in ("count", "row"):
        code = crosstab_code("sehir", "cinsiyet", normalize, margins=False)
        namespace = {"df": sales_df.copy(), "pd": pd}
        exec(compile(code, "<crosstab>", "exec"), namespace)  # noqa: S102 - test amaçlı
        assert not namespace["crosstab_df"].empty


def test_chart_ready_pivot(sales_df: pd.DataFrame) -> None:
    table = build_pivot(sales_df, ["sehir"], ["cinsiyet"], "satis", "mean")
    long_format = chart_ready_pivot(table)
    assert list(long_format.columns) == ["kategori", "seri", "deger"]
    assert len(long_format) == table.shape[0] * 2


def test_chart_ready_pivot_requires_numeric() -> None:
    with pytest.raises(ValidationError):
        chart_ready_pivot(pd.DataFrame({"a": ["x"], "b": ["y"]}))


# ---------------------------------------------------------------------------
# ML hazırlığı
# ---------------------------------------------------------------------------


def test_detect_task_type() -> None:
    assert detect_task_type(pd.Series([0, 1, 1, 0])) == "binary_classification"
    assert detect_task_type(pd.Series(["a", "b", "c"] * 10)) == "multiclass_classification"
    assert detect_task_type(pd.Series(np.random.default_rng(0).normal(size=200))) == "regression"
    assert detect_task_type(pd.Series([np.nan, np.nan])) == "unknown"
    assert detect_task_type(pd.Series([5, 5, 5])) == "unknown"
    assert detect_task_type(pd.Series([True, False, True])) == "binary_classification"


def test_readiness_basic(sales_df: pd.DataFrame) -> None:
    report = analyze_readiness(sales_df, "hedef")
    assert report.task_type == "binary_classification"
    assert report.is_ready
    assert "satis" in report.numeric_features
    assert "sehir" in report.categorical_features
    assert report.target_classes
    assert 0 < report.imbalance_ratio <= 1


def test_readiness_flags_problem_columns(sales_df: pd.DataFrame) -> None:
    frame = sales_df.assign(
        sabit="X",
        musteri_id=[f"ID{i:05d}" for i in range(len(sales_df))],
        hedef_kopya=sales_df["hedef"],
    )
    frame.loc[frame.index[:20], "satis"] = np.nan
    report = analyze_readiness(frame, "hedef")

    assert "sabit" in report.constant_features
    assert "musteri_id" in report.id_like_features
    assert "hedef_kopya" in [column for column, _ in report.leakage_suspects]
    assert "satis" in [column for column, _ in report.features_with_missing]
    drops = report.recommended_drops()
    assert {"sabit", "musteri_id", "hedef_kopya"} <= set(drops)
    assert "satis" not in drops  # sürekli ölçüm yanlışlıkla işaretlenmemeli
    assert not report.summary_table().empty


def test_readiness_target_missing_blocks(sales_df: pd.DataFrame) -> None:
    frame = sales_df.copy()
    frame.loc[frame.index[: len(frame) // 2 + 5], "hedef"] = np.nan
    report = analyze_readiness(frame, "hedef")
    assert report.blockers


def test_readiness_validation(sales_df: pd.DataFrame) -> None:
    with pytest.raises(ValidationError):
        analyze_readiness(sales_df, "yok")
    with pytest.raises(ValidationError):
        analyze_readiness(pd.DataFrame(), "hedef")
    with pytest.raises(ValidationError):
        analyze_readiness(sales_df, "hedef", features=["hedef"])


def test_readiness_warns_on_imbalance() -> None:
    frame = pd.DataFrame(
        {"x": range(100), "hedef": [1] * 95 + [0] * 5}
    )
    report = analyze_readiness(frame, "hedef")
    assert any("dengesiz" in message for message in report.warnings)


def test_readiness_small_dataset_warning() -> None:
    frame = pd.DataFrame({"x": range(10), "hedef": [0, 1] * 5})
    report = analyze_readiness(frame, "hedef")
    assert any("satır" in message for message in report.warnings)


def test_train_test_code_runs(sales_df: pd.DataFrame) -> None:
    code = train_test_code("hedef", ["satis", "adet"], 0.25, 7, stratify=True)
    namespace = {"df": sales_df.copy(), "pd": pd}
    exec(compile(code, "<split>", "exec"), namespace)  # noqa: S102 - test amaçlı
    assert len(namespace["X_test"]) == pytest.approx(len(sales_df) * 0.25, abs=2)
    assert list(namespace["X_train"].columns) == ["satis", "adet"]


def test_pipeline_code_runs(sales_df: pd.DataFrame) -> None:
    split = train_test_code("hedef", ["satis", "sehir"], 0.2, 42)
    pipeline = pipeline_code(["satis"], ["sehir"], "median", "standard", "most_frequent", "onehot")
    namespace = {"df": sales_df.copy(), "pd": pd}
    exec(compile(split + "\n" + pipeline, "<pipeline>", "exec"), namespace)  # noqa: S102
    assert namespace["X_train_prepared"].shape[0] == len(namespace["X_train"])
    assert namespace["X_train_prepared"].shape[1] >= 3  # 1 sayısal + 3 kategori - 1


def test_pipeline_code_variants() -> None:
    for scaler in ("none", "standard", "minmax", "robust"):
        for encoder in ("onehot", "ordinal"):
            code = pipeline_code(["a"], ["b"], "mean", scaler, "constant", encoder)
            compile(code, "<pipeline>", "exec")


def test_pipeline_validation() -> None:
    with pytest.raises(ValidationError):
        pipeline_code([], [])
    with pytest.raises(ValidationError):
        pipeline_code(["a"], [], numeric_scaler="yanlis")


def test_full_ml_code_compiles(sales_df: pd.DataFrame) -> None:
    code = full_ml_code(
        "hedef", ["satis", "sehir"], ["satis"], ["sehir"], drop_columns=["adet"]
    )
    compile(code, "<ml>", "exec")
    assert "drop(columns=" in code
    assert "train_test_split" in code
    assert "ColumnTransformer" in code


def test_max_classes_constant() -> None:
    assert MAX_CLASSES >= 2


# ---------------------------------------------------------------------------
# HTML rapor
# ---------------------------------------------------------------------------


def test_html_report_sections(sales_df: pd.DataFrame) -> None:
    html = build_html_report(
        sales_df,
        meta={"filename": "satislar.csv"},
        history=[{"description": "test işlemi", "action": "fill_missing"}],
    )
    for marker in (
        "Dataset Özeti",
        "Veri Kalitesi",
        "İstatistiksel Özet",
        "Korelasyonlar",
        "Kural Tabanlı Bulgular",
        "Uygulanan Veri İşlemleri",
        "satislar.csv",
    ):
        assert marker in html
    assert html.strip().startswith("<!DOCTYPE html>")
    assert "cdn.plot.ly" in html


def test_html_report_escapes_user_content() -> None:
    frame = pd.DataFrame({"<script>alert(1)</script>": [1, 2], "b": ["<b>x</b>", "y"]})
    html = build_html_report(frame, meta={"filename": "<img src=x>"})
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_html_report_requires_data() -> None:
    with pytest.raises(EdaError):
        build_html_report(pd.DataFrame())


def test_report_bytes_and_filename(sales_df: pd.DataFrame) -> None:
    data = report_to_bytes(sales_df, {"filename": "a.csv"})
    assert data.startswith(b"<!DOCTYPE html>")
    assert report_filename("a.csv") == "a_eda_raporu.html"


# ---------------------------------------------------------------------------
# Yeni export biçimleri
# ---------------------------------------------------------------------------


def test_parquet_round_trip(sales_df: pd.DataFrame) -> None:
    payload = dataframe_to_parquet_bytes(sales_df)
    restored = pd.read_parquet(io.BytesIO(payload))
    assert restored.shape == sales_df.shape


def test_parquet_handles_mixed_types() -> None:
    frame = pd.DataFrame({"m": [1, "iki", 3.0, None]})
    payload = dataframe_to_parquet_bytes(frame)
    assert len(pd.read_parquet(io.BytesIO(payload))) == 4


def test_json_export(sales_df: pd.DataFrame) -> None:
    records = json.loads(dataframe_to_json_bytes(sales_df).decode("utf-8"))
    assert len(records) == len(sales_df)
    assert set(records[0]) == set(sales_df.columns)


def test_all_data_formats(sales_df: pd.DataFrame) -> None:
    for file_format in DATA_FORMATS:
        data = dataframe_to_bytes(sales_df, file_format)
        assert data
    with pytest.raises(EdaError):
        dataframe_to_bytes(sales_df, "xml")


def test_insights_json_export(sales_df: pd.DataFrame) -> None:
    payload = json.loads(insights_to_json_bytes(generate_insights(sales_df)).decode("utf-8"))
    assert isinstance(payload, list)
    if payload:
        assert set(payload[0]) == {
            "severity", "category", "column", "title", "message", "metric", "rule"
        }
