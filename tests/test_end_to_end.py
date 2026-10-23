"""Uçtan uca kullanıcı akışı testi.

Gerçek bir kullanıcının izlediği yolu baştan sona çalıştırır:

    yükle → boş değer doldur → tekrar sil → metin temizle → ölçekle →
    tarih özelliği → yeni sütun → filtre → group by → pivot → grafik →
    korelasyon → aykırı değer → insights → skor → ML hazırlığı →
    kod üret → notebook → rapor → export → undo → reset

Her adım gerçekten çalıştırılır; üretilen kod derlenir ve replay doğrulanır.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import nbformat
import pandas as pd
import pytest

from core import operations as ops
from core.chart_builder import build_chart, suggest_charts
from core.code_generator import generate_full_script
from core.correlations import correlation_matrix, cramers_v, strong_pairs
from core.data_quality_score import compute_quality_score
from core.exporter import (
    dataframe_to_bytes,
    notebook_bytes,
    report_to_bytes,
    script_to_bytes,
)
from core.filters import apply_filters, apply_groupby
from core.insights import generate_insights
from core.loader import load_dataframe
from core.ml_preparation import analyze_readiness, full_ml_code
from core.outliers import detect_outliers
from core.pivot import build_crosstab, build_pivot
from core.profiler import duplicate_report, missing_value_table
from core.statistics import categorical_summary, numeric_summary

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CSV = ROOT / "sample_data" / "ornek_musteriler.csv"


@pytest.mark.parametrize("sample", [SAMPLE_CSV])
def test_full_user_workflow(sample: Path) -> None:
    """Tüm ürün akışı tek bir oturumda hatasız tamamlanmalı."""
    assert sample.exists(), "Örnek veri dosyası bulunamadı"

    # 1) Yükleme
    result = load_dataframe(sample.read_bytes(), sample.name)
    original = result.df
    meta = {
        "filename": result.filename,
        "file_type": result.file_type,
        "separator": result.separator,
        "encoding": result.encoding,
    }
    assert len(original) > 100
    assert result.separator == ";"

    # 2) Kalite analizi
    assert not missing_value_table(original, only_missing=True).empty
    assert duplicate_report(original)["count"] > 0

    # 3) Temizleme + dönüşümler (işlem geçmişiyle)
    steps = [
        ("fill_missing", {"column": "gelir", "method": "median"}),
        ("fill_missing", {"column": "harcama", "method": "mean"}),
        ("drop_duplicates", {}),
        ("drop_columns", {"columns": ["bos_sutun"]}),
        ("convert_dtype", {"column": "uyelik_tarihi", "target": "datetime"}),
        ("text_transform", {"column": "sehir", "method": "trim"}),
        ("numeric_transform", {"column": "gelir", "method": "standard", "new_column": "gelir_z"}),
        ("datetime_features", {"column": "uyelik_tarihi", "features": ["year", "month"]}),
        (
            "arithmetic_feature",
            {"left": "harcama", "right": "gelir", "operation": "ratio", "new_column": "harcama_orani"},
        ),
        ("bin_column", {"column": "yas", "method": "quantile", "bins": 3, "new_column": "yas_grup"}),
        ("encode_categorical", {"column": "segment", "method": "label", "new_column": "segment_kod"}),
        ("sort_rows", {"columns": ["gelir"], "ascending": [False]}),
    ]
    working = original
    history: list[dict] = []
    for action, params in steps:
        working, operation = ops.apply_operation(working, action, params)
        history.append(operation.to_dict())

    assert working["gelir"].isna().sum() == 0
    assert "bos_sutun" not in working.columns
    assert {"gelir_z", "uyelik_tarihi_year", "harcama_orani", "yas_grup", "segment_kod"} <= set(
        working.columns
    )
    assert pd.api.types.is_datetime64_any_dtype(working["uyelik_tarihi"])

    # 4) Replay determinizmi (undo altyapısı)
    replayed, errors = ops.replay(original, history)
    assert errors == []
    pd.testing.assert_frame_equal(replayed, working)

    # 5) Filtre + group by + pivot + crosstab
    filters = [{"column": "gelir", "operator": "gt", "value": 15000}]
    filtered = apply_filters(working, filters)
    assert 0 < len(filtered) <= len(working)

    grouped = apply_groupby(working, ["sehir"], ["gelir"], "mean")
    assert not grouped.empty

    pivot = build_pivot(working, ["sehir"], ["segment"], "gelir", "mean")
    assert not pivot.empty
    crosstab = build_crosstab(working, "sehir", "segment", "row")
    assert not crosstab.empty

    # 6) Grafikler (öneri motoru + manuel)
    suggestions = suggest_charts(working)
    assert suggestions
    charts = [build_chart(working, item["chart_type"], item["params"]).spec for item in suggestions[:3]]
    charts.append(
        build_chart(
            working,
            "bar",
            {"x": "sehir", "y": "gelir", "aggregation": "mean", "sort": "value_desc"},
        ).spec
    )
    assert len(charts) == 4

    # 7) İstatistik, korelasyon, aykırı değer
    assert numeric_summary(working, "gelir")["count"] > 0
    assert categorical_summary(working, "sehir")["unique"] >= 2
    matrix = correlation_matrix(working, ["gelir", "harcama", "yas"], "pearson")
    assert matrix.shape == (3, 3)
    strong_pairs(matrix, 0.7)  # hata vermemeli
    assert cramers_v(working, "sehir", "segment")["v"] >= 0
    outliers = detect_outliers(working, "gelir", 1.5)
    assert outliers.count >= 0

    # 8) Insights + kalite skoru
    insights = generate_insights(working)
    assert insights
    score = compute_quality_score(working)
    assert 0 <= score.overall <= 100

    # 9) ML hazırlığı
    readiness = analyze_readiness(working, "aktif")
    assert readiness.task_type in ("binary_classification", "multiclass_classification")
    ml_code = full_ml_code(
        "aktif",
        ["gelir", "harcama", "sehir"],
        ["gelir", "harcama"],
        ["sehir"],
        drop_columns=readiness.recommended_drops(),
    )
    compile(ml_code, "<ml>", "exec")

    # 10) Kod üretimi: script gerçekten çalışmalı
    script = generate_full_script(
        {**meta, "filename": str(sample)}, history, charts, ml_code=None
    )
    namespace: dict[str, object] = {}
    exec(compile(script, "<script>", "exec"), namespace)  # noqa: S102 - test amaçlı
    produced = namespace["df"]
    assert isinstance(produced, pd.DataFrame)
    assert set(working.columns) <= set(produced.columns)
    assert len(produced) == len(working)

    # 11) Notebook + rapor + export
    notebook = nbformat.reads(
        notebook_bytes(meta, history, charts, insights, ml_code=ml_code).decode("utf-8"),
        as_version=4,
    )
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell["cell_type"] == "code":
            compile(cell["source"].replace("display(", "print("), "<cell>", "exec")

    compile(script_to_bytes(meta, history, charts, ml_code=ml_code).decode("utf-8"), "<s>", "exec")
    assert report_to_bytes(working, meta, charts, history, insights).startswith(b"<!DOCTYPE")

    for file_format in ("csv", "xlsx", "parquet", "json"):
        payload = dataframe_to_bytes(working, file_format)
        assert payload
    restored = pd.read_csv(io.BytesIO(dataframe_to_bytes(working, "csv")), encoding="utf-8-sig")
    assert len(restored) == len(working)
    assert len(json.loads(dataframe_to_bytes(working, "json").decode("utf-8"))) == len(working)

    # 12) Undo ve reset
    undone, remaining, removed, undo_errors = ops.undo_last(original, history)
    assert undo_errors == []
    assert removed["action"] == "sort_rows"
    assert len(remaining) == len(history) - 1

    reset = ops.reset_dataframe(original)
    pd.testing.assert_frame_equal(reset, original)
    # Orijinal veri hiç değişmemiş olmalı
    fresh = load_dataframe(sample.read_bytes(), sample.name).df
    pd.testing.assert_frame_equal(original, fresh)
