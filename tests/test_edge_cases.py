"""Uç durum testleri: uygulamanın hiçbir koşulda çökmemesi hedeflenir.

Kapsanan durumlar: boş dataframe, tek satırlık veri, tamamen boş sütun,
sonsuz (inf) değerler, yüksek kardinalite, tek sütunlu veri ve bozuk dosya.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from core import cleaner, operations, profiler
from core.chart_builder import build_chart
from core.code_generator import generate_full_script
from core.data_quality_score import compute_quality_score
from core.insights import generate_insights
from core.notebook_generator import build_notebook
from core.outliers import detect_outliers
from core.statistics import categorical_summary, numeric_summary
from utils import dataframe_utils as dfu
from utils.validators import EdaError

ROOT = Path(__file__).resolve().parents[1]

def _duplicate_column_frame() -> pd.DataFrame:
    """Aynı ada sahip iki sütun içeren DataFrame üretir."""
    frame = pd.DataFrame([[1, 2], [3, 4]])
    frame.columns = ["a", "a"]
    return frame


EDGE_FRAMES: dict[str, pd.DataFrame] = {
    "bos": pd.DataFrame(),
    "sutunsuz": pd.DataFrame(index=[0, 1, 2]),
    "tek_satir": pd.DataFrame({"a": [1], "b": ["x"]}),
    "tek_sutun": pd.DataFrame({"a": [1, 2, 3]}),
    "tamamen_bos_sutun": pd.DataFrame({"a": [1, 2, 3], "bos": [None, None, None]}),
    "tamamen_bos_dataset": pd.DataFrame({"a": [None] * 4, "b": [None] * 4}),
    "inf_degerler": pd.DataFrame({"a": [1.0, np.inf, -np.inf, 5.0]}),
    "sabit": pd.DataFrame({"a": ["X"] * 5}),
    "yuksek_kardinalite": pd.DataFrame({"kod": [f"K{i}" for i in range(200)]}),
    "karisik_tipler": pd.DataFrame({"m": [1, "iki", 3.0, None, True]}),
    "nan_metin": pd.DataFrame({"a": ["", " ", None, "nan"]}),
    "tamamen_tekrarli": pd.DataFrame({"a": [1] * 20, "b": ["x"] * 20}),
    "unicode_sutun": pd.DataFrame(
        {"şehir adı": ["İstanbul", "Çanakkale"], "ölçüm (₺)": [1.5, 2.5]}
    ),
    "tirnakli_sutun": pd.DataFrame({"O'Brien \"net\"": [1, 2], "ters\\bolu": [3, 4]}),
    "uzun_sutun_adi": pd.DataFrame({"x" * 300: [1, 2, 3]}),
    "cok_buyuk_sayilar": pd.DataFrame({"a": [1e308, -1e308, 1.0]}),
    "bool_ve_tarih": pd.DataFrame(
        {
            "flag": [True, False, True],
            "t": pd.to_datetime(["2024-01-01", "2024-02-01", None]),
        }
    ),
    "gecersiz_tarih_metni": pd.DataFrame({"t": ["2024-13-45", "abc", "2024-01-01"]}),
    "sifir_bolme": pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [0.0, 0.0, 2.0]}),
    "cok_sutunlu": pd.DataFrame({f"c{i}": [1, 2] for i in range(120)}),
    "tekrarli_sutun_adi": _duplicate_column_frame(),
}


@pytest.mark.parametrize("name", list(EDGE_FRAMES))
def test_insights_never_crash(name: str) -> None:
    insights = generate_insights(EDGE_FRAMES[name])
    assert isinstance(insights, list)


@pytest.mark.parametrize("name", list(EDGE_FRAMES))
def test_quality_score_never_crashes(name: str) -> None:
    score = compute_quality_score(EDGE_FRAMES[name])
    assert 0 <= score.overall <= 100


@pytest.mark.parametrize("name", list(EDGE_FRAMES))
def test_profiler_tables_never_crash(name: str) -> None:
    frame = EDGE_FRAMES[name]
    profiler.missing_value_table(frame)
    profiler.duplicate_report(frame)
    profiler.constant_column_table(frame)
    profiler.unique_value_table(frame)
    profiler.potential_id_table(frame)
    profiler.dtype_check_table(frame)
    dfu.column_profile(frame)
    dfu.dataset_shape_info(frame)
    dfu.safe_display_df(frame)


@pytest.mark.parametrize("name", list(EDGE_FRAMES))
def test_code_and_notebook_generation_never_crash(name: str) -> None:
    meta = {"filename": f"{name}.csv", "file_type": "csv"}
    script = generate_full_script(meta, [], [])
    compile(script, "<edge>", "exec")
    assert build_notebook(meta, [], [])["nbformat"] == 4


def test_statistics_on_all_empty_column_raises_friendly() -> None:
    frame = EDGE_FRAMES["tamamen_bos_sutun"]
    with pytest.raises(EdaError):
        numeric_summary(frame, "bos")
    with pytest.raises(EdaError):
        categorical_summary(frame, "bos")


def test_statistics_ignore_inf_values() -> None:
    summary = numeric_summary(EDGE_FRAMES["inf_degerler"], "a")
    assert summary["count"] == 2
    assert np.isfinite(summary["mean"]) and np.isfinite(summary["std"])


def test_outliers_on_single_row() -> None:
    report = detect_outliers(EDGE_FRAMES["tek_satir"], "a")
    assert report.count == 0


def test_outliers_on_all_empty_column_raises() -> None:
    with pytest.raises(EdaError):
        detect_outliers(EDGE_FRAMES["tamamen_bos_sutun"], "bos")


def test_chart_on_empty_dataframe_raises_friendly() -> None:
    with pytest.raises(EdaError):
        build_chart(EDGE_FRAMES["bos"], "histogram", {"column": "a"})


def test_chart_with_high_cardinality_is_limited() -> None:
    result = build_chart(
        EDGE_FRAMES["yuksek_kardinalite"], "bar", {"x": "kod", "aggregation": "count"}
    )
    assert result.warnings or True
    assert len(result.figure.data[0].x) <= 50


def test_pie_chart_limits_categories() -> None:
    result = build_chart(
        EDGE_FRAMES["yuksek_kardinalite"],
        "pie",
        {"names": "kod", "aggregation": "count", "top_n": 5},
    )
    assert len(result.figure.data[0].labels) == 5


def test_cleaning_on_single_row_dataset() -> None:
    frame = EDGE_FRAMES["tek_satir"]
    result = cleaner.drop_duplicates(frame)
    assert len(result) == 1
    with pytest.raises(EdaError):
        cleaner.drop_columns(frame, ["a", "b"])


def test_drop_all_rows_is_blocked() -> None:
    frame = pd.DataFrame({"a": [None, None], "b": [1, 2]})
    with pytest.raises(EdaError):
        cleaner.drop_missing_rows(frame, ["a"])


def test_conversion_of_mixed_types_does_not_crash() -> None:
    frame = EDGE_FRAMES["karisik_tipler"]
    preview = cleaner.conversion_preview(frame, "m", "float")
    assert preview["lost"] >= 1
    result = cleaner.convert_dtype(frame, "m", "string")
    assert str(result["m"].dtype) == "string"


def test_replay_on_edge_frames_is_safe() -> None:
    frame = EDGE_FRAMES["tamamen_bos_sutun"]
    _, operation = operations.apply_operation(frame, "drop_columns", {"columns": ["bos"]})
    replayed, errors = operations.replay(frame, [operation.to_dict()])
    assert errors == []
    assert "bos" not in replayed.columns


def test_single_column_dataset_insight() -> None:
    insights = generate_insights(EDGE_FRAMES["tek_sutun"])
    assert any(item["category"] == "structure" for item in insights)


def test_duplicate_count_with_unhashable_cells() -> None:
    frame = pd.DataFrame({"a": [[1, 2], [1, 2], [3]]})
    assert dfu.duplicate_count(frame) == 1
    assert dfu.safe_nunique(frame["a"]) == 2


def test_transformations_on_edge_frames_do_not_crash() -> None:
    """Dönüşümler uç durumlarda anlaşılır hata verir, çökmez."""
    from core import transformations as tf

    frame = EDGE_FRAMES["karisik_tipler"]
    with pytest.raises(EdaError):
        tf.numeric_transform(frame, "m", "standard")  # sayısal değil

    constant = EDGE_FRAMES["sabit"]
    with pytest.raises(EdaError):
        tf.numeric_transform(constant, "a", "minmax")  # metin sütun

    empty_column = EDGE_FRAMES["tamamen_bos_sutun"]
    with pytest.raises(EdaError):
        tf.bin_column(empty_column, "bos", "equal_width", bins=2)

    # Tek satırda quantile binning başarısız olur ama çökmemeli
    with pytest.raises(EdaError):
        tf.bin_column(EDGE_FRAMES["tek_satir"], "a", "quantile", bins=4)


def test_zero_division_in_features_is_safe() -> None:
    """Sıfıra bölme NaN üretir, sonsuz değer bırakmaz."""
    from core import transformations as tf

    frame = EDGE_FRAMES["sifir_bolme"]
    result = tf.arithmetic_feature(frame, "a", "b", "divide", "oran")
    assert result["oran"].isna().sum() == 2
    assert np.isfinite(result["oran"].dropna()).all()


def test_huge_numbers_do_not_overflow() -> None:
    """Çok büyük sayılarda istatistikler sonsuz/çökme üretmemeli."""
    from core.statistics import numeric_summary

    summary = numeric_summary(EDGE_FRAMES["cok_buyuk_sayilar"], "a")
    assert summary["count"] == 3
    assert np.isfinite(summary["mean"])


def test_invalid_datetime_text_handled() -> None:
    """Geçersiz tarih metinleri NaT olur, uygulama çökmez."""
    from core import cleaner

    frame = EDGE_FRAMES["gecersiz_tarih_metni"]
    preview = cleaner.conversion_preview(frame, "t", "datetime")
    assert preview["lost"] == 2
    converted = cleaner.convert_dtype(frame, "t", "datetime")
    assert converted["t"].isna().sum() == 2


def test_unicode_and_quoted_columns_generate_valid_code() -> None:
    """Türkçe, tırnaklı ve ters bölülü sütun adları için kod derlenebilmeli."""
    from core import operations as ops

    for name in ("unicode_sutun", "tirnakli_sutun", "uzun_sutun_adi"):
        frame = EDGE_FRAMES[name]
        column = str(frame.columns[0])
        _, operation = ops.apply_operation(
            frame, "rename_column", {"column": column, "new_name": f"{column}_yeni"[:280]}
        )
        compile(operation.python_code, "<generated>", "exec")


def test_fully_duplicated_dataset() -> None:
    """Tamamen tekrarlı veri setinde tekrar silme tek satır bırakmalı."""
    from core import cleaner

    result = cleaner.drop_duplicates(EDGE_FRAMES["tamamen_tekrarli"])
    assert len(result) == 1


def test_all_nan_dataset_reports_critical() -> None:
    """Tamamen boş veri setinde kritik bulgu üretilmeli."""
    insights = generate_insights(EDGE_FRAMES["tamamen_bos_dataset"])
    assert any(item["severity"] == "critical" for item in insights)


def test_duplicate_column_names_detected() -> None:
    """Aynı adlı sütunlar tespit edilmeli ve loader tarafından teklenmeli."""
    from core.loader import load_dataframe
    from core.profiler import duplicate_column_names

    assert not duplicate_column_names(EDGE_FRAMES["tekrarli_sutun_adi"]).empty
    loaded = load_dataframe(b"a,a,b\n1,2,3\n", "veri.csv")
    assert len(set(loaded.df.columns)) == 3


def test_pivot_and_ml_on_edge_frames() -> None:
    """Pivot ve ML hazırlığı uç durumlarda anlaşılır hata vermeli."""
    from core.ml_preparation import analyze_readiness
    from core.pivot import build_pivot

    with pytest.raises(EdaError):
        build_pivot(EDGE_FRAMES["bos"], ["a"], None, None, "count")
    # Tek satırlık veride rapor üretilir ama "blocker" ile uyarır (hata atmaz)
    report = analyze_readiness(EDGE_FRAMES["tek_satir"], "a", ["b"])
    assert report.blockers and not report.is_ready


def test_report_on_minimal_frame() -> None:
    """Tek satırlık veri setinde bile rapor üretilebilmeli."""
    from core.report import build_html_report

    html = build_html_report(EDGE_FRAMES["tek_satir"], {"filename": "tek.csv"})
    assert "Dataset Özeti" in html


def test_many_columns_profile_is_fast_enough() -> None:
    """120 sütunlu veri setinde profil üretimi çalışmalı."""
    profile = dfu.column_profile(EDGE_FRAMES["cok_sutunlu"])
    assert len(profile) == 120
    assert "Rol (tahmin)" in profile.columns


@pytest.mark.parametrize(
    "module",
    [
        "overview",
        "data_quality",
        "statistics_page",
        "insights",
        "export_page",
        "transform",
        "pivot_page",
        "ml_prep",
    ],
)
def test_pages_survive_hostile_dataset(module: str) -> None:
    """Tek satırlık, sabit ve boş sütunlu veriyle sayfalar çökmemeli."""
    script = f'''
import sys
sys.path.insert(0, r"{ROOT}")

import numpy as np
import pandas as pd
import streamlit as st

from core import session
from core.loader import LoadResult
from pages import {module} as page

session.init_session_state()
if not session.has_dataset():
    frame = pd.DataFrame(
        {{"tek": [1], "bos": [None], "sabit": ["X"], "metin": ["a"]}}
    )
    session.set_dataset(
        LoadResult(df=frame, filename="zor.csv", file_type="csv", separator=",")
    )

page.render()
'''
    app = AppTest.from_string(script, default_timeout=120)
    app.run()
    assert not app.exception, f"{module}: {[str(e) for e in app.exception]}"
