"""Streamlit arayüzü duman (smoke) testleri.

Her sayfanın gerçek bir veri setiyle hatasız render edilmesini doğrular.
``st.testing.v1.AppTest`` kullanıldığı için tarayıcı gerekmez.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CSV = ROOT / "sample_data" / "ornek_musteriler.csv"
SAMPLE = SAMPLE_CSV

PAGE_MODULES = [
    "upload",
    "overview",
    "data_quality",
    "cleaning",
    "transform",
    "statistics_page",
    "visualization",
    "relationships",
    "outliers_page",
    "insights",
    "pivot_page",
    "ml_prep",
    "history",
    "code_viewer",
    "export_page",
]


def _page_script(module: str, with_data: bool = True, with_history: bool = True) -> str:
    """Tek bir sayfayı izole çalıştıran mini Streamlit script'i üretir."""
    return f'''
import sys
sys.path.insert(0, r"{ROOT}")

import streamlit as st

from core import session
from core.loader import load_dataframe
from pages import {module} as page

session.init_session_state()

if {with_data} and not session.has_dataset():
    with open(r"{SAMPLE_CSV}", "rb") as handle:
        session.set_dataset(load_dataframe(handle.read(), "ornek_musteriler.csv"))
    if {with_history}:
        session.apply_and_record("fill_missing", {{"column": "gelir", "method": "median"}})
        session.apply_and_record("drop_duplicates", {{}})
        session.add_chart(
            {{
                "id": "chart-1",
                "type": "histogram",
                "label": "Histogram",
                "title": "gelir dağılımı",
                "description": "Histogram - gelir",
                "params": {{"column": "gelir", "bins": 20}},
                "python_code": "fig = px.histogram(df, x='gelir', nbins=20)\\nfig.show()",
                "source_note": "working_df",
            }}
        )

page.render()
'''


@pytest.mark.parametrize("module", PAGE_MODULES)
def test_page_renders_with_dataset(module: str) -> None:
    """Veri yüklüyken her sayfa hatasız çizilmeli."""
    app = AppTest.from_string(_page_script(module), default_timeout=120)
    app.run()
    assert not app.exception, f"{module}: {[str(e) for e in app.exception]}"
    assert not app.error, f"{module}: {[e.value for e in app.error]}"


@pytest.mark.parametrize("module", PAGE_MODULES)
def test_page_renders_without_dataset(module: str) -> None:
    """Veri yokken sayfalar çökmek yerine boş durum göstermeli."""
    app = AppTest.from_string(_page_script(module, with_data=False), default_timeout=120)
    app.run()
    assert not app.exception, f"{module}: {[str(e) for e in app.exception]}"
    assert not app.error, f"{module}: {[e.value for e in app.error]}"


def test_app_boots() -> None:
    """app.py navigasyonla birlikte açılabilmeli."""
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    app.run()
    assert not app.exception, [str(e) for e in app.exception]
    assert any("No-Code EDA Studio" in str(item.value) for item in app.markdown)


def test_data_quality_analyses_run() -> None:
    """Veri Kalitesi sayfasındaki analizler otomatik çalışmalı ve eşikler değişebilmeli."""
    app = AppTest.from_string(_page_script("data_quality"), default_timeout=180)
    app.run()
    assert not app.exception, [str(e) for e in app.exception]
    assert not app.error, [e.value for e in app.error]

    # Eşik kaydırıcıları analizleri yeniden çalıştırmalı
    app.slider(key="dq_near_threshold").set_value(0.85).run()
    app.slider(key="dq_id_threshold").set_value(0.80).run()
    assert not app.exception, [str(e) for e in app.exception]

    # Kalite skoru ve tüm sekmeler çizilmiş olmalı
    rendered = " ".join(str(item.value) for item in app.markdown)
    assert "Sabit sütunlar" in rendered or "Boş Değerler" in rendered


def test_cleaning_fill_missing_flow() -> None:
    """Veri Temizleme sayfasından boş değer doldurma işlemi uygulanabilmeli."""
    script = f'''
import sys
sys.path.insert(0, r"{ROOT}")

import streamlit as st

from core import session
from core.loader import load_dataframe
from pages import cleaning

session.init_session_state()
if not session.has_dataset():
    with open(r"{SAMPLE_CSV}", "rb") as handle:
        session.set_dataset(load_dataframe(handle.read(), "ornek_musteriler.csv"))

st.session_state["missing_before"] = int(session.get_working_df()["gelir"].isna().sum())
cleaning.render()
'''
    app = AppTest.from_string(script, default_timeout=180)
    app.run()
    assert not app.exception
    assert app.session_state["missing_before"] > 0

    app.radio(key="clean_missing_mode").set_value("Boş değerleri doldur").run()
    app.selectbox(key="clean_fill_col").set_value("gelir").run()
    app.selectbox(key="clean_fill_method").set_value("median").run()
    app.button(key="btn_fill_missing").click().run()

    assert not app.exception, [str(e) for e in app.exception]
    assert int(app.session_state["working_df"]["gelir"].isna().sum()) == 0
    assert len(app.session_state["operation_history"]) == 1
    assert app.session_state["original_df"]["gelir"].isna().sum() > 0  # orijinal korunur


def test_undo_and_reset_flow() -> None:
    """İşlem Geçmişi sayfasında geri alma ve sıfırlama çalışmalı."""
    app = AppTest.from_string(_page_script("history"), default_timeout=180)
    app.run()
    assert len(app.session_state["operation_history"]) == 2

    app.button[0].click().run()  # Son İşlemi Geri Al
    assert not app.exception, [str(e) for e in app.exception]
    assert len(app.session_state["operation_history"]) == 1

    app.button[1].click().run()  # Orijinale Sıfırla
    assert not app.exception
    assert app.session_state["operation_history"] == []
    original = app.session_state["original_df"]
    working = app.session_state["working_df"]
    assert original.shape == working.shape


def test_transform_scaling_flow() -> None:
    """Dönüşümler sayfasından ölçekleme uygulanabilmeli ve yeni sütun oluşmalı."""
    app = AppTest.from_string(
        _page_script("transform", with_history=False), default_timeout=180
    )
    app.run()
    assert not app.exception, [str(e) for e in app.exception]

    app.selectbox(key="tf_num_col").set_value("gelir").run()
    app.selectbox(key="tf_num_method").set_value("standard").run()
    # Önerilen sütun adı yöntem değişince güncellenir (anahtar sütun+yöntem içerir)
    name_widget = app.text_input(key="tf_num_name_gelir_z")
    assert name_widget.value == "gelir_z"

    app.button(key="tf_num_apply").click().run()
    assert not app.exception, [str(e) for e in app.exception]
    working = app.session_state["working_df"]
    assert "gelir_z" in working.columns
    assert app.session_state["operation_history"][-1]["action"] == "numeric_transform"
    # Orijinal veri korunmalı
    assert "gelir_z" not in app.session_state["original_df"].columns


def test_transform_datetime_features_flow() -> None:
    """Tarih sütunu dönüştürüldükten sonra tarih özellikleri üretilebilmeli."""
    script = f'''
import sys
sys.path.insert(0, r"{ROOT}")

import streamlit as st

from core import session
from core.loader import load_dataframe
from pages import transform

session.init_session_state()
if not session.has_dataset():
    with open(r"{SAMPLE}", "rb") as handle:
        session.set_dataset(load_dataframe(handle.read(), "ornek_musteriler.csv"))
    session.apply_and_record(
        "convert_dtype", {{"column": "uyelik_tarihi", "target": "datetime"}}
    )

transform.render()
'''
    app = AppTest.from_string(script, default_timeout=180)
    app.run()
    assert not app.exception, [str(e) for e in app.exception]

    app.multiselect(key="tf_dt_features").set_value(["year", "month"]).run()
    app.button(key="tf_dt_apply").click().run()
    assert not app.exception, [str(e) for e in app.exception]
    working = app.session_state["working_df"]
    assert "uyelik_tarihi_year" in working.columns
    assert "uyelik_tarihi_month" in working.columns


def test_pivot_page_builds_table() -> None:
    """Pivot sayfası gruplama ve pivot sonucu üretebilmeli."""
    app = AppTest.from_string(_page_script("pivot_page"), default_timeout=180)
    app.run()
    assert not app.exception, [str(e) for e in app.exception]

    app.multiselect(key="gb_cols").set_value(["sehir"]).run()
    assert not app.exception, [str(e) for e in app.exception]
    assert app.dataframe  # sonuç tablosu çizildi

    app.multiselect(key="pv_rows").set_value(["sehir"]).run()
    assert not app.exception, [str(e) for e in app.exception]


def test_ml_prep_generates_code() -> None:
    """ML Hazırlığı sayfası kod üretip export'a eklemeli."""
    app = AppTest.from_string(_page_script("ml_prep"), default_timeout=180)
    app.run()
    assert not app.exception, [str(e) for e in app.exception]

    app.selectbox(key="ml_target").set_value("segment").run()
    assert not app.exception, [str(e) for e in app.exception]
    code_blocks = " ".join(str(item.value) for item in app.code)
    assert "train_test_split" in code_blocks

    app.button(key="ml_save").click().run()
    assert "train_test_split" in app.session_state["ml_prep_code"]


def test_chart_builder_creates_chart() -> None:
    """Grafik Oluşturucu gerçek bir grafik üretip koda eklemeli."""
    app = AppTest.from_string(
        _page_script("visualization", with_history=False), default_timeout=180
    )
    app.run()
    assert not app.exception

    app.selectbox(key="chart_type").set_value("histogram").run()
    app.selectbox(key="hist_col").set_value("gelir").run()
    app.button(key="build_chart").click().run()

    assert not app.exception, [str(e) for e in app.exception]
    charts = app.session_state["generated_charts"]
    assert len(charts) == 1
    assert charts[0]["type"] == "histogram"
    assert "px.histogram" in charts[0]["python_code"]
