"""core.notebook_generator ve core.exporter testleri."""

from __future__ import annotations

import io
import json

import nbformat
import pandas as pd
import pytest

from core import operations
from core.chart_builder import build_chart
from core.exporter import (
    dataframe_to_csv_bytes,
    dataframe_to_excel_bytes,
    export_filename,
    figure_to_html_bytes,
    notebook_bytes,
    script_to_bytes,
)
from core.notebook_generator import (
    build_notebook,
    notebook_filename,
    notebook_to_bytes,
    script_filename,
)


@pytest.fixture
def session_fixture(messy_df: pd.DataFrame, numeric_df: pd.DataFrame) -> dict:
    """Operasyon geçmişi ve grafik içeren örnek bir oturum."""
    step1, op1 = operations.apply_operation(
        messy_df, "fill_missing", {"column": "gelir", "method": "median"}
    )
    _, op2 = operations.apply_operation(step1, "drop_duplicates", {})
    chart = build_chart(numeric_df, "scatter", {"x": "x", "y": "y"}).spec
    return {
        "meta": {"filename": "musteriler.csv", "file_type": "csv", "separator": ";"},
        "history": operations.history_to_dicts([op1, op2]),
        "charts": [chart],
        "insights": [
            {
                "severity": "warning",
                "category": "missing",
                "column": "gelir",
                "title": "Boş değer",
                "message": "Doldurma yapıldı.",
                "metric": 14.2,
            }
        ],
    }


def test_notebook_is_valid_v4(session_fixture: dict) -> None:
    notebook = build_notebook(
        session_fixture["meta"], session_fixture["history"], session_fixture["charts"]
    )
    nbformat.validate(notebook)
    assert notebook["nbformat"] == 4


def test_notebook_bytes_parse_as_json(session_fixture: dict) -> None:
    payload = notebook_bytes(
        session_fixture["meta"],
        session_fixture["history"],
        session_fixture["charts"],
        session_fixture["insights"],
    )
    parsed = json.loads(payload.decode("utf-8"))
    assert parsed["nbformat"] == 4
    assert parsed["cells"]

    # nbformat ile tekrar okunabilmeli (Jupyter/Colab uyumluluğu)
    reloaded = nbformat.reads(payload.decode("utf-8"), as_version=4)
    nbformat.validate(reloaded)


def test_notebook_has_expected_sections(session_fixture: dict) -> None:
    notebook = build_notebook(
        session_fixture["meta"],
        session_fixture["history"],
        session_fixture["charts"],
        session_fixture["insights"],
    )
    markdown = "\n".join(
        cell["source"] for cell in notebook.cells if cell["cell_type"] == "markdown"
    )
    for heading in (
        "# Exploratory Data Analysis",
        "## Imports",
        "## Data Loading",
        "## Dataset Overview",
        "## Data Cleaning",
        "## Statistical Analysis",
        "## Relationships",
        "## Outlier Analysis",
        "## Visualizations",
        "## ML Preparation",
        "## Insights",
    ):
        assert heading in markdown


def test_notebook_code_cells_compile(session_fixture: dict) -> None:
    notebook = build_notebook(
        session_fixture["meta"], session_fixture["history"], session_fixture["charts"]
    )
    for cell in notebook.cells:
        if cell["cell_type"] == "code":
            source = cell["source"].replace("display(", "print(")
            compile(source, "<cell>", "exec")


def test_notebook_includes_insight_notes(session_fixture: dict) -> None:
    notebook = build_notebook(
        session_fixture["meta"],
        session_fixture["history"],
        session_fixture["charts"],
        session_fixture["insights"],
    )
    notes = notebook.cells[-1]["source"]
    assert "Boş değer" in notes
    assert "Korelasyon nedensellik göstermez" in notes


def test_notebook_without_history_still_valid() -> None:
    notebook = build_notebook({"filename": "veri.csv", "file_type": "csv"}, [], [])
    nbformat.validate(notebook)
    markdown = "\n".join(
        cell["source"] for cell in notebook.cells if cell["cell_type"] == "markdown"
    )
    assert "veri temizleme işlemi uygulanmadı" in markdown


def test_notebook_does_not_embed_user_data(session_fixture: dict, messy_df: pd.DataFrame) -> None:
    payload = notebook_bytes(
        session_fixture["meta"], session_fixture["history"], session_fixture["charts"]
    ).decode("utf-8")
    # Satır değerleri notebook içine gömülmemeli
    for value in ("Zeynep", "Ayse", "Bursa"):
        assert value not in payload
    assert len(messy_df) > 0


def test_filenames() -> None:
    assert notebook_filename("musteriler.csv") == "musteriler_eda.ipynb"
    assert script_filename("musteriler.csv") == "musteriler_eda.py"
    assert notebook_filename(None) == "dataset_eda.ipynb"
    assert export_filename("satis verisi.xlsx", "temiz", "csv") == "satis_verisi_temiz.csv"
    # Yol ayraclari ve guvensiz karakterler temizlenir
    assert script_filename(r"C:\veri\musteri raporu.csv") == "musteri_raporu_eda.py"


def test_notebook_metadata_has_kernelspec(session_fixture: dict) -> None:
    notebook = build_notebook(session_fixture["meta"], session_fixture["history"], [])
    assert notebook["metadata"]["kernelspec"]["name"] == "python3"
    assert notebook["metadata"]["eda_studio"]["operations"] == 2


def test_notebook_to_bytes_validates() -> None:
    notebook = build_notebook({"filename": "a.csv", "file_type": "csv"}, [], [])
    assert notebook_to_bytes(notebook).startswith(b"{")


# ---------------------------------------------------------------------------
# Exporter
# ---------------------------------------------------------------------------


def test_csv_export_round_trip(messy_df: pd.DataFrame) -> None:
    payload = dataframe_to_csv_bytes(messy_df, ";")
    frame = pd.read_csv(io.BytesIO(payload), sep=";", encoding="utf-8-sig")
    assert list(frame.columns) == list(messy_df.columns)
    assert len(frame) == len(messy_df)


def test_excel_export_round_trip(messy_df: pd.DataFrame) -> None:
    payload = dataframe_to_excel_bytes(messy_df, "Veri")
    frame = pd.read_excel(io.BytesIO(payload), sheet_name="Veri")
    assert len(frame) == len(messy_df)


def test_excel_export_with_extra_sheets(messy_df: pd.DataFrame) -> None:
    extra = pd.DataFrame({"adim": [1, 2], "aciklama": ["a", "b"]})
    payload = dataframe_to_excel_bytes(messy_df, "Veri", {"Gecmis": extra})
    book = pd.ExcelFile(io.BytesIO(payload))
    assert set(book.sheet_names) == {"Veri", "Gecmis"}


def test_excel_export_handles_timezone_and_category() -> None:
    frame = pd.DataFrame(
        {
            "tarih": pd.to_datetime(["2024-01-01", "2024-01-02"]).tz_localize("Europe/Istanbul"),
            "kategori": pd.Categorical(["a", "b"]),
        }
    )
    payload = dataframe_to_excel_bytes(frame)
    assert len(pd.read_excel(io.BytesIO(payload))) == 2


def test_script_export_compiles(session_fixture: dict) -> None:
    payload = script_to_bytes(
        session_fixture["meta"], session_fixture["history"], session_fixture["charts"]
    )
    compile(payload.decode("utf-8"), "<script>", "exec")


def test_figure_html_export(numeric_df: pd.DataFrame) -> None:
    result = build_chart(numeric_df, "histogram", {"column": "x", "bins": 20})
    html = figure_to_html_bytes(result.figure).decode("utf-8")
    assert "<html" in html.lower()
    assert "plotly" in html.lower()
