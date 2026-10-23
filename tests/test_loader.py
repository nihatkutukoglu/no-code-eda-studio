"""core.loader testleri: encoding, ayraç ve hata yönetimi."""

from __future__ import annotations

import io

import pandas as pd
import pytest

from core.loader import detect_extension, excel_sheet_names, load_dataframe, loading_code
from utils.validators import LoaderError


def test_detects_comma_separator() -> None:
    data = b"a,b,c\n1,2,3\n4,5,6\n"
    result = load_dataframe(data, "veri.csv")
    assert result.separator == ","
    assert result.df.shape == (2, 3)


def test_detects_semicolon_separator() -> None:
    data = "ad;gelir\nAli;1000\nAyse;2000\n".encode("utf-8")
    result = load_dataframe(data, "veri.csv")
    assert result.separator == ";"
    assert list(result.df.columns) == ["ad", "gelir"]


def test_detects_tab_separator() -> None:
    data = b"a\tb\n1\t2\n3\t4\n"
    result = load_dataframe(data, "veri.txt")
    assert result.separator == "\t"
    assert result.df.shape == (2, 2)


def test_reads_windows_turkish_encoding() -> None:
    data = "sehir;deger\nİstanbul;5\nŞırnak;7\n".encode("cp1254")
    result = load_dataframe(data, "veri.csv")
    assert result.encoding in ("cp1254", "iso-8859-9", "latin-1")
    assert "sehir" in result.df.columns


def test_normalizes_unnamed_and_duplicate_columns() -> None:
    data = b"a, ,a\n1,2,3\n"
    result = load_dataframe(data, "veri.csv")
    assert len(set(result.df.columns)) == 3
    assert any(warning for warning in result.warnings)


def test_drops_fully_empty_rows() -> None:
    data = b"a,b\n1,2\n,\n3,4\n"
    result = load_dataframe(data, "veri.csv")
    assert len(result.df) == 2


def test_empty_file_raises_loader_error() -> None:
    with pytest.raises(LoaderError):
        load_dataframe(b"", "veri.csv")


def test_only_header_raises_loader_error() -> None:
    with pytest.raises(LoaderError):
        load_dataframe(b"a,b,c\n", "veri.csv")


def test_unsupported_extension_raises() -> None:
    with pytest.raises(LoaderError):
        load_dataframe(b"veri", "veri.parquet")
    with pytest.raises(LoaderError):
        detect_extension("uzantisiz")


def test_corrupt_excel_raises_friendly_error() -> None:
    with pytest.raises(LoaderError):
        load_dataframe(b"bu bir excel degil", "veri.xlsx")


def test_excel_round_trip() -> None:
    frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    buffer = io.BytesIO()
    frame.to_excel(buffer, index=False, sheet_name="Sayfa1")
    payload = buffer.getvalue()

    assert excel_sheet_names(payload, "veri.xlsx") == ["Sayfa1"]
    result = load_dataframe(payload, "veri.xlsx", sheet_name="Sayfa1")
    assert result.file_type == "excel"
    assert result.df.shape == (2, 2)


def test_single_row_dataset_warns() -> None:
    result = load_dataframe(b"a,b\n1,2\n", "veri.csv")
    assert any("tek satır" in warning.lower() for warning in result.warnings)


def test_loading_code_for_csv(dataset_meta: dict) -> None:
    code = loading_code(dataset_meta)
    assert code == "df = pd.read_csv('musteriler.csv', sep=';')"


def test_loading_code_for_excel() -> None:
    code = loading_code(
        {"filename": "satis.xlsx", "file_type": "excel", "sheet_name": "Ocak"}
    )
    assert code == "df = pd.read_excel('satis.xlsx', sheet_name='Ocak')"


def test_loading_code_escapes_quotes() -> None:
    code = loading_code({"filename": "o'ne ad.csv", "file_type": "csv"})
    assert "o'ne ad.csv" in code
    compile(code, "<test>", "exec")
