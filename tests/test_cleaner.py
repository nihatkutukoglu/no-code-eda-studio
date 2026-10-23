"""core.cleaner testleri: dönüşümlerin doğruluğu ve hata yönetimi."""

from __future__ import annotations

import pandas as pd
import pytest

from core import cleaner
from utils.validators import CleaningError


def test_fill_missing_with_median(messy_df: pd.DataFrame) -> None:
    result = cleaner.fill_missing(messy_df, "gelir", "median")
    assert result["gelir"].isna().sum() == 0
    assert messy_df["gelir"].isna().sum() == 1  # orijinal değişmedi
    assert result.loc[2, "gelir"] == pytest.approx(messy_df["gelir"].median())


def test_fill_missing_with_mean_on_text_raises(messy_df: pd.DataFrame) -> None:
    with pytest.raises(CleaningError) as error:
        cleaner.fill_missing(messy_df, "ad", "mean")
    assert "sayısal" in str(error.value)


def test_fill_missing_with_mode(messy_df: pd.DataFrame) -> None:
    result = cleaner.fill_missing(messy_df, "ad", "mode")
    assert result["ad"].isna().sum() == 0


def test_fill_missing_with_custom_value_coerces_type(messy_df: pd.DataFrame) -> None:
    result = cleaner.fill_missing(messy_df, "gelir", "value", "1500")
    assert result.loc[2, "gelir"] == 1500.0


def test_fill_missing_custom_value_invalid_raises(messy_df: pd.DataFrame) -> None:
    with pytest.raises(CleaningError):
        cleaner.fill_missing(messy_df, "gelir", "value", "abc")


def test_fill_missing_on_all_empty_column_raises(messy_df: pd.DataFrame) -> None:
    with pytest.raises(CleaningError):
        cleaner.fill_missing(messy_df, "bos", "mode")


def test_drop_missing_rows_subset(messy_df: pd.DataFrame) -> None:
    result = cleaner.drop_missing_rows(messy_df, ["gelir"])
    assert len(result) == len(messy_df) - 1
    assert result.index.tolist() == list(range(len(result)))


def test_drop_missing_rows_all_columns_would_empty_raises(messy_df: pd.DataFrame) -> None:
    # 'bos' sütunu tamamen boş olduğu için any kuralı tüm satırları silerdi.
    with pytest.raises(CleaningError):
        cleaner.drop_missing_rows(messy_df, None, "any")


def test_drop_duplicates(messy_df: pd.DataFrame) -> None:
    result = cleaner.drop_duplicates(messy_df)
    assert len(result) == len(messy_df) - 1


def test_drop_duplicates_subset(messy_df: pd.DataFrame) -> None:
    result = cleaner.drop_duplicates(messy_df, ["sehir"])
    assert len(result) == 3


def test_rename_column(messy_df: pd.DataFrame) -> None:
    result = cleaner.rename_column(messy_df, "ad", "isim")
    assert "isim" in result.columns and "ad" not in result.columns


def test_rename_to_existing_raises(messy_df: pd.DataFrame) -> None:
    with pytest.raises(CleaningError):
        cleaner.rename_column(messy_df, "ad", "sehir")


def test_rename_empty_raises(messy_df: pd.DataFrame) -> None:
    with pytest.raises(CleaningError):
        cleaner.rename_column(messy_df, "ad", "   ")


def test_drop_columns(messy_df: pd.DataFrame) -> None:
    result = cleaner.drop_columns(messy_df, ["bos", "sabit"])
    assert "bos" not in result.columns and "sabit" not in result.columns


def test_drop_all_columns_raises(messy_df: pd.DataFrame) -> None:
    with pytest.raises(CleaningError):
        cleaner.drop_columns(messy_df, list(messy_df.columns))


def test_drop_missing_column_raises(messy_df: pd.DataFrame) -> None:
    with pytest.raises(CleaningError):
        cleaner.drop_columns(messy_df, ["olmayan"])


def test_convert_to_datetime(messy_df: pd.DataFrame) -> None:
    result = cleaner.convert_dtype(messy_df, "tarih_metin", "datetime")
    assert pd.api.types.is_datetime64_any_dtype(result["tarih_metin"])


def test_convert_text_number_to_float(messy_df: pd.DataFrame) -> None:
    result = cleaner.convert_dtype(messy_df, "sayi_metin", "float")
    assert float(result["sayi_metin"].iloc[0]) == pytest.approx(1.5)


def test_convert_to_integer_rounds(messy_df: pd.DataFrame) -> None:
    result = cleaner.convert_dtype(messy_df, "sayi_metin", "integer")
    assert str(result["sayi_metin"].dtype) == "Int64"


def test_convert_impossible_raises_without_crash(messy_df: pd.DataFrame) -> None:
    with pytest.raises(CleaningError):
        cleaner.convert_dtype(messy_df, "ad", "datetime")


def test_convert_to_category_and_string(messy_df: pd.DataFrame) -> None:
    assert str(cleaner.convert_dtype(messy_df, "sehir", "category")["sehir"].dtype) == "category"
    assert str(cleaner.convert_dtype(messy_df, "sehir", "string")["sehir"].dtype) == "string"


def test_convert_to_boolean() -> None:
    frame = pd.DataFrame({"flag": ["Evet", "Hayır", "true", "0", None]})
    result = cleaner.convert_dtype(frame, "flag", "boolean")
    assert str(result["flag"].dtype) == "boolean"
    assert bool(result["flag"].iloc[0]) is True
    assert bool(result["flag"].iloc[1]) is False


def test_drop_outlier_rows() -> None:
    frame = pd.DataFrame({"gelir": [100, 110, 120, 130, 140, 150, 160, 5000]})
    result = cleaner.drop_outlier_rows(frame, "gelir", 1.5)
    assert len(result) == 7
    assert result["gelir"].max() == 160


def test_drop_outlier_rows_keeps_missing_values() -> None:
    frame = pd.DataFrame({"gelir": [100, 110, 120, 130, None, 5000]})
    result = cleaner.drop_outlier_rows(frame, "gelir", 1.5)
    assert result["gelir"].isna().sum() == 1
    assert 5000 not in result["gelir"].tolist()


def test_filter_rows() -> None:
    frame = pd.DataFrame({"a": [1, 5, 10]})
    result = cleaner.filter_rows(frame, [{"column": "a", "operator": "gte", "value": 5}])
    assert result["a"].tolist() == [5, 10]


def test_impact_helpers(messy_df: pd.DataFrame) -> None:
    assert cleaner.fill_impact(messy_df, "gelir") == 1
    assert cleaner.missing_row_impact(messy_df, ["gelir"]) == 1
    assert cleaner.duplicate_impact(messy_df) == 1
    assert cleaner.fill_impact(messy_df, "olmayan") == 0
    assert cleaner.outlier_impact(messy_df, "olmayan") == 0

    outlier_frame = pd.DataFrame({"v": [1, 2, 3, 4, 5, 6, 7, 500]})
    assert cleaner.outlier_impact(outlier_frame, "v") == 1


def test_conversion_preview_reports_losses(messy_df: pd.DataFrame) -> None:
    preview = cleaner.conversion_preview(messy_df, "sayi_metin", "float")
    assert preview["lost"] == 0 and preview["ok"] == 7

    partial = pd.DataFrame({"mixed": ["1", "2", "abc", None]})
    preview = cleaner.conversion_preview(partial, "mixed", "float")
    assert preview["lost"] == 1
    assert "abc" in preview["examples"]


def test_original_dataframe_never_mutated(messy_df: pd.DataFrame) -> None:
    snapshot = messy_df.copy(deep=True)
    cleaner.fill_missing(messy_df, "gelir", "median")
    cleaner.drop_duplicates(messy_df)
    cleaner.drop_columns(messy_df, ["bos"])
    cleaner.convert_dtype(messy_df, "tarih_metin", "datetime")
    pd.testing.assert_frame_equal(messy_df, snapshot)
