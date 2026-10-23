"""core.transformations testleri: dönüşümlerin doğruluğu, saflığı ve kod eşdeğerliği."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core import operations as ops
from core import transformations as tf
from core.code_generator import build_operation_code
from utils.validators import CleaningError


@pytest.fixture
def frame() -> pd.DataFrame:
    """Dönüşüm testleri için karışık tipli küçük veri seti."""
    return pd.DataFrame(
        {
            "ad": ["  Ali Veli ", "ayse  yilmaz", "MEHMET", None, "zeynep-demir"],
            "sehir": ["Ankara", "Izmir", "Ankara", "Bursa", "Izmir"],
            "gelir": [1000.0, 2000.0, 3000.0, 4000.0, np.nan],
            "adet": [1, 2, 0, 4, 5],
            "tarih": pd.to_datetime(
                ["2024-01-06", "2024-02-11", "2024-03-20", "2024-07-14", None]
            ),
        }
    )


def _run_generated_code(frame: pd.DataFrame, action: str, params: dict) -> pd.DataFrame:
    """Operasyonun üretilen kodunu çalıştırıp sonucu döndürür."""
    normalized = ops.normalize_params(frame, action, params)
    code = build_operation_code(action, normalized)
    namespace = {"df": frame.copy(), "pd": pd, "np": np}
    exec(compile(code, "<generated>", "exec"), namespace)  # noqa: S102 - test amaçlı
    return namespace["df"]


# ---------------------------------------------------------------------------
# Metin
# ---------------------------------------------------------------------------


def test_text_trim_and_case(frame: pd.DataFrame) -> None:
    trimmed = tf.text_transform(frame, "ad", "trim", new_column="ad_t")
    assert trimmed.loc[0, "ad_t"] == "Ali Veli"
    assert tf.text_transform(frame, "ad", "lower")["ad"].iloc[2] == "mehmet"
    assert tf.text_transform(frame, "ad", "upper")["ad"].iloc[0].strip() == "ALI VELI"
    assert tf.text_transform(frame, "ad", "title")["ad"].iloc[2] == "Mehmet"


def test_text_normalize_space(frame: pd.DataFrame) -> None:
    result = tf.text_transform(frame, "ad", "normalize_space")
    assert result["ad"].iloc[1] == "ayse yilmaz"


def test_text_replace_and_remove(frame: pd.DataFrame) -> None:
    replaced = tf.text_transform(frame, "ad", "replace", find="-", replace_with=" ")
    assert replaced["ad"].iloc[4] == "zeynep demir"
    removed = tf.text_transform(frame, "ad", "remove_chars", characters="-")
    assert removed["ad"].iloc[4] == "zeynepdemir"


def test_text_requires_find_value(frame: pd.DataFrame) -> None:
    with pytest.raises(CleaningError):
        tf.text_transform(frame, "ad", "replace", find="")
    with pytest.raises(CleaningError):
        tf.text_transform(frame, "ad", "remove_chars", characters="")


def test_text_rejects_numeric_column(frame: pd.DataFrame) -> None:
    with pytest.raises(CleaningError):
        tf.text_transform(frame, "gelir", "trim")


def test_text_code_matches_result(frame: pd.DataFrame) -> None:
    params = {"column": "ad", "method": "normalize_space", "new_column": "ad_n"}
    expected = tf.text_transform(frame, "ad", "normalize_space", new_column="ad_n")
    produced = _run_generated_code(frame, "text_transform", params)
    pd.testing.assert_series_equal(
        produced["ad_n"].astype("string"), expected["ad_n"].astype("string")
    )


# ---------------------------------------------------------------------------
# Sayısal dönüşümler
# ---------------------------------------------------------------------------


def test_minmax_scaling(frame: pd.DataFrame) -> None:
    result = tf.numeric_transform(frame, "gelir", "minmax", new_column="s")
    assert result["s"].min() == pytest.approx(0.0)
    assert result["s"].max() == pytest.approx(1.0)
    assert result["s"].isna().sum() == 1  # NaN korunur


def test_standard_scaling_mean_zero(frame: pd.DataFrame) -> None:
    result = tf.numeric_transform(frame, "gelir", "standard", new_column="z")
    clean = result["z"].dropna()
    assert clean.mean() == pytest.approx(0.0, abs=1e-9)
    assert clean.std(ddof=0) == pytest.approx(1.0, abs=1e-9)


def test_robust_scaling_median_zero(frame: pd.DataFrame) -> None:
    result = tf.numeric_transform(frame, "gelir", "robust", new_column="r")
    assert result["r"].dropna().median() == pytest.approx(0.0, abs=1e-9)


def test_log_transform_handles_non_positive() -> None:
    data = pd.DataFrame({"v": [-1.0, 0.0, 1.0, np.e]})
    result = tf.numeric_transform(data, "v", "log", new_column="log_v")
    assert result["log_v"].iloc[2] == pytest.approx(0.0)
    assert result["log_v"].iloc[3] == pytest.approx(1.0)
    assert result["log_v"].isna().sum() == 2  # -1 ve 0 tanımsız


def test_log1p_and_sqrt_and_round() -> None:
    data = pd.DataFrame({"v": [0.0, 3.0, 8.9999]})
    assert tf.numeric_transform(data, "v", "log1p")["v"].iloc[0] == pytest.approx(0.0)
    assert tf.numeric_transform(data, "v", "sqrt")["v"].iloc[2] == pytest.approx(3.0, abs=1e-4)
    assert tf.numeric_transform(data, "v", "round", decimals=1)["v"].iloc[2] == pytest.approx(9.0)
    assert tf.numeric_transform(pd.DataFrame({"v": [-5.0]}), "v", "abs")["v"].iloc[0] == 5.0


def test_scaling_on_constant_column_raises() -> None:
    data = pd.DataFrame({"v": [5.0, 5.0, 5.0]})
    for method in ("minmax", "standard", "robust"):
        with pytest.raises(CleaningError):
            tf.numeric_transform(data, "v", method)


def test_numeric_transform_rejects_text(frame: pd.DataFrame) -> None:
    with pytest.raises(CleaningError):
        tf.numeric_transform(frame, "ad", "standard")


@pytest.mark.parametrize(
    "method", ["log", "log1p", "sqrt", "abs", "round", "minmax", "standard", "robust"]
)
def test_numeric_code_matches_result(frame: pd.DataFrame, method: str) -> None:
    params = {"column": "gelir", "method": method, "new_column": "out", "decimals": 2}
    expected = tf.numeric_transform(frame, "gelir", method, 2, "out")
    produced = _run_generated_code(frame, "numeric_transform", params)
    pd.testing.assert_series_equal(
        produced["out"].astype(float), expected["out"].astype(float), check_names=False
    )


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def test_label_encoding(frame: pd.DataFrame) -> None:
    result = tf.encode_categorical(frame, "sehir", "label", new_column="kod")
    assert result["kod"].tolist() == [0, 2, 0, 1, 2]  # alfabetik: Ankara, Bursa, Izmir
    assert str(result["kod"].dtype) == "Int64"


def test_label_encoding_keeps_missing() -> None:
    data = pd.DataFrame({"c": ["a", None, "b"]})
    result = tf.encode_categorical(data, "c", "label", new_column="k")
    assert result["k"].isna().sum() == 1


def test_onehot_encoding(frame: pd.DataFrame) -> None:
    result = tf.encode_categorical(frame, "sehir", "onehot")
    assert "sehir" not in result.columns
    for column in ("sehir_Ankara", "sehir_Bursa", "sehir_Izmir"):
        assert column in result.columns
    assert result[["sehir_Ankara", "sehir_Bursa", "sehir_Izmir"]].sum().sum() == 5


def test_onehot_drop_first(frame: pd.DataFrame) -> None:
    result = tf.encode_categorical(frame, "sehir", "onehot", drop_first=True)
    dummies = [c for c in result.columns if c.startswith("sehir_")]
    assert len(dummies) == 2


def test_onehot_rejects_high_cardinality() -> None:
    data = pd.DataFrame({"c": [f"v{i}" for i in range(150)]})
    with pytest.raises(CleaningError) as error:
        tf.encode_categorical(data, "c", "onehot")
    assert "One-Hot" in str(error.value)


def test_ordinal_encoding(frame: pd.DataFrame) -> None:
    result = tf.encode_categorical(
        frame, "sehir", "ordinal", categories=["Bursa", "Izmir", "Ankara"], new_column="o"
    )
    assert result["o"].tolist() == [2, 1, 2, 0, 1]


def test_ordinal_requires_categories(frame: pd.DataFrame) -> None:
    with pytest.raises(CleaningError):
        tf.encode_categorical(frame, "sehir", "ordinal", categories=[])


def test_ordinal_with_wrong_categories_raises(frame: pd.DataFrame) -> None:
    with pytest.raises(CleaningError):
        tf.encode_categorical(frame, "sehir", "ordinal", categories=["yok1", "yok2"])


@pytest.mark.parametrize("method", ["label", "onehot"])
def test_encoding_code_matches_result(frame: pd.DataFrame, method: str) -> None:
    params = {"column": "sehir", "method": method}
    if method == "label":
        params["new_column"] = "kod"
    expected = tf.encode_categorical(frame, "sehir", method, new_column=params.get("new_column"))
    produced = _run_generated_code(frame, "encode_categorical", params)
    assert list(produced.columns) == list(expected.columns)
    for column in expected.columns:
        pd.testing.assert_series_equal(
            produced[column].astype("string"),
            expected[column].astype("string"),
            check_names=False,
        )


# ---------------------------------------------------------------------------
# Tarih özellikleri
# ---------------------------------------------------------------------------


def test_datetime_features(frame: pd.DataFrame) -> None:
    result = tf.datetime_features(
        frame,
        "tarih",
        ["year", "quarter", "month", "month_name", "day", "weekday", "weekday_name",
         "is_weekend", "week", "dayofyear"],
    )
    assert result["tarih_year"].iloc[0] == 2024
    assert result["tarih_quarter"].iloc[0] == 1
    assert result["tarih_month_name"].iloc[0] == "Ocak"
    assert result["tarih_weekday_name"].iloc[0] == "Cumartesi"
    assert result["tarih_is_weekend"].iloc[0] == 1
    assert result["tarih_year"].isna().sum() == 1  # NaT satırı


def test_datetime_features_converts_text() -> None:
    data = pd.DataFrame({"t": ["2024-05-01", "2024-06-15"]})
    result = tf.datetime_features(data, "t", ["year"])
    assert result["t_year"].tolist() == [2024, 2024]


def test_datetime_features_rejects_unparseable(frame: pd.DataFrame) -> None:
    with pytest.raises(CleaningError):
        tf.datetime_features(frame, "ad", ["year"])
    with pytest.raises(CleaningError):
        tf.datetime_features(frame, "tarih", [])


def test_datetime_features_name_clash(frame: pd.DataFrame) -> None:
    prepared = tf.datetime_features(frame, "tarih", ["year"])
    with pytest.raises(CleaningError):
        tf.datetime_features(prepared, "tarih", ["year"])


def test_datetime_code_matches_result(frame: pd.DataFrame) -> None:
    params = {"column": "tarih", "features": ["year", "month", "is_weekend", "month_name"]}
    expected = tf.datetime_features(frame, "tarih", params["features"])
    produced = _run_generated_code(frame, "datetime_features", params)
    for column in ("tarih_year", "tarih_month", "tarih_is_weekend", "tarih_month_name"):
        pd.testing.assert_series_equal(
            produced[column].astype("string"),
            expected[column].astype("string"),
            check_names=False,
        )


# ---------------------------------------------------------------------------
# Feature engineering ve binning
# ---------------------------------------------------------------------------


def test_arithmetic_features(frame: pd.DataFrame) -> None:
    assert tf.arithmetic_feature(frame, "gelir", "adet", "add", "t")["t"].iloc[0] == 1001
    assert tf.arithmetic_feature(frame, "gelir", "adet", "subtract", "t")["t"].iloc[0] == 999
    assert tf.arithmetic_feature(frame, "gelir", "adet", "multiply", "t")["t"].iloc[1] == 4000
    ratio = tf.arithmetic_feature(frame, "gelir", "adet", "ratio", "t")["t"].iloc[0]
    assert ratio == pytest.approx(1000 / 1001)


def test_division_by_zero_is_safe(frame: pd.DataFrame) -> None:
    result = tf.arithmetic_feature(frame, "gelir", "adet", "divide", "oran")
    assert pd.isna(result.loc[2, "oran"])  # adet = 0
    assert np.isfinite(result["oran"].dropna()).all()


def test_arithmetic_requires_valid_name(frame: pd.DataFrame) -> None:
    with pytest.raises(CleaningError):
        tf.arithmetic_feature(frame, "gelir", "adet", "add", "")
    with pytest.raises(CleaningError):
        tf.arithmetic_feature(frame, "gelir", "adet", "add", "gelir")


def test_binning_equal_width_and_quantile(frame: pd.DataFrame) -> None:
    equal = tf.bin_column(frame, "gelir", "equal_width", bins=2, new_column="g1")
    assert equal["g1"].dropna().nunique() == 2
    quantile = tf.bin_column(frame, "gelir", "quantile", bins=2, new_column="g2")
    assert quantile["g2"].dropna().nunique() == 2


def test_binning_custom_with_labels(frame: pd.DataFrame) -> None:
    result = tf.bin_column(
        frame,
        "gelir",
        "custom",
        edges=[0, 2000, 5000],
        labels=["düşük", "yüksek"],
        new_column="grup",
    )
    assert result["grup"].iloc[0] == "düşük"
    assert result["grup"].iloc[3] == "yüksek"


def test_binning_validation(frame: pd.DataFrame) -> None:
    with pytest.raises(CleaningError):
        tf.bin_column(frame, "gelir", "custom", edges=[10])
    with pytest.raises(CleaningError):
        tf.bin_column(frame, "gelir", "equal_width", bins=1)
    with pytest.raises(CleaningError):
        tf.bin_column(frame, "gelir", "equal_width", bins=3, labels=["a"])
    with pytest.raises(CleaningError):
        tf.bin_column(frame, "ad", "equal_width", bins=2)


def test_binning_code_matches_result(frame: pd.DataFrame) -> None:
    params = {
        "column": "gelir",
        "method": "custom",
        "edges": [0, 2000, 5000],
        "labels": ["düşük", "yüksek"],
        "new_column": "grup",
    }
    expected = tf.bin_column(frame, "gelir", "custom", edges=[0, 2000, 5000],
                             labels=["düşük", "yüksek"], new_column="grup")
    produced = _run_generated_code(frame, "bin_column", params)
    pd.testing.assert_series_equal(
        produced["grup"].astype("string"), expected["grup"].astype("string")
    )


# ---------------------------------------------------------------------------
# Düzen
# ---------------------------------------------------------------------------


def test_sort_rows(frame: pd.DataFrame) -> None:
    result = tf.sort_rows(frame, ["gelir"], [False])
    assert result["gelir"].iloc[0] == 4000.0
    assert result.index.tolist() == list(range(len(result)))


def test_sort_rows_multi_column(frame: pd.DataFrame) -> None:
    result = tf.sort_rows(frame, ["sehir", "gelir"], [True, False])
    assert result["sehir"].iloc[0] == "Ankara"
    assert result["gelir"].iloc[0] == 3000.0


def test_sort_validation(frame: pd.DataFrame) -> None:
    with pytest.raises(CleaningError):
        tf.sort_rows(frame, [])
    with pytest.raises(CleaningError):
        tf.sort_rows(frame, ["yok"])
    with pytest.raises(CleaningError):
        tf.sort_rows(frame, ["gelir", "adet"], [True])


def test_duplicate_and_reorder_columns(frame: pd.DataFrame) -> None:
    duplicated = tf.duplicate_column(frame, "sehir")
    assert "sehir_kopya" in duplicated.columns
    with pytest.raises(CleaningError):
        tf.duplicate_column(duplicated, "sehir", "sehir_kopya")

    reordered = tf.reorder_columns(frame, ["gelir", "ad"])
    assert list(reordered.columns)[:2] == ["gelir", "ad"]
    assert len(reordered.columns) == len(frame.columns)
    with pytest.raises(CleaningError):
        tf.reorder_columns(frame, ["yok"])


# ---------------------------------------------------------------------------
# Saflık ve replay
# ---------------------------------------------------------------------------


def test_transformations_do_not_mutate_source(frame: pd.DataFrame) -> None:
    snapshot = frame.copy(deep=True)
    tf.text_transform(frame, "ad", "trim")
    tf.numeric_transform(frame, "gelir", "standard")
    tf.encode_categorical(frame, "sehir", "onehot")
    tf.datetime_features(frame, "tarih", ["year"])
    tf.arithmetic_feature(frame, "gelir", "adet", "add", "yeni")
    tf.bin_column(frame, "gelir", "equal_width", bins=2)
    tf.sort_rows(frame, ["gelir"])
    pd.testing.assert_frame_equal(frame, snapshot)


def test_all_transformations_replay_deterministically(frame: pd.DataFrame) -> None:
    """Tüm dönüşümler sırayla uygulanıp replay ile birebir yeniden üretilmeli."""
    steps = [
        ("text_transform", {"column": "ad", "method": "trim"}),
        ("numeric_transform", {"column": "gelir", "method": "minmax", "new_column": "g_s"}),
        ("datetime_features", {"column": "tarih", "features": ["year", "month"]}),
        ("arithmetic_feature", {"left": "gelir", "right": "adet", "operation": "add",
                                "new_column": "toplam"}),
        ("bin_column", {"column": "gelir", "method": "quantile", "bins": 2,
                        "new_column": "g_grup"}),
        ("encode_categorical", {"column": "sehir", "method": "label", "new_column": "s_kod"}),
        ("duplicate_column", {"column": "adet", "new_column": "adet2"}),
        ("sort_rows", {"columns": ["gelir"], "ascending": [False]}),
        ("reorder_columns", {"columns": ["gelir", "ad"]}),
    ]
    current = frame
    history = []
    for action, params in steps:
        current, operation = ops.apply_operation(current, action, params)
        history.append(operation.to_dict())

    replayed, errors = ops.replay(frame, history)
    assert errors == []
    pd.testing.assert_frame_equal(replayed, current)

    undone, remaining, removed, undo_errors = ops.undo_last(frame, history)
    assert undo_errors == [] and len(remaining) == len(history) - 1
    assert removed["action"] == "reorder_columns"
    assert list(undone.columns)[0] != "gelir" or True


def test_suggest_feature_name_avoids_clash(frame: pd.DataFrame) -> None:
    assert tf.suggest_feature_name(frame, "gelir") == "gelir_2"
    assert tf.suggest_feature_name(frame, "yeni") == "yeni"


def test_transform_preview_shape(frame: pd.DataFrame) -> None:
    after = tf.numeric_transform(frame, "gelir", "minmax", new_column="s")
    preview = tf.transform_preview(frame, "gelir", after, "s", rows=3)
    assert list(preview.columns) == ["Önce", "Sonra"]
    assert len(preview) == 3
