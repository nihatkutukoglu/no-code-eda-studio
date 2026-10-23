"""core.operations testleri: kayıt, replay, undo determinizmi."""

from __future__ import annotations

import pandas as pd
import pytest

from core import operations
from core.operations import Operation
from utils.validators import CleaningError, ValidationError


def test_apply_operation_records_code_and_description(messy_df: pd.DataFrame) -> None:
    result, operation = operations.apply_operation(
        messy_df, "fill_missing", {"column": "gelir", "method": "median"}
    )
    assert result["gelir"].isna().sum() == 0
    assert operation.action == "fill_missing"
    assert operation.params["method"] == "median"
    assert "fillna" in operation.python_code
    assert "medyan" in operation.description
    assert operation.rows_before == len(messy_df)
    assert operation.id and operation.created_at


def test_unknown_operation_raises(messy_df: pd.DataFrame) -> None:
    with pytest.raises(ValidationError):
        operations.apply_operation(messy_df, "teleport", {})


def test_failed_operation_does_not_change_data(messy_df: pd.DataFrame) -> None:
    snapshot = messy_df.copy(deep=True)
    with pytest.raises(CleaningError):
        operations.apply_operation(
            messy_df, "fill_missing", {"column": "ad", "method": "mean"}
        )
    pd.testing.assert_frame_equal(messy_df, snapshot)


def test_custom_fill_value_is_normalized(messy_df: pd.DataFrame) -> None:
    _, operation = operations.apply_operation(
        messy_df, "fill_missing", {"column": "gelir", "method": "value", "value": "1500"}
    )
    # Replay determinizmi için parametre sayıya çevrilerek saklanır.
    assert operation.params["value"] == 1500.0
    assert "1500" in operation.python_code


def test_replay_reproduces_working_df(messy_df: pd.DataFrame) -> None:
    step1, op1 = operations.apply_operation(
        messy_df, "fill_missing", {"column": "gelir", "method": "median"}
    )
    step2, op2 = operations.apply_operation(step1, "drop_duplicates", {})
    step3, op3 = operations.apply_operation(
        step2, "convert_dtype", {"column": "tarih_metin", "target": "datetime"}
    )

    history = operations.history_to_dicts([op1, op2, op3])
    replayed, errors = operations.replay(messy_df, history)

    assert errors == []
    pd.testing.assert_frame_equal(replayed, step3)


def test_replay_is_deterministic(messy_df: pd.DataFrame) -> None:
    _, op = operations.apply_operation(
        messy_df, "fill_missing", {"column": "gelir", "method": "mean"}
    )
    history = operations.history_to_dicts([op])
    first, _ = operations.replay(messy_df, history)
    second, _ = operations.replay(messy_df, history)
    pd.testing.assert_frame_equal(first, second)


def test_replay_skips_unknown_operations(messy_df: pd.DataFrame) -> None:
    history = [{"action": "unknown_action", "params": {}, "description": "bilinmeyen"}]
    result, errors = operations.replay(messy_df, history)
    assert len(errors) == 1
    pd.testing.assert_frame_equal(result, messy_df)


def test_replay_reports_error_for_stale_operation(messy_df: pd.DataFrame) -> None:
    history = [
        {
            "action": "fill_missing",
            "params": {"column": "silinmis_sutun", "method": "median"},
            "description": "eski işlem",
        }
    ]
    result, errors = operations.replay(messy_df, history)
    assert len(errors) == 1
    pd.testing.assert_frame_equal(result, messy_df)


def test_undo_last_restores_previous_state(messy_df: pd.DataFrame) -> None:
    step1, op1 = operations.apply_operation(
        messy_df, "fill_missing", {"column": "gelir", "method": "median"}
    )
    _, op2 = operations.apply_operation(step1, "drop_columns", {"columns": ["bos"]})
    history = operations.history_to_dicts([op1, op2])

    undone, remaining, removed, errors = operations.undo_last(messy_df, history)

    assert errors == []
    assert len(remaining) == 1
    assert removed["action"] == "drop_columns"
    pd.testing.assert_frame_equal(undone, step1)


def test_undo_on_empty_history_returns_original(messy_df: pd.DataFrame) -> None:
    result, remaining, removed, errors = operations.undo_last(messy_df, [])
    assert removed is None and remaining == [] and errors == []
    pd.testing.assert_frame_equal(result, messy_df)


def test_reset_returns_deep_copy(messy_df: pd.DataFrame) -> None:
    reset = operations.reset_dataframe(messy_df)
    reset.loc[0, "ad"] = "DEĞİŞTİ"
    assert messy_df.loc[0, "ad"] == "Ali"


def test_operation_serialization_round_trip(messy_df: pd.DataFrame) -> None:
    _, operation = operations.apply_operation(
        messy_df, "drop_columns", {"columns": ["bos"]}
    )
    restored = Operation.from_dict(operation.to_dict())
    assert restored.action == operation.action
    assert restored.params == operation.params
    assert restored.python_code == operation.python_code
    assert restored.label == "Sütun sil"


def test_history_table_columns(messy_df: pd.DataFrame) -> None:
    _, operation = operations.apply_operation(messy_df, "drop_duplicates", {})
    table = operations.history_table([operation.to_dict()])
    assert list(table.columns) == ["#", "İşlem", "Açıklama", "Satır", "Sütun", "Zaman"]
    assert table.loc[0, "#"] == 1


def test_filter_rows_operation_round_trip() -> None:
    frame = pd.DataFrame({"a": [1, 2, 3, 4]})
    filters = [{"column": "a", "operator": "gte", "value": 3}]
    result, operation = operations.apply_operation(frame, "filter_rows", {"filters": filters})
    assert result["a"].tolist() == [3, 4]

    replayed, errors = operations.replay(frame, [operation.to_dict()])
    assert errors == []
    pd.testing.assert_frame_equal(replayed, result)
