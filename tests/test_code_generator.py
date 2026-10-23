"""core.code_generator testleri: üretilen kod çalıştırılabilir ve güvenli mi?"""

from __future__ import annotations

import pandas as pd
import pytest

from core import operations
from core.chart_builder import build_chart
from core.code_generator import (
    build_operation_code,
    generate_charts_section,
    generate_cleaning_section,
    generate_full_script,
    generate_imports,
)
from utils.validators import ValidationError


def test_build_operation_code_for_each_action() -> None:
    cases = [
        ("drop_missing_rows", {"columns": ["a"], "how": "any"}),
        ("fill_missing", {"column": "a", "method": "median"}),
        ("fill_missing", {"column": "a", "method": "mode"}),
        ("fill_missing", {"column": "a", "method": "value", "value": 5}),
        ("fill_missing", {"column": "a", "method": "ffill"}),
        ("drop_duplicates", {"subset": ["a"], "keep": "last"}),
        ("rename_column", {"column": "a", "new_name": "b"}),
        ("drop_columns", {"columns": ["a", "b"]}),
        ("convert_dtype", {"column": "a", "target": "integer"}),
        ("convert_dtype", {"column": "a", "target": "datetime", "datetime_format": "%d/%m/%Y"}),
        ("convert_dtype", {"column": "a", "target": "boolean"}),
        ("drop_outlier_rows", {"column": "a", "factor": 1.5}),
        ("filter_rows", {"filters": [{"column": "a", "operator": "gt", "value": 1}]}),
    ]
    for action, params in cases:
        code = build_operation_code(action, params)
        assert code.strip()
        compile(code, "<generated>", "exec")


def test_unknown_action_raises() -> None:
    with pytest.raises(ValidationError):
        build_operation_code("teleport", {})


def test_missing_parameter_raises() -> None:
    with pytest.raises(ValidationError):
        build_operation_code("fill_missing", {})


def test_column_names_with_spaces_and_quotes_are_escaped() -> None:
    column = "O'Brien Gelir \"net\" \\ ters"
    code = build_operation_code("fill_missing", {"column": column, "method": "mean"})
    compile(code, "<generated>", "exec")

    frame = pd.DataFrame({column: [1.0, None, 3.0]})
    namespace = {"df": frame, "pd": pd}
    exec(compile(code, "<generated>", "exec"), namespace)  # noqa: S102 - test amaçlı
    assert namespace["df"][column].isna().sum() == 0


def test_header_handles_windows_paths(tmp_path) -> None:
    """Windows yolu içeren dosya adları geçerli kod üretmelidir."""
    meta = {"filename": str(tmp_path / "veri.csv"), "file_type": "csv"}
    script = generate_full_script(meta, [], [])
    compile(script, "<generated>", "exec")


def test_generated_script_runs_and_reproduces_result(tmp_path, messy_df: pd.DataFrame) -> None:
    """Üretilen script'i gerçekten çalıştırıp sonucu karşılaştırır."""
    csv_path = tmp_path / "veri.csv"
    messy_df.to_csv(csv_path, index=False)
    source = pd.read_csv(csv_path)

    step1, op1 = operations.apply_operation(
        source, "fill_missing", {"column": "gelir", "method": "median"}
    )
    step2, op2 = operations.apply_operation(step1, "drop_duplicates", {})
    step3, op3 = operations.apply_operation(step2, "drop_columns", {"columns": ["bos"]})
    expected = step3

    meta = {"filename": str(csv_path), "file_type": "csv", "separator": ","}
    script = generate_full_script(meta, operations.history_to_dicts([op1, op2, op3]), [])

    namespace: dict[str, object] = {}
    exec(compile(script, "<script>", "exec"), namespace)  # noqa: S102 - test amaçlı
    produced = namespace["df"]

    pd.testing.assert_frame_equal(
        produced.reset_index(drop=True), expected.reset_index(drop=True)
    )


def test_generated_chart_code_runs(numeric_df: pd.DataFrame) -> None:
    result = build_chart(numeric_df, "scatter", {"x": "x", "y": "y"})
    namespace = {"df": numeric_df.copy()}
    code = result.spec["python_code"].replace("fig.show()", "")
    exec(  # noqa: S102 - test amaçlı
        compile("import plotly.express as px\n" + code, "<chart>", "exec"), namespace
    )
    assert namespace["fig"] is not None


def test_cleaning_section_includes_comments(messy_df: pd.DataFrame) -> None:
    _, operation = operations.apply_operation(messy_df, "drop_duplicates", {})
    section = generate_cleaning_section([operation.to_dict()])
    assert section.startswith("# 1.")
    assert "drop_duplicates" in section

    plain = generate_cleaning_section([operation.to_dict()], with_comments=False)
    assert not plain.startswith("#")


def test_charts_section_skips_empty_code() -> None:
    assert generate_charts_section([{"python_code": ""}]) == ""


def test_imports_include_plotly_only_when_needed() -> None:
    assert "plotly" in generate_imports(True)
    assert "plotly" not in generate_imports(False)


def test_empty_session_script_is_still_valid(dataset_meta: dict) -> None:
    script = generate_full_script(dataset_meta, [], [])
    compile(script, "<empty>", "exec")
    assert "pd.read_csv" in script
    assert "describe" in script
