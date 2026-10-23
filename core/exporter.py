"""Export katmanı: CSV, Excel, Parquet, JSON, Python script, Notebook,
Plotly HTML ve HTML EDA raporu.

Tüm fonksiyonlar indirilebilir ``bytes`` döndürür; Streamlit
``st.download_button`` doğrudan bu çıktıları kullanır.
"""

from __future__ import annotations

import io
import json
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from core.code_generator import generate_full_script
from core.notebook_generator import (
    build_notebook,
    notebook_filename,
    notebook_to_bytes,
    script_filename,
)
from core.report import build_html_report, report_filename
from utils.helpers import safe_filename
from utils.validators import EdaError

__all__ = [
    "DATA_FORMATS",
    "dataframe_to_csv_bytes",
    "dataframe_to_excel_bytes",
    "dataframe_to_parquet_bytes",
    "dataframe_to_json_bytes",
    "dataframe_to_bytes",
    "script_to_bytes",
    "notebook_bytes",
    "figure_to_html_bytes",
    "report_to_bytes",
    "insights_to_json_bytes",
    "export_filename",
    "notebook_filename",
    "script_filename",
    "report_filename",
]

#: Desteklenen veri export biçimleri: ``kod -> (etiket, uzantı, MIME)``
DATA_FORMATS: dict[str, tuple[str, str, str]] = {
    "csv": ("CSV", "csv", "text/csv"),
    "xlsx": (
        "Excel (XLSX)",
        "xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ),
    "parquet": ("Parquet", "parquet", "application/octet-stream"),
    "json": ("JSON (kayıt listesi)", "json", "application/json"),
}


def export_filename(dataset_name: str | None, suffix: str, extension: str) -> str:
    """``<dataset>_<suffix>.<extension>`` biçiminde dosya adı üretir."""
    base = safe_filename(dataset_name or "dataset")
    return f"{base}_{suffix}.{extension.lstrip('.')}"


def dataframe_to_csv_bytes(
    df: pd.DataFrame, separator: str = ",", encoding: str = "utf-8-sig"
) -> bytes:
    """DataFrame'i CSV baytlarına çevirir.

    Args:
        df: Dışa aktarılacak DataFrame.
        separator: Alan ayracı.
        encoding: Çıktı encoding'i (Excel uyumluluğu için varsayılan utf-8-sig).
    """
    buffer = io.StringIO()
    df.to_csv(buffer, index=False, sep=separator)
    return buffer.getvalue().encode(encoding, errors="replace")


def dataframe_to_excel_bytes(
    df: pd.DataFrame,
    sheet_name: str = "Veri",
    extra_sheets: dict[str, pd.DataFrame] | None = None,
) -> bytes:
    """DataFrame'i XLSX baytlarına çevirir.

    Args:
        df: Ana veri sayfası.
        sheet_name: Ana sayfanın adı.
        extra_sheets: Ek sayfalar (``{"İşlem Geçmişi": df}`` gibi).

    Raises:
        EdaError: Excel yazma motoru yoksa veya yazma başarısız olursa.
    """
    buffer = io.BytesIO()
    try:
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            _write_sheet(writer, df, sheet_name)
            for name, extra in (extra_sheets or {}).items():
                if extra is not None and not extra.empty:
                    _write_sheet(writer, extra, name)
    except ImportError as exc:
        raise EdaError(
            "Excel export için 'openpyxl' paketi gerekli. "
            "requirements.txt içindeki paketleri kurun."
        ) from exc
    except Exception as exc:
        raise EdaError(f"Excel dosyası oluşturulamadı: {exc}") from exc
    return buffer.getvalue()


def _write_sheet(writer: pd.ExcelWriter, df: pd.DataFrame, sheet_name: str) -> None:
    """Excel sayfasını güvenli adla ve timezone'suz yazar."""
    safe_sheet = str(sheet_name)[:31] or "Sayfa"
    frame = df.copy()
    for column in frame.columns:
        series = frame[column]
        # Excel timezone'lu tarihleri desteklemez.
        if isinstance(series.dtype, pd.DatetimeTZDtype):
            frame[column] = series.dt.tz_localize(None)
        elif isinstance(series.dtype, pd.CategoricalDtype):
            frame[column] = series.astype(str)
    frame.columns = [str(c) for c in frame.columns]
    frame.to_excel(writer, sheet_name=safe_sheet, index=False)


def dataframe_to_parquet_bytes(df: pd.DataFrame) -> bytes:
    """DataFrame'i Parquet baytlarına çevirir.

    Raises:
        EdaError: pyarrow yoksa veya sütun tipleri Parquet'e yazılamıyorsa.
    """
    buffer = io.BytesIO()
    frame = df.copy()
    frame.columns = [str(column) for column in frame.columns]
    try:
        frame.to_parquet(buffer, index=False)
    except ImportError as exc:
        raise EdaError(
            "Parquet export için 'pyarrow' paketi gerekli. "
            "requirements.txt içindeki paketleri kurun."
        ) from exc
    except Exception:
        # Karışık tipli object sütunları Parquet'e yazılamaz; metne çevirip dene.
        fallback = frame.copy()
        for column in fallback.columns:
            if fallback[column].dtype == object:
                fallback[column] = fallback[column].astype("string")
        buffer = io.BytesIO()
        try:
            fallback.to_parquet(buffer, index=False)
        except Exception as exc:
            raise EdaError(
                f"Parquet dosyası oluşturulamadı: {exc}. Karışık tipli sütunları "
                "önce dönüştürmeyi deneyin."
            ) from exc
    return buffer.getvalue()


def dataframe_to_json_bytes(df: pd.DataFrame, orient: str = "records") -> bytes:
    """DataFrame'i JSON baytlarına çevirir (varsayılan: kayıt listesi)."""
    frame = df.copy()
    frame.columns = [str(column) for column in frame.columns]
    try:
        text = frame.to_json(orient=orient, force_ascii=False, indent=2, date_format="iso")
    except Exception as exc:
        raise EdaError(f"JSON dosyası oluşturulamadı: {exc}") from exc
    return (text or "[]").encode("utf-8")


def dataframe_to_bytes(df: pd.DataFrame, file_format: str, **kwargs: Any) -> bytes:
    """Seçilen biçime göre veri export'u yapar.

    Args:
        df: Dışa aktarılacak DataFrame.
        file_format: ``DATA_FORMATS`` içindeki bir anahtar.
        **kwargs: Biçime özel parametreler (CSV ayracı gibi).

    Raises:
        EdaError: Biçim desteklenmiyorsa.
    """
    if file_format not in DATA_FORMATS:
        raise EdaError(f"Desteklenmeyen export biçimi: {file_format}")
    if file_format == "csv":
        return dataframe_to_csv_bytes(df, kwargs.get("separator", ","))
    if file_format == "xlsx":
        return dataframe_to_excel_bytes(
            df, kwargs.get("sheet_name", "Veri"), kwargs.get("extra_sheets")
        )
    if file_format == "parquet":
        return dataframe_to_parquet_bytes(df)
    return dataframe_to_json_bytes(df)


def script_to_bytes(
    meta: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
    charts: list[dict[str, Any]] | None = None,
    include_cleaning: bool = True,
    include_charts: bool = True,
    analysis_code: str | None = None,
    ml_code: str | None = None,
) -> bytes:
    """Oturumu çalıştırılabilir ``.py`` script baytlarına çevirir."""
    script = generate_full_script(
        meta,
        history=history,
        charts=charts,
        include_cleaning=include_cleaning,
        include_charts=include_charts,
        analysis_code=analysis_code,
        ml_code=ml_code,
    )
    return script.encode("utf-8")


def notebook_bytes(
    meta: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
    charts: list[dict[str, Any]] | None = None,
    insights: list[dict[str, Any]] | None = None,
    include_statistics: bool = True,
    include_relationships: bool = True,
    include_outliers: bool = True,
    numeric_columns: list[str] | None = None,
    ml_code: str | None = None,
) -> bytes:
    """Oturumu geçerli bir ``.ipynb`` dosyasının baytlarına çevirir.

    Raises:
        EdaError: Notebook üretilemezse.
    """
    try:
        notebook = build_notebook(
            meta,
            history=history,
            charts=charts,
            insights=insights,
            include_statistics=include_statistics,
            include_relationships=include_relationships,
            include_outliers=include_outliers,
            numeric_columns=numeric_columns,
            ml_code=ml_code,
        )
        return notebook_to_bytes(notebook)
    except EdaError:
        raise
    except Exception as exc:
        raise EdaError(f"Notebook oluşturulamadı: {exc}") from exc


def figure_to_html_bytes(figure: go.Figure) -> bytes:
    """Plotly grafiğini tek dosyalık interaktif HTML baytlarına çevirir."""
    try:
        html = figure.to_html(include_plotlyjs="cdn", full_html=True)
    except Exception as exc:
        raise EdaError(f"Grafik HTML olarak dışa aktarılamadı: {exc}") from exc
    return html.encode("utf-8")


def figure_to_png_bytes(figure: go.Figure, scale: float = 2.0) -> bytes:
    """Plotly grafiğini PNG baytlarına çevirir (kaleido kuruluysa).

    Raises:
        EdaError: ``kaleido`` kurulu değilse veya dönüştürme başarısız olursa.
    """
    try:
        return figure.to_image(format="png", scale=scale)
    except Exception as exc:  # kaleido yok veya render hatası
        raise EdaError(
            "PNG export için 'kaleido' paketi gerekli (pip install kaleido). "
            "Bunun yerine HTML olarak indirebilirsiniz."
        ) from exc


def report_to_bytes(
    df: pd.DataFrame,
    meta: dict[str, Any] | None = None,
    charts: list[dict[str, Any]] | None = None,
    history: list[dict[str, Any]] | None = None,
    insights: list[dict[str, Any]] | None = None,
    include_charts: bool = True,
) -> bytes:
    """HTML EDA raporunu baytlara çevirir."""
    html = build_html_report(
        df,
        meta=meta,
        charts=charts,
        history=history,
        insights=insights,
        include_charts=include_charts,
    )
    return html.encode("utf-8")


def insights_to_json_bytes(insights: list[dict[str, Any]]) -> bytes:
    """Kural tabanlı bulguları JSON baytlarına çevirir."""
    payload = [
        {
            "severity": item.get("severity"),
            "category": item.get("category"),
            "column": item.get("column"),
            "title": item.get("title"),
            "message": item.get("message"),
            "metric": item.get("metric"),
            "rule": item.get("rule"),
        }
        for item in insights or []
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
