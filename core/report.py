"""Tek dosyalık HTML EDA raporu üretir (API/LLM kullanmadan).

Rapor; dataset özeti, veri kalitesi, kalite skoru, istatistikler, korelasyonlar,
kural tabanlı bulgular ve kullanıcının oluşturduğu grafikleri içerir.
Çıktı tek bir ``.html`` dosyasıdır: Plotly grafiklerini CDN üzerinden yükler,
başka hiçbir dosyaya ihtiyaç duymaz.
"""

from __future__ import annotations

import html
from datetime import datetime
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from core import profiler, statistics as stats_mod
from core.chart_builder import PLOTLY_TEMPLATE, build_chart
from core.correlations import correlation_matrix, strong_pairs
from core.data_quality_score import SCORE_DISCLAIMER, compute_quality_score
from core.insights import CATEGORIES, SEVERITIES, generate_insights
from utils import dataframe_utils as dfu
from utils.helpers import format_bytes, format_number, format_percent
from utils.validators import EdaError

__all__ = ["build_html_report", "report_filename"]

_CSS = """
:root {
  --bg: #f7f8fa;
  --card: #ffffff;
  --text: #1f2430;
  --muted: #667085;
  --border: #e6e8ee;
  --accent: #2563eb;
  --critical: #c62828;
  --warning: #b26a00;
  --info: #2563eb;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 32px 20px 64px;
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  line-height: 1.55;
}
.wrap { max-width: 1120px; margin: 0 auto; }
header.page { margin-bottom: 28px; }
header.page h1 { margin: 0 0 6px; font-size: 1.9rem; letter-spacing: -0.02em; }
header.page .sub { color: var(--muted); font-size: 0.95rem; }
section { margin: 28px 0; }
h2 {
  font-size: 1.2rem; margin: 0 0 14px; padding-bottom: 8px;
  border-bottom: 1px solid var(--border);
}
h3 { font-size: 1rem; margin: 20px 0 8px; color: var(--muted); font-weight: 600; }
.card {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 12px; padding: 18px 20px; margin-bottom: 16px;
}
.metrics { display: flex; flex-wrap: wrap; gap: 12px; }
.metric {
  background: var(--card); border: 1px solid var(--border); border-radius: 12px;
  padding: 14px 18px; min-width: 150px; flex: 1 1 150px;
}
.metric .label { color: var(--muted); font-size: 0.8rem; text-transform: uppercase; letter-spacing: .04em; }
.metric .value { font-size: 1.45rem; font-weight: 650; margin-top: 2px; }
table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); }
th { color: var(--muted); font-weight: 600; background: #fbfcfe; position: sticky; top: 0; }
tr:last-child td { border-bottom: none; }
.table-wrap { max-height: 420px; overflow: auto; border: 1px solid var(--border); border-radius: 10px; }
.insight { border-left: 3px solid var(--border); padding: 10px 14px; margin-bottom: 10px; background: var(--card); border-radius: 0 10px 10px 0; }
.insight.critical { border-left-color: var(--critical); }
.insight.warning { border-left-color: var(--warning); }
.insight.info { border-left-color: var(--info); }
.insight .title { font-weight: 600; }
.insight .meta { color: var(--muted); font-size: 0.82rem; margin-top: 2px; }
.score { display: flex; align-items: center; gap: 22px; flex-wrap: wrap; }
.score .big { font-size: 3rem; font-weight: 700; line-height: 1; }
.score .big span { font-size: 1.1rem; color: var(--muted); font-weight: 500; }
.note { color: var(--muted); font-size: 0.85rem; }
.pill {
  display: inline-block; padding: 2px 10px; border-radius: 999px;
  background: #eef2ff; color: var(--accent); font-size: 0.78rem; font-weight: 600;
}
footer { margin-top: 40px; color: var(--muted); font-size: 0.82rem; border-top: 1px solid var(--border); padding-top: 16px; }
.empty { color: var(--muted); font-style: italic; }
@media (max-width: 640px) {
  body { padding: 20px 12px 48px; }
  .metric { min-width: 120px; }
}
"""


def report_filename(dataset_name: str | None) -> str:
    """``<dataset>_eda_raporu.html`` biçiminde dosya adı üretir."""
    from utils.helpers import safe_filename

    return f"{safe_filename(dataset_name or 'dataset')}_eda_raporu.html"


def _esc(value: Any) -> str:
    """HTML kaçışlı metin döndürür."""
    return html.escape("" if value is None else str(value))


def _metric(label: str, value: Any) -> str:
    """Metrik kartı HTML'i üretir."""
    return (
        '<div class="metric">'
        f'<div class="label">{_esc(label)}</div>'
        f'<div class="value">{_esc(value)}</div>'
        "</div>"
    )


def _table(df: pd.DataFrame, max_rows: int = 60, empty_message: str = "Kayıt yok.") -> str:
    """DataFrame'i HTML tablosuna çevirir."""
    if df is None or df.empty:
        return f'<p class="empty">{_esc(empty_message)}</p>'
    view = dfu.safe_display_df(df.head(max_rows))
    head = "".join(f"<th>{_esc(column)}</th>" for column in view.columns)
    rows = []
    for _, row in view.iterrows():
        cells = "".join(f"<td>{_esc(_format_cell(value))}</td>" for value in row)
        rows.append(f"<tr>{cells}</tr>")
    more = (
        f'<p class="note">İlk {max_rows} satır gösteriliyor '
        f"(toplam {format_number(len(df))}).</p>"
        if len(df) > max_rows
        else ""
    )
    return (
        '<div class="table-wrap"><table><thead><tr>'
        + head
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
        + more
    )


def _format_cell(value: Any) -> str:
    """Tablo hücresini okunabilir biçimde döndürür."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if isinstance(value, float):
        return format_number(value, 3)
    return str(value)


def _figure_div(figure: go.Figure, height: int = 420) -> str:
    """Plotly figürünü (plotly.js hariç) HTML parçasına çevirir."""
    figure.update_layout(template=PLOTLY_TEMPLATE, height=height)
    return figure.to_html(include_plotlyjs=False, full_html=False, default_width="100%")


def _overview_section(df: pd.DataFrame, meta: dict[str, Any]) -> str:
    """Dataset özeti bölümü."""
    info = dfu.dataset_shape_info(df)
    metrics = "".join(
        [
            _metric("Satır", format_number(info["rows"])),
            _metric("Sütun", format_number(info["columns"])),
            _metric("Toplam hücre", format_number(info["cells"])),
            _metric("Bellek", format_bytes(info["memory_bytes"])),
            _metric("Boş hücre", f"{format_number(info['missing_cells'])} ({format_percent(info['missing_ratio'])})"),
            _metric("Tekrarlı satır", format_number(info["duplicate_rows"])),
            _metric("Sayısal sütun", format_number(info["numeric_cols"])),
            _metric("Kategorik sütun", format_number(info["categorical_cols"])),
            _metric("Tarih sütunu", format_number(info["datetime_cols"])),
        ]
    )
    profile = dfu.column_profile(df)
    return (
        "<section><h2>1. Dataset Özeti</h2>"
        f'<div class="metrics">{metrics}</div>'
        "<h3>Sütun profili</h3>"
        f"{_table(profile, max_rows=80)}"
        "</section>"
    )


def _quality_section(df: pd.DataFrame) -> str:
    """Veri kalitesi ve skor bölümü."""
    score = compute_quality_score(df)
    components = "".join(
        f"<tr><td>{_esc(component.label)}</td>"
        f"<td>{format_number(component.score, 1)}</td>"
        f"<td>{component.weight}</td>"
        f"<td>{format_number(component.weighted_points, 1)}</td>"
        f"<td>{_esc(component.detail)}</td></tr>"
        for component in score.components
    )
    missing = profiler.missing_value_table(df, only_missing=True)
    duplicates = profiler.duplicate_report(df)
    constants = profiler.constant_column_table(df)
    dtype_issues = profiler.dtype_check_table(df)

    return (
        "<section><h2>2. Veri Kalitesi</h2>"
        '<div class="card"><div class="score">'
        f'<div class="big">{score.overall}<span>/100</span></div>'
        f'<div><div class="pill">{_esc(score.label)}</div>'
        f'<p class="note">{_esc(SCORE_DISCLAIMER)}</p></div>'
        "</div>"
        "<table><thead><tr><th>Bileşen</th><th>Alt skor</th><th>Ağırlık</th>"
        "<th>Katkı</th><th>Açıklama</th></tr></thead>"
        f"<tbody>{components}</tbody></table></div>"
        "<h3>Boş değer içeren sütunlar</h3>"
        f"{_table(missing, empty_message='Hiçbir sütunda boş değer yok.')}"
        "<h3>Tekrarlı satırlar</h3>"
        f'<p>{format_number(duplicates["count"])} tekrarlı satır '
        f'({format_percent(duplicates["ratio"])}).</p>'
        "<h3>Sabit (tek değerli) sütunlar</h3>"
        f"{_table(constants, empty_message='Sabit sütun bulunamadı.')}"
        "<h3>Veri tipi uyarıları</h3>"
        f"{_table(dtype_issues, empty_message='Belirgin tip sorunu bulunamadı.')}"
        "</section>"
    )


def _statistics_section(df: pd.DataFrame) -> str:
    """İstatistik bölümü."""
    numeric_table = stats_mod.describe_all_numeric(df)
    categorical = dfu.categorical_columns(df)
    blocks: list[str] = [
        "<section><h2>3. İstatistiksel Özet</h2>",
        "<h3>Sayısal sütunlar</h3>",
        _table(numeric_table, empty_message="Sayısal sütun bulunamadı."),
    ]
    if categorical:
        blocks.append("<h3>Kategorik sütunlar (ilk 5)</h3>")
        for column in categorical[:5]:
            try:
                summary = stats_mod.categorical_summary(df, column, top_n=5)
            except EdaError:
                continue
            blocks.append(
                f'<div class="card"><strong>{_esc(column)}</strong>'
                f'<p class="note">{format_number(summary["unique"])} tekil değer · '
                f'mod: {_esc(summary["mode"])} ({format_number(summary["mode_frequency"])}) · '
                f'dengelilik: {format_number(summary["balance"] * 100, 1)}%</p>'
                f'{_table(summary["top_values"], max_rows=5)}</div>'
            )
    blocks.append("</section>")
    return "".join(blocks)


def _correlation_section(df: pd.DataFrame) -> str:
    """Korelasyon bölümü."""
    numeric = dfu.numeric_columns(df)
    if len(numeric) < 2:
        return (
            "<section><h2>4. Korelasyonlar</h2>"
            '<p class="empty">Korelasyon için en az 2 sayısal sütun gerekir.</p>'
            "</section>"
        )
    try:
        matrix = correlation_matrix(df, numeric[:12], "pearson")
    except EdaError:
        return (
            "<section><h2>4. Korelasyonlar</h2>"
            '<p class="empty">Korelasyon hesaplanamadı.</p></section>'
        )
    pairs = strong_pairs(matrix, 0.70)
    try:
        chart = build_chart(
            df, "heatmap", {"columns": numeric[:12], "method": "pearson"}
        )
        figure_html = _figure_div(chart.figure, height=460)
    except EdaError:
        figure_html = ""
    return (
        "<section><h2>4. Korelasyonlar</h2>"
        f"{figure_html}"
        "<h3>Güçlü ilişkiler (|r| ≥ 0.70)</h3>"
        f"{_table(pairs, empty_message='Eşiği aşan güçlü ilişki bulunamadı.')}"
        '<p class="note">Korelasyon nedensellik göstermez.</p>'
        "</section>"
    )


def _insights_section(insights: list[dict[str, Any]]) -> str:
    """Kural tabanlı bulgular bölümü."""
    if not insights:
        return (
            "<section><h2>5. Kural Tabanlı Bulgular</h2>"
            '<p class="empty">Tanımlı kurallara göre öne çıkan bir sorun '
            "bulunamadı.</p></section>"
        )
    items = []
    for item in insights:
        severity = str(item.get("severity", "info"))
        column = item.get("column")
        meta_parts = [
            SEVERITIES.get(severity, severity),
            CATEGORIES.get(str(item.get("category")), str(item.get("category"))),
        ]
        if column:
            meta_parts.append(str(column))
        if item.get("rule"):
            meta_parts.append(f"kural: {item['rule']}")
        items.append(
            f'<div class="insight {_esc(severity)}">'
            f'<div class="title">{_esc(item.get("title"))}</div>'
            f'<div>{_esc(item.get("message"))}</div>'
            f'<div class="meta">{_esc(" · ".join(meta_parts))}</div>'
            "</div>"
        )
    return (
        "<section><h2>5. Kural Tabanlı Bulgular</h2>"
        f'<p class="note">{len(insights)} bulgu, önem derecesine göre sıralı.</p>'
        + "".join(items)
        + "</section>"
    )


def _charts_section(df: pd.DataFrame, charts: list[dict[str, Any]]) -> str:
    """Kullanıcının oluşturduğu grafikler bölümü."""
    if not charts:
        return (
            "<section><h2>6. Grafikler</h2>"
            '<p class="empty">Rapora eklenecek grafik seçilmedi.</p></section>'
        )
    blocks = ["<section><h2>6. Grafikler</h2>"]
    for index, chart in enumerate(charts, start=1):
        try:
            result = build_chart(
                df,
                str(chart.get("type")),
                dict(chart.get("params") or {}),
                source_note=chart.get("source_note"),
            )
        except EdaError as exc:
            blocks.append(
                f'<p class="empty">Grafik {index} yeniden oluşturulamadı: {_esc(exc)}</p>'
            )
            continue
        blocks.append(
            f'<div class="card"><strong>{index}. {_esc(chart.get("title"))}</strong>'
            f"{_figure_div(result.figure)}</div>"
        )
    blocks.append("</section>")
    return "".join(blocks)


def _history_section(history: list[dict[str, Any]]) -> str:
    """Uygulanan veri işlemleri bölümü."""
    if not history:
        return ""
    rows = "".join(
        f"<tr><td>{index}</td><td>{_esc(item.get('description'))}</td>"
        f"<td><code>{_esc(item.get('action'))}</code></td></tr>"
        for index, item in enumerate(history, start=1)
    )
    return (
        "<section><h2>7. Uygulanan Veri İşlemleri</h2>"
        "<table><thead><tr><th>#</th><th>Açıklama</th><th>İşlem</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
        '<p class="note">Bu işlemlerin Python kodu uygulamadan .py veya .ipynb '
        "olarak indirilebilir.</p></section>"
    )


def build_html_report(
    df: pd.DataFrame,
    meta: dict[str, Any] | None = None,
    charts: list[dict[str, Any]] | None = None,
    history: list[dict[str, Any]] | None = None,
    insights: list[dict[str, Any]] | None = None,
    include_charts: bool = True,
) -> str:
    """Tek dosyalık HTML EDA raporu üretir.

    Args:
        df: Raporlanacak DataFrame (``working_df``).
        meta: Dataset bilgisi (``filename`` vb.).
        charts: Rapora eklenecek grafik tanımları.
        history: Uygulanan operasyonlar.
        insights: Kural tabanlı bulgular (None ise yeniden üretilir).
        include_charts: Grafik bölümünü ekle.

    Returns:
        Tam HTML metni.

    Raises:
        EdaError: Veri yoksa.
    """
    if df is None or df.empty or df.shape[1] == 0:
        raise EdaError("Rapor oluşturmak için veri seti gerekli.")

    meta = meta or {}
    charts = charts or []
    history = history or []
    if insights is None:
        insights = generate_insights(df)

    dataset_name = str(meta.get("filename") or "dataset")
    generated_at = datetime.now().strftime("%d.%m.%Y %H:%M")

    sections = [
        _overview_section(df, meta),
        _quality_section(df),
        _statistics_section(df),
        _correlation_section(df),
        _insights_section(insights),
    ]
    if include_charts:
        sections.append(_charts_section(df, charts))
    sections.append(_history_section(history))

    plotly_script = (
        '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>'
    )
    return f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EDA Raporu — {_esc(dataset_name)}</title>
{plotly_script}
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">
<header class="page">
  <h1>Keşifsel Veri Analizi Raporu</h1>
  <div class="sub">
    <strong>{_esc(dataset_name)}</strong> · {format_number(len(df))} satır ×
    {format_number(df.shape[1])} sütun · oluşturma: {_esc(generated_at)}
  </div>
</header>
{"".join(sections)}
<footer>
  Bu rapor <strong>No-Code EDA Studio</strong> tarafından otomatik üretildi.
  Hiçbir dil modeli veya harici API kullanılmadı; tüm bulgular kural tabanlıdır.<br>
  Hatırlatmalar: Korelasyon nedensellik göstermez · İstatistiksel olarak aykırı
  bir değer hatalı veri olmak zorunda değildir · Veri kalite skoru evrensel bir
  standart değildir.
</footer>
</div>
</body>
</html>
"""
