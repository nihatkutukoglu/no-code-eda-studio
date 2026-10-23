"""No-Code Chart Builder: Plotly grafiklerini ve kod karşılıklarını üretir.

Her grafik için hem interaktif ``plotly.graph_objects.Figure`` hem de aynı
grafiği yeniden üreten Python kodu döndürülür. Grafik tanımları session state
içindeki ``generated_charts`` listesine yazılır ve Notebook/Script export
sırasında yeniden kullanılır.

Grafik üretimi DataFrame'i **değiştirmez**; bu yüzden ``operation_history``
yerine ayrı bir ``generated_charts`` yapısı kullanılır.

Tasarım kararları:
    * Arayüzü boğmamak için zorunlu alanlar az tutulur; başlık, etiket,
      sıralama, Top N gibi ayarlar "gelişmiş" parametrelerdir.
    * Gruplu/yığılmış çubuk ayrı menü maddesi değil, çubuk grafiğin
      ``barmode`` ayarıdır (daha az seçenek, daha net arayüz).
    * Büyük veri setlerinde noktasal grafikler örneklenir ve kullanıcı uyarılır.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from core.correlations import CORRELATION_METHODS, correlation_matrix
from utils import dataframe_utils as dfu
from utils.helpers import new_id, py_literal
from utils.validators import ChartError, ValidationError

__all__ = [
    "CHART_TYPES",
    "AGGREGATIONS",
    "BAR_MODES",
    "SORT_OPTIONS",
    "MAX_SCATTER_POINTS",
    "MAX_CATEGORIES",
    "PLOTLY_TEMPLATE",
    "COLOR_SEQUENCE",
    "ChartResult",
    "build_chart",
    "suggest_charts",
    "default_params",
    "figure_to_html",
]

#: Plotly tema adı (tüm grafiklerde tutarlı, sade görünüm).
PLOTLY_TEMPLATE = "plotly_white"
#: Sade ve birbiriyle uyumlu renk paleti (tek accent ağırlıklı).
COLOR_SEQUENCE = [
    "#2563eb",
    "#0ea5e9",
    "#14b8a6",
    "#f59e0b",
    "#8b5cf6",
    "#ef4444",
    "#64748b",
    "#22c55e",
]
#: Scatter/line gibi noktasal grafiklerde üst sınır; aşılırsa örnekleme yapılır.
MAX_SCATTER_POINTS = 20_000
#: Kategorik eksende gösterilecek maksimum kategori sayısı.
MAX_CATEGORIES = 50

#: Bar/pie grafiklerinde kullanılabilen aggregation fonksiyonları.
AGGREGATIONS: dict[str, str] = {
    "count": "count (adet)",
    "sum": "sum (toplam)",
    "mean": "mean (ortalama)",
    "median": "median (medyan)",
    "min": "min (en küçük)",
    "max": "max (en büyük)",
}

#: Çubuk grafik modları.
BAR_MODES: dict[str, str] = {
    "group": "Gruplu (yan yana)",
    "stack": "Yığılmış (üst üste)",
    "overlay": "Üst üste bindirilmiş",
}

#: Kategorik eksen sıralama seçenekleri.
SORT_OPTIONS: dict[str, str] = {
    "value_desc": "Değere göre (büyükten küçüğe)",
    "value_asc": "Değere göre (küçükten büyüğe)",
    "category": "Kategori adına göre",
    "none": "Sıralama yok (veri sırası)",
}

#: Grafik tipleri ve açıklamaları.
CHART_TYPES: dict[str, dict[str, Any]] = {
    "bar": {
        "label": "Bar Chart (Çubuk)",
        "icon": "📊",
        "help": "Kategorilere göre adet veya toplam/ortalama karşılaştırması. "
        "Renk sütunu seçerseniz gruplu veya yığılmış çubuk olur.",
        "group": "Karşılaştırma",
    },
    "line": {
        "label": "Line Chart (Çizgi)",
        "icon": "📈",
        "help": "Zaman veya sıralı bir eksen boyunca değişimi gösterir.",
        "group": "Zaman / Eğilim",
    },
    "area": {
        "label": "Area Chart (Alan)",
        "icon": "🏔️",
        "help": "Çizgi grafiğin altını doldurur; toplam büyüklüğü vurgular.",
        "group": "Zaman / Eğilim",
    },
    "scatter": {
        "label": "Scatter Plot (Dağılım)",
        "icon": "⚫",
        "help": "İki sayısal değişken arasındaki ilişkiyi gösterir.",
        "group": "İlişki",
    },
    "bubble": {
        "label": "Bubble Chart (Balon)",
        "icon": "🫧",
        "help": "Scatter'a üçüncü bir boyut (nokta büyüklüğü) ekler.",
        "group": "İlişki",
    },
    "histogram": {
        "label": "Histogram",
        "icon": "📶",
        "help": "Tek bir sayısal sütunun dağılımını gösterir.",
        "group": "Dağılım",
    },
    "box": {
        "label": "Box Plot (Kutu)",
        "icon": "🧰",
        "help": "Medyan, çeyrekler ve aykırı değerleri birlikte gösterir.",
        "group": "Dağılım",
    },
    "violin": {
        "label": "Violin Plot (Keman)",
        "icon": "🎻",
        "help": "Dağılımın yoğunluğunu kutu grafiğiyle birlikte gösterir.",
        "group": "Dağılım",
    },
    "pie": {
        "label": "Pie Chart (Pasta)",
        "icon": "🥧",
        "help": "Kategorilerin toplam içindeki payını gösterir.",
        "group": "Kompozisyon",
    },
    "donut": {
        "label": "Donut Chart (Halka)",
        "icon": "🍩",
        "help": "Pasta grafiğin ortası boş hali; payları daha sade gösterir.",
        "group": "Kompozisyon",
    },
    "heatmap": {
        "label": "Correlation Heatmap",
        "icon": "🔥",
        "help": "Sayısal sütunlar arasındaki korelasyonu renk haritasıyla gösterir.",
        "group": "İlişki",
    },
    "category_heatmap": {
        "label": "Kategori Heatmap (Çapraz tablo)",
        "icon": "🧮",
        "help": "İki kategorik sütunun kesişim adetlerini renkle gösterir.",
        "group": "İlişki",
    },
    "density_heatmap": {
        "label": "Density Heatmap (Yoğunluk)",
        "icon": "🌡️",
        "help": "Çok sayıda nokta olduğunda scatter yerine yoğunluk gösterir.",
        "group": "İlişki",
    },
}


@dataclass
class ChartResult:
    """Grafik üretim sonucunu taşır."""

    figure: go.Figure
    spec: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------


def _require(params: dict[str, Any], key: str, label: str) -> Any:
    """Zorunlu parametreyi doğrular."""
    value = params.get(key)
    if value is None or (isinstance(value, str) and not value.strip()) or (
        isinstance(value, (list, tuple)) and not value
    ):
        raise ChartError(f"{label} seçmelisiniz.")
    return value


def _check_column(df: pd.DataFrame, column: str, label: str) -> str:
    """Sütunun varlığını doğrular."""
    if column not in df.columns:
        raise ChartError(
            f"{label} olarak seçilen '{column}' sütunu veri setinde yok. "
            "Sütun silinmiş veya adı değişmiş olabilir."
        )
    return str(column)


def _check_numeric(df: pd.DataFrame, column: str, label: str) -> str:
    """Sütunun sayısal olduğunu doğrular."""
    column = _check_column(df, column, label)
    if column not in dfu.numeric_columns(df):
        raise ChartError(
            f"{label} sayısal bir sütun olmalı. '{column}' sütunu {df[column].dtype} "
            "tipinde. Veri Temizleme sayfasından tipi dönüştürebilirsiniz."
        )
    return column


def _title(params: dict[str, Any], default: str) -> str:
    """Kullanıcı başlığı yoksa varsayılan başlığı kullanır."""
    title = str(params.get("title") or "").strip()
    return title or default


def _sample_for_plot(df: pd.DataFrame, limit: int) -> tuple[pd.DataFrame, list[str]]:
    """Çok büyük veri setlerinde grafik için örnekleme yapar."""
    if len(df) <= limit:
        return df, []
    sampled = df.sample(limit, random_state=42)
    return sampled, [
        f"Veri seti {len(df):,} satır olduğu için grafik {limit:,} rastgele satır "
        "üzerinden çizildi (görüntüleme amaçlı örnekleme)."
    ]


def _aggregate(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str | None,
    aggregation: str,
    data_name: str = "df",
) -> tuple[pd.DataFrame, str, str]:
    """Bar/pie için gruplama yapar.

    Returns:
        ``(aggregated_df, value_column_name, python_code)``
    """
    if aggregation not in AGGREGATIONS:
        raise ChartError(f"Desteklenmeyen aggregation: {aggregation}")

    if aggregation == "count" or not value_col:
        aggregated = df.groupby(group_cols, dropna=False).size().reset_index(name="count")
        code = (
            f"plot_df = {data_name}.groupby({py_literal(group_cols)}, dropna=False)"
            ".size().reset_index(name='count')"
        )
        return aggregated, "count", code

    if aggregation in ("sum", "mean", "median") and value_col not in dfu.numeric_columns(df):
        raise ChartError(
            f"'{aggregation}' yalnızca sayısal sütunlarla çalışır. "
            f"'{value_col}' sayısal değil."
        )
    aggregated = (
        df.groupby(group_cols, dropna=False)[value_col].agg(aggregation).reset_index()
    )
    code = (
        f"plot_df = ({data_name}.groupby({py_literal(group_cols)}, dropna=False)"
        f"[{py_literal(value_col)}].agg({py_literal(aggregation)}).reset_index())"
    )
    return aggregated, value_col, code


def _sort_and_limit(
    data: pd.DataFrame,
    category_col: str,
    value_col: str,
    sort: str,
    top_n: int | None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Kategorik ekseni sıralar ve Top N uygular.

    Returns:
        ``(data, kod_satırları, uyarılar)``
    """
    warnings: list[str] = []
    code_lines: list[str] = []

    if top_n and len(data) > int(top_n):
        data = data.nlargest(int(top_n), value_col)
        code_lines.append(f"plot_df = plot_df.nlargest({int(top_n)}, {py_literal(value_col)})")
    elif len(data) > MAX_CATEGORIES:
        data = data.nlargest(MAX_CATEGORIES, value_col)
        warnings.append(
            f"Kategori sayısı çok yüksek olduğu için yalnızca en büyük "
            f"{MAX_CATEGORIES} kategori gösteriliyor."
        )
        code_lines.append(
            f"plot_df = plot_df.nlargest({MAX_CATEGORIES}, {py_literal(value_col)})"
        )

    if sort == "value_desc":
        data = data.sort_values(value_col, ascending=False)
        code_lines.append(f"plot_df = plot_df.sort_values({py_literal(value_col)}, ascending=False)")
    elif sort == "value_asc":
        data = data.sort_values(value_col, ascending=True)
        code_lines.append(f"plot_df = plot_df.sort_values({py_literal(value_col)})")
    elif sort == "category":
        data = data.sort_values(category_col)
        code_lines.append(f"plot_df = plot_df.sort_values({py_literal(category_col)})")

    return data, code_lines, warnings


def _px_call(func: str, data_name: str, kwargs: dict[str, Any]) -> str:
    """``px.<func>(...)`` çağrısının kodunu üretir."""
    parts = [data_name]
    parts += [f"{key}={py_literal(value)}" for key, value in kwargs.items() if value is not None]
    return f"fig = px.{func}(" + ", ".join(parts) + ")\nfig.show()"


def _layout_code(params: dict[str, Any]) -> list[str]:
    """Başlık/etiket/legend ayarlarının kodunu üretir."""
    lines: list[str] = []
    updates: dict[str, Any] = {}
    if params.get("x_label"):
        updates["xaxis_title"] = str(params["x_label"])
    if params.get("y_label"):
        updates["yaxis_title"] = str(params["y_label"])
    if params.get("show_legend") is False:
        updates["showlegend"] = False
    if updates:
        args = ", ".join(f"{key}={py_literal(value)}" for key, value in updates.items())
        lines.append(f"fig.update_layout({args})")
    return lines


def _apply_customization(figure: go.Figure, params: dict[str, Any]) -> None:
    """Kullanıcının seçtiği eksen/legend ayarlarını uygular."""
    updates: dict[str, Any] = {}
    if params.get("x_label"):
        updates["xaxis_title"] = str(params["x_label"])
    if params.get("y_label"):
        updates["yaxis_title"] = str(params["y_label"])
    if params.get("show_legend") is False:
        updates["showlegend"] = False
    if updates:
        figure.update_layout(**updates)


# ---------------------------------------------------------------------------
# Grafik üreticileri
# ---------------------------------------------------------------------------


def _build_bar(
    df: pd.DataFrame, params: dict[str, Any], data_name: str = "df"
) -> tuple[go.Figure, str, str, list[str]]:
    """Bar chart üretir (gruplu / yığılmış / yatay destekli)."""
    x = _check_column(df, _require(params, "x", "X ekseni (kategori)"), "X ekseni")
    y = params.get("y")
    color = params.get("color")
    aggregation = str(params.get("aggregation") or "count")
    bar_mode = str(params.get("barmode") or "group")
    horizontal = bool(params.get("horizontal"))
    sort = str(params.get("sort") or "value_desc")
    if y:
        y = _check_column(df, y, "Y ekseni")
    if color:
        color = _check_column(df, color, "Renk")
    if bar_mode not in BAR_MODES:
        raise ChartError(f"Desteklenmeyen çubuk modu: {bar_mode}")

    group_cols = [x] + ([color] if color and color != x else [])
    data, value_col, agg_code = _aggregate(df, group_cols, y, aggregation, data_name)

    code_lines = [agg_code]
    if not color:
        data, sort_code, warnings = _sort_and_limit(
            data, x, value_col, sort, params.get("top_n")
        )
        code_lines.extend(sort_code)
    else:
        warnings = []
        if dfu.safe_nunique(data[x]) > MAX_CATEGORIES:
            keep = (
                data.groupby(x, dropna=False)[value_col]
                .sum()
                .nlargest(MAX_CATEGORIES)
                .index
            )
            data = data[data[x].isin(keep)]
            warnings.append(
                f"Kategori sayısı çok yüksek; en büyük {MAX_CATEGORIES} kategori gösteriliyor."
            )
            code_lines.append(
                f"keep = (plot_df.groupby({py_literal(x)})[{py_literal(value_col)}]"
                f".sum().nlargest({MAX_CATEGORIES}).index)"
            )
            code_lines.append(f"plot_df = plot_df[plot_df[{py_literal(x)}].isin(keep)]")

    label = AGGREGATIONS.get(aggregation, aggregation).split(" ")[0]
    title = _title(params, f"{x} bazında {value_col} ({label})")

    axis_x, axis_y = (value_col, x) if horizontal else (x, value_col)
    figure = px.bar(
        data,
        x=axis_x,
        y=axis_y,
        color=color,
        orientation="h" if horizontal else "v",
        barmode=bar_mode if color else "relative",
        title=title,
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    code_lines.append(
        _px_call(
            "bar",
            "plot_df",
            {
                "x": axis_x,
                "y": axis_y,
                "color": color,
                "orientation": "h" if horizontal else None,
                "barmode": bar_mode if color else None,
                "title": title,
            },
        )
    )
    return figure, title, "\n".join(code_lines), warnings


def _build_line(
    df: pd.DataFrame, params: dict[str, Any], data_name: str = "df"
) -> tuple[go.Figure, str, str, list[str]]:
    """Line chart üretir."""
    x = _check_column(df, _require(params, "x", "X ekseni"), "X ekseni")
    y = _check_numeric(df, _require(params, "y", "Y ekseni"), "Y ekseni")
    color = params.get("color")
    if color:
        color = _check_column(df, color, "Renk")

    sort_cols = [x] if not color else [color, x]
    data = df.sort_values(sort_cols)
    data, warnings = _sample_for_plot(data, MAX_SCATTER_POINTS)
    title = _title(params, f"{x} - {y} değişimi")
    figure = px.line(
        data,
        x=x,
        y=y,
        color=color,
        markers=bool(params.get("markers")),
        title=title,
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    code = f"plot_df = {data_name}.sort_values({py_literal(sort_cols)})\n" + _px_call(
        "line",
        "plot_df",
        {
            "x": x,
            "y": y,
            "color": color,
            "markers": True if params.get("markers") else None,
            "title": title,
        },
    )
    return figure, title, code, warnings


def _build_area(
    df: pd.DataFrame, params: dict[str, Any], data_name: str = "df"
) -> tuple[go.Figure, str, str, list[str]]:
    """Area chart üretir."""
    x = _check_column(df, _require(params, "x", "X ekseni"), "X ekseni")
    y = _check_numeric(df, _require(params, "y", "Y ekseni"), "Y ekseni")
    color = params.get("color")
    if color:
        color = _check_column(df, color, "Renk")

    sort_cols = [x] if not color else [color, x]
    data = df.sort_values(sort_cols)
    data, warnings = _sample_for_plot(data, MAX_SCATTER_POINTS)
    title = _title(params, f"{x} - {y} alan grafiği")
    figure = px.area(
        data,
        x=x,
        y=y,
        color=color,
        title=title,
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    code = f"plot_df = {data_name}.sort_values({py_literal(sort_cols)})\n" + _px_call(
        "area", "plot_df", {"x": x, "y": y, "color": color, "title": title}
    )
    return figure, title, code, warnings


def _build_scatter(
    df: pd.DataFrame, params: dict[str, Any], data_name: str = "df"
) -> tuple[go.Figure, str, str, list[str]]:
    """Scatter plot üretir."""
    x = _check_numeric(df, _require(params, "x", "X ekseni"), "X ekseni")
    y = _check_numeric(df, _require(params, "y", "Y ekseni"), "Y ekseni")
    color = params.get("color")
    size = params.get("size")
    if color:
        color = _check_column(df, color, "Renk")
    if size:
        size = _check_numeric(df, size, "Nokta boyutu")

    data = df
    if size:
        # Negatif/boş boyut değerleri Plotly'de hata verir.
        data = data[data[size].notna() & (data[size] >= 0)]
        if data.empty:
            raise ChartError(
                f"'{size}' sütununda grafik için kullanılabilir (negatif olmayan) "
                "değer bulunamadı."
            )
    data, warnings = _sample_for_plot(data, MAX_SCATTER_POINTS)
    title = _title(params, f"{x} - {y} dağılımı")
    figure = px.scatter(
        data,
        x=x,
        y=y,
        color=color,
        size=size,
        title=title,
        opacity=0.75,
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    code = _px_call(
        "scatter",
        data_name,
        {"x": x, "y": y, "color": color, "size": size, "opacity": 0.75, "title": title},
    )
    return figure, title, code, warnings


def _build_bubble(
    df: pd.DataFrame, params: dict[str, Any], data_name: str = "df"
) -> tuple[go.Figure, str, str, list[str]]:
    """Bubble chart üretir (boyut sütunu zorunlu)."""
    params = dict(params)
    _require(params, "size", "Nokta boyutu için bir sayısal sütun")
    figure, title, code, warnings = _build_scatter(df, params, data_name)
    return figure, _title(params, title), code, warnings


def _build_histogram(
    df: pd.DataFrame, params: dict[str, Any], data_name: str = "df"
) -> tuple[go.Figure, str, str, list[str]]:
    """Histogram üretir."""
    column = _check_numeric(df, _require(params, "column", "Sütun"), "Sütun")
    bins = int(params.get("bins") or 30)
    if bins < 2:
        raise ChartError("Bin sayısı en az 2 olmalıdır.")
    color = params.get("color")
    if color:
        color = _check_column(df, color, "Renk")
    title = _title(params, f"{column} dağılımı (histogram)")
    figure = px.histogram(
        df,
        x=column,
        nbins=bins,
        color=color,
        title=title,
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    code = _px_call(
        "histogram", data_name, {"x": column, "nbins": bins, "color": color, "title": title}
    )
    return figure, title, code, []


def _build_box(
    df: pd.DataFrame, params: dict[str, Any], data_name: str = "df"
) -> tuple[go.Figure, str, str, list[str]]:
    """Box plot üretir."""
    y = _check_numeric(df, _require(params, "y", "Y ekseni (sayısal)"), "Y ekseni")
    x = params.get("x")
    color = params.get("color")
    if x:
        x = _check_column(df, x, "X ekseni (kategori)")
    if color:
        color = _check_column(df, color, "Renk")
    title = _title(params, f"{y} kutu grafiği" + (f" ({x} bazında)" if x else ""))
    figure = px.box(
        df,
        x=x,
        y=y,
        color=color,
        points="outliers",
        title=title,
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    code = _px_call(
        "box", data_name, {"x": x, "y": y, "color": color, "points": "outliers", "title": title}
    )
    return figure, title, code, []


def _build_violin(
    df: pd.DataFrame, params: dict[str, Any], data_name: str = "df"
) -> tuple[go.Figure, str, str, list[str]]:
    """Violin plot üretir."""
    y = _check_numeric(df, _require(params, "y", "Y ekseni (sayısal)"), "Y ekseni")
    x = params.get("x")
    color = params.get("color")
    if x:
        x = _check_column(df, x, "X ekseni (kategori)")
    if color:
        color = _check_column(df, color, "Renk")
    title = _title(params, f"{y} dağılımı (violin)")
    figure = px.violin(
        df,
        x=x,
        y=y,
        color=color,
        box=True,
        points=False,
        title=title,
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    code = _px_call(
        "violin", data_name, {"x": x, "y": y, "color": color, "box": True, "title": title}
    )
    return figure, title, code, []


def _build_pie_like(
    df: pd.DataFrame, params: dict[str, Any], data_name: str, hole: float
) -> tuple[go.Figure, str, str, list[str]]:
    """Pie / donut grafiklerinin ortak üreticisi."""
    names = _check_column(df, _require(params, "names", "Kategori sütunu"), "Kategori sütunu")
    values = params.get("values")
    aggregation = str(params.get("aggregation") or "count")
    if values:
        values = _check_column(df, values, "Değer sütunu")

    data, value_col, agg_code = _aggregate(df, [names], values, aggregation, data_name)
    top_n = params.get("top_n") or 10
    data, sort_code, warnings = _sort_and_limit(data, names, value_col, "value_desc", top_n)
    title = _title(params, f"{names} dağılımı")
    figure = px.pie(
        data,
        names=names,
        values=value_col,
        title=title,
        hole=hole,
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    figure.update_traces(textposition="inside", textinfo="percent+label")

    code_lines = [agg_code, *sort_code]
    code_lines.append(
        _px_call(
            "pie",
            "plot_df",
            {
                "names": names,
                "values": value_col,
                "hole": hole if hole else None,
                "title": title,
            },
        )
    )
    return figure, title, "\n".join(code_lines), warnings


def _build_pie(
    df: pd.DataFrame, params: dict[str, Any], data_name: str = "df"
) -> tuple[go.Figure, str, str, list[str]]:
    """Pie chart üretir."""
    return _build_pie_like(df, params, data_name, hole=0.0)


def _build_donut(
    df: pd.DataFrame, params: dict[str, Any], data_name: str = "df"
) -> tuple[go.Figure, str, str, list[str]]:
    """Donut chart üretir."""
    return _build_pie_like(df, params, data_name, hole=0.45)


def _build_heatmap(
    df: pd.DataFrame, params: dict[str, Any], data_name: str = "df"
) -> tuple[go.Figure, str, str, list[str]]:
    """Korelasyon ısı haritası üretir."""
    columns = params.get("columns") or dfu.numeric_columns(df)
    method = str(params.get("method") or "pearson")
    if method not in CORRELATION_METHODS:
        raise ChartError(f"Desteklenmeyen korelasyon yöntemi: {method}")
    try:
        matrix = correlation_matrix(df, list(columns), method)
    except ValidationError as exc:
        raise ChartError(str(exc)) from exc

    title = _title(params, f"Korelasyon matrisi ({method})")
    figure = px.imshow(
        matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title=title,
        template=PLOTLY_TEMPLATE,
    )
    code = (
        f"corr_matrix = {data_name}[{py_literal(list(matrix.columns))}]"
        f".corr(method={py_literal(method)})\n"
        f"fig = px.imshow(corr_matrix, text_auto='.2f', color_continuous_scale='RdBu_r',\n"
        f"                zmin=-1, zmax=1, aspect='auto', title={py_literal(title)})\n"
        "fig.show()"
    )
    return figure, title, code, []


def _build_category_heatmap(
    df: pd.DataFrame, params: dict[str, Any], data_name: str = "df"
) -> tuple[go.Figure, str, str, list[str]]:
    """İki kategorik sütun için çapraz tablo ısı haritası üretir."""
    x = _check_column(df, _require(params, "x", "X ekseni (kategori)"), "X ekseni")
    y = _check_column(df, _require(params, "y", "Y ekseni (kategori)"), "Y ekseni")
    if x == y:
        raise ChartError("X ve Y için farklı sütunlar seçmelisiniz.")

    warnings: list[str] = []
    for column in (x, y):
        if dfu.safe_nunique(df[column]) > 30:
            warnings.append(
                f"'{column}' sütununda çok fazla kategori var; grafik okunması "
                "zor olabilir. Binning veya gruplama düşünün."
            )

    table = pd.crosstab(df[y].astype("string"), df[x].astype("string"))
    if table.empty:
        raise ChartError("Çapraz tablo boş; farklı sütunlar seçin.")
    title = _title(params, f"{y} / {x} kesişim adetleri")
    figure = px.imshow(
        table,
        text_auto=True,
        color_continuous_scale="Blues",
        aspect="auto",
        title=title,
        template=PLOTLY_TEMPLATE,
    )
    code = (
        f"table = pd.crosstab({data_name}[{py_literal(y)}], {data_name}[{py_literal(x)}])\n"
        f"fig = px.imshow(table, text_auto=True, color_continuous_scale='Blues',\n"
        f"                aspect='auto', title={py_literal(title)})\n"
        "fig.show()"
    )
    return figure, title, code, warnings


def _build_density_heatmap(
    df: pd.DataFrame, params: dict[str, Any], data_name: str = "df"
) -> tuple[go.Figure, str, str, list[str]]:
    """İki sayısal sütun için yoğunluk ısı haritası üretir."""
    x = _check_numeric(df, _require(params, "x", "X ekseni"), "X ekseni")
    y = _check_numeric(df, _require(params, "y", "Y ekseni"), "Y ekseni")
    bins = int(params.get("bins") or 30)
    title = _title(params, f"{x} - {y} yoğunluk haritası")
    figure = px.density_heatmap(
        df,
        x=x,
        y=y,
        nbinsx=bins,
        nbinsy=bins,
        color_continuous_scale="Blues",
        title=title,
        template=PLOTLY_TEMPLATE,
    )
    code = _px_call(
        "density_heatmap",
        data_name,
        {"x": x, "y": y, "nbinsx": bins, "nbinsy": bins, "title": title},
    )
    return figure, title, code, []


_BUILDERS = {
    "bar": _build_bar,
    "line": _build_line,
    "area": _build_area,
    "scatter": _build_scatter,
    "bubble": _build_bubble,
    "histogram": _build_histogram,
    "box": _build_box,
    "violin": _build_violin,
    "pie": _build_pie,
    "donut": _build_donut,
    "heatmap": _build_heatmap,
    "category_heatmap": _build_category_heatmap,
    "density_heatmap": _build_density_heatmap,
}


def build_chart(
    df: pd.DataFrame,
    chart_type: str,
    params: dict[str, Any],
    source_note: str | None = None,
    data_name: str = "df",
    code_prefix: str = "",
) -> ChartResult:
    """Grafiği oluşturur ve yeniden üretilebilir tanımını döndürür.

    Args:
        df: Grafiğin çizileceği DataFrame (filtrelenmiş olabilir).
        chart_type: ``CHART_TYPES`` içindeki bir anahtar.
        params: Grafik parametreleri (x, y, color, bins, title, sort ...).
        source_note: Grafiğin hangi veri üzerinde çizildiğini anlatan not.
        data_name: Üretilen kodda kaynak DataFrame için kullanılacak değişken adı.
        code_prefix: Üretilen kodun başına eklenecek hazırlık kodu (ör. group by).

    Returns:
        Figure, grafik tanımı (spec) ve uyarıları içeren ChartResult.

    Raises:
        ChartError: Parametreler eksik/uyumsuzsa veya veri uygun değilse.
    """
    builder = _BUILDERS.get(chart_type)
    if builder is None:
        raise ChartError(f"Desteklenmeyen grafik tipi: {chart_type}")
    if df is None or df.empty:
        raise ChartError("Grafik çizilecek veri yok. Filtreleri gevşetmeyi deneyin.")

    try:
        figure, title, code, warnings = builder(df, params or {}, data_name)
    except (ChartError, ValidationError):
        raise
    except Exception as exc:  # Plotly/pandas kaynaklı beklenmeyen hatalar
        raise ChartError(
            f"Grafik oluşturulamadı: {exc}. Sütun seçimlerinizi kontrol edin."
        ) from exc

    figure.update_layout(
        title={"text": title, "x": 0.01, "xanchor": "left", "font": {"size": 17}},
        margin={"l": 60, "r": 30, "t": 60, "b": 60},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    _apply_customization(figure, params or {})

    layout_lines = _layout_code(params or {})
    if layout_lines:
        code = code.replace("\nfig.show()", "\n" + "\n".join(layout_lines) + "\nfig.show()")
    if code_prefix:
        code = code_prefix.rstrip() + "\n" + code

    clean_params = {k: v for k, v in (params or {}).items() if v not in (None, "", [])}
    spec = {
        "id": new_id(),
        "type": chart_type,
        "label": CHART_TYPES[chart_type]["label"],
        "title": title,
        "description": f"{CHART_TYPES[chart_type]['label']} - {title}",
        "params": clean_params,
        "python_code": code,
        "source_note": source_note or "working_df",
    }
    return ChartResult(figure=figure, spec=spec, warnings=warnings)


# ---------------------------------------------------------------------------
# Akıllı grafik önerileri ve varsayılanlar
# ---------------------------------------------------------------------------


def default_params(df: pd.DataFrame, chart_type: str) -> dict[str, Any]:
    """Grafik tipi için makul varsayılan parametreler üretir (smart defaults).

    Kullanıcı grafik tipini seçtiğinde formun hazır gelmesini sağlar.
    """
    numeric = dfu.numeric_columns(df)
    categorical = dfu.categorical_columns(df)
    datetimes = dfu.datetime_columns(df)
    low_cardinality = [
        column for column in categorical if 2 <= dfu.safe_nunique(df[column]) <= 20
    ]

    first_numeric = numeric[0] if numeric else None
    second_numeric = numeric[1] if len(numeric) > 1 else first_numeric
    first_category = (low_cardinality or categorical or [None])[0]
    first_date = datetimes[0] if datetimes else None

    presets: dict[str, dict[str, Any]] = {
        "bar": {"x": first_category, "aggregation": "count", "sort": "value_desc"},
        "line": {"x": first_date or first_numeric, "y": first_numeric},
        "area": {"x": first_date or first_numeric, "y": first_numeric},
        "scatter": {"x": first_numeric, "y": second_numeric},
        "bubble": {"x": first_numeric, "y": second_numeric, "size": first_numeric},
        "histogram": {"column": first_numeric, "bins": 30},
        "box": {"y": first_numeric, "x": first_category},
        "violin": {"y": first_numeric, "x": first_category},
        "pie": {"names": first_category, "aggregation": "count", "top_n": 8},
        "donut": {"names": first_category, "aggregation": "count", "top_n": 8},
        "heatmap": {"columns": numeric[: min(8, len(numeric))], "method": "pearson"},
        "category_heatmap": {
            "x": first_category,
            "y": low_cardinality[1] if len(low_cardinality) > 1 else None,
        },
        "density_heatmap": {"x": first_numeric, "y": second_numeric, "bins": 30},
    }
    return {key: value for key, value in presets.get(chart_type, {}).items() if value is not None}


def suggest_charts(df: pd.DataFrame, limit: int = 6) -> list[dict[str, Any]]:
    """Sütun tiplerine göre kural tabanlı grafik önerileri üretir.

    LLM/API kullanılmaz; öneriler tamamen sütun tiplerine ve kardinaliteye
    bakan kurallardan gelir.

    Args:
        df: Kaynak DataFrame.
        limit: Döndürülecek maksimum öneri sayısı.

    Returns:
        ``{"chart_type", "params", "reason", "label"}`` sözlüklerinden liste.
    """
    if df is None or df.empty:
        return []

    numeric = dfu.numeric_columns(df)
    datetimes = dfu.datetime_columns(df)
    categorical = [
        column
        for column in dfu.categorical_columns(df)
        if 2 <= dfu.safe_nunique(df[column]) <= 20
    ]
    suggestions: list[dict[str, Any]] = []

    def add(chart_type: str, params: dict[str, Any], reason: str) -> None:
        suggestions.append(
            {
                "chart_type": chart_type,
                "label": CHART_TYPES[chart_type]["label"],
                "params": params,
                "reason": reason,
            }
        )

    if datetimes and numeric:
        add(
            "line",
            {"x": datetimes[0], "y": numeric[0]},
            f"'{datetimes[0]}' bir tarih sütunu: zaman içindeki değişimi "
            f"'{numeric[0]}' ile çizgi grafikte görebilirsiniz.",
        )

    if len(numeric) >= 2:
        if len(df) > 5_000:
            add(
                "density_heatmap",
                {"x": numeric[0], "y": numeric[1]},
                f"{len(df):,} satır var: noktalar üst üste bineceği için "
                "yoğunluk haritası daha okunaklı olur.",
            )
        add(
            "scatter",
            {"x": numeric[0], "y": numeric[1]},
            f"'{numeric[0]}' ve '{numeric[1]}' sayısal: aralarındaki ilişkiyi "
            "dağılım grafiği gösterir.",
        )

    if categorical and numeric:
        add(
            "bar",
            {
                "x": categorical[0],
                "y": numeric[0],
                "aggregation": "mean",
                "sort": "value_desc",
            },
            f"'{categorical[0]}' kategorik, '{numeric[0]}' sayısal: kategori "
            "ortalamalarını çubuk grafikle karşılaştırabilirsiniz.",
        )
        add(
            "box",
            {"x": categorical[0], "y": numeric[0]},
            f"'{numeric[0]}' değerinin '{categorical[0]}' gruplarına göre "
            "yayılımını kutu grafiği özetler.",
        )

    if numeric:
        add(
            "histogram",
            {"column": numeric[0], "bins": 30},
            f"'{numeric[0]}' sütununun dağılımını (çarpıklık, yoğunluk) "
            "histogram gösterir.",
        )

    if categorical:
        add(
            "pie",
            {"names": categorical[0], "aggregation": "count", "top_n": 8},
            f"'{categorical[0]}' az sayıda kategori içeriyor: payları pasta "
            "grafikte görebilirsiniz.",
        )

    if len(categorical) >= 2:
        add(
            "category_heatmap",
            {"x": categorical[0], "y": categorical[1]},
            f"'{categorical[0]}' ve '{categorical[1]}' kategorik: kesişim "
            "adetlerini ısı haritası gösterir.",
        )

    if not suggestions and numeric:
        add(
            "histogram",
            {"column": numeric[0]},
            "Veri setinde tek sayısal sütun var; dağılımına bakabilirsiniz.",
        )
    return suggestions[:limit]


def figure_to_html(figure: go.Figure, include_plotlyjs: str | bool = "cdn") -> str:
    """Plotly figürünü tek dosyalık interaktif HTML'e çevirir."""
    return figure.to_html(include_plotlyjs=include_plotlyjs, full_html=True)
