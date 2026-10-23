"""Jupyter Notebook (.ipynb) üretimi.

``nbformat`` ile gerçek, geçerli bir v4 notebook üretilir; dosya Jupyter ve
Google Colab'da sorunsuz açılır. Kullanıcı verisi notebook içine gömülmez;
yalnızca yapılan işlemlerin kodu yazılır.

Notebook yapısı:
    ``# Exploratory Data Analysis`` → Imports → Data Loading → Dataset Overview
    → Data Cleaning / Transformations / Feature Engineering (işlemler
    **kronolojik sırada**, kategori değiştikçe yeni başlık) → Statistical
    Analysis → Relationships → Outlier Analysis → Visualizations →
    ML Preparation → Insights

Önemli: operasyonlar bölümlere dağıtılırken sıra korunur. Bölüm başlıkları
okunabilirlik içindir; kodun çalışma sırası uygulamadaki sırayla aynıdır.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from core.code_generator import (
    action_category,
    build_operation_code,
    generate_imports,
    generate_loading_section,
)
from utils.helpers import py_literal, safe_filename

__all__ = [
    "build_notebook",
    "notebook_to_bytes",
    "notebook_filename",
    "script_filename",
]

#: Operasyon kategorisi -> notebook bölüm başlığı.
_SECTION_TITLES = {
    "cleaning": "Data Cleaning",
    "transformation": "Transformations",
    "feature": "Feature Engineering",
    "arrangement": "Sorting & Column Order",
}


def notebook_filename(dataset_name: str | None) -> str:
    """``<dataset_name>_eda.ipynb`` biçiminde dosya adı üretir."""
    return f"{safe_filename(dataset_name or 'dataset')}_eda.ipynb"


def script_filename(dataset_name: str | None) -> str:
    """``<dataset_name>_eda.py`` biçiminde dosya adı üretir."""
    return f"{safe_filename(dataset_name or 'dataset')}_eda.py"


def _insight_markdown(insights: list[dict[str, Any]]) -> str:
    """Insight listesini markdown madde listesine çevirir."""
    if not insights:
        return "Kural tabanlı tarama sonucunda öne çıkan bir bulgu kaydedilmedi."
    severity_labels = {"critical": "KRİTİK", "warning": "UYARI", "info": "BİLGİ"}
    lines: list[str] = []
    for item in insights:
        severity = severity_labels.get(str(item.get("severity")), "BİLGİ")
        column = item.get("column")
        scope = f"`{column}` · " if column else ""
        lines.append(
            f"- **[{severity}]** {scope}{item.get('title', '')}: {item.get('message', '')}"
        )
    return "\n".join(lines)


def _operation_cells(history: list[dict[str, Any]]) -> list[Any]:
    """Operasyonları kronolojik sırada, bölüm başlıklarıyla hücrelere çevirir."""
    cells: list[Any] = []
    current_category: str | None = None
    seen_sections: set[str] = set()

    for index, operation in enumerate(history, start=1):
        category = action_category(operation.get("action", ""))
        if category != current_category:
            current_category = category
            title = _SECTION_TITLES.get(category, "Data Cleaning")
            suffix = " (devam)" if title in seen_sections else ""
            seen_sections.add(title)
            cells.append(new_markdown_cell(f"## {title}{suffix}"))
        description = operation.get("description") or operation.get("action", "")
        code = operation.get("python_code") or build_operation_code(
            operation.get("action", ""), operation.get("params", {}) or {}
        )
        cells.append(new_markdown_cell(f"**{index}.** {description}"))
        cells.append(new_code_cell(code))
    if history:
        cells.append(new_code_cell("df.shape"))
    return cells


def _overview_cells() -> list[Any]:
    """Dataset Overview bölümü hücreleri."""
    return [
        new_markdown_cell("## Dataset Overview"),
        new_code_cell(
            "print('Satır:', df.shape[0], '| Sütun:', df.shape[1])\n"
            "df.info()"
        ),
        new_code_cell(
            "# Sütun profili: tip, boş değer ve tekillik\n"
            "profile = pd.DataFrame({\n"
            "    'dtype': df.dtypes.astype(str),\n"
            "    'missing': df.isna().sum(),\n"
            "    'missing_%': (df.isna().mean() * 100).round(2),\n"
            "    'unique': df.nunique(dropna=True),\n"
            "})\n"
            "profile.sort_values('missing_%', ascending=False)"
        ),
    ]


def _statistics_cells() -> list[Any]:
    """Statistical Analysis bölümü hücreleri."""
    return [
        new_markdown_cell("## Statistical Analysis"),
        new_code_cell("# Sayısal sütun özetleri\ndf.describe().T"),
        new_code_cell(
            "# Kategorik sütun özetleri\n"
            "categorical_cols = df.select_dtypes(\n"
            "    include=['object', 'string', 'category', 'bool']\n"
            ").columns\n"
            "if len(categorical_cols) > 0:\n"
            "    display(df[categorical_cols].describe().T)"
        ),
    ]


def _relationship_cells(numeric_columns: list[str] | None) -> list[Any]:
    """Relationships bölümü hücreleri."""
    columns_literal = py_literal(list(numeric_columns or []))
    code = (
        f"numeric_cols = {columns_literal} or df.select_dtypes('number').columns.tolist()\n"
        "if len(numeric_cols) >= 2:\n"
        "    corr_matrix = df[numeric_cols].corr(method='pearson')\n"
        "    display(corr_matrix.round(3))\n"
        "    fig = px.imshow(corr_matrix, text_auto='.2f', zmin=-1, zmax=1,\n"
        "                    color_continuous_scale='RdBu_r', aspect='auto',\n"
        "                    title='Korelasyon Matrisi')\n"
        "    fig.show()\n"
        "else:\n"
        "    print('Korelasyon için en az 2 sayısal sütun gerekir.')"
    )
    return [
        new_markdown_cell(
            "## Relationships\n\n"
            "> Korelasyon nedensellik göstermez; yalnızca birlikte hareket etmeyi ölçer."
        ),
        new_code_cell(code),
    ]


def _outlier_cells(numeric_columns: list[str] | None) -> list[Any]:
    """Outlier Analysis bölümü hücreleri."""
    columns_literal = py_literal(list(numeric_columns or []))
    code = (
        "def iqr_outlier_summary(frame, columns=None, factor=1.5):\n"
        '    """IQR yöntemiyle sütun bazında aykırı değer özeti üretir."""\n'
        "    rows = []\n"
        "    for column in (columns or frame.select_dtypes('number').columns):\n"
        "        series = frame[column].dropna()\n"
        "        if series.empty:\n"
        "            continue\n"
        "        q1, q3 = series.quantile(0.25), series.quantile(0.75)\n"
        "        iqr = q3 - q1\n"
        "        lower, upper = q1 - factor * iqr, q3 + factor * iqr\n"
        "        mask = (series < lower) | (series > upper)\n"
        "        rows.append({\n"
        "            'column': column, 'Q1': q1, 'Q3': q3, 'IQR': iqr,\n"
        "            'lower_bound': lower, 'upper_bound': upper,\n"
        "            'outliers': int(mask.sum()),\n"
        "            'outlier_%': round(float(mask.mean()) * 100, 2),\n"
        "        })\n"
        "    return pd.DataFrame(rows)\n"
        "\n"
        f"iqr_outlier_summary(df, {columns_literal} or None)"
    )
    return [
        new_markdown_cell(
            "## Outlier Analysis\n\n"
            "> Bir değerin istatistiksel olarak aykırı olması hatalı olduğu "
            "anlamına gelmez."
        ),
        new_code_cell(code),
    ]


def build_notebook(
    meta: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
    charts: list[dict[str, Any]] | None = None,
    insights: list[dict[str, Any]] | None = None,
    include_statistics: bool = True,
    include_relationships: bool = True,
    include_outliers: bool = True,
    numeric_columns: list[str] | None = None,
    ml_code: str | None = None,
) -> nbformat.NotebookNode:
    """Analiz oturumundan bir Jupyter Notebook üretir.

    Args:
        meta: Dataset bilgisi (``filename``, ``file_type``, ``separator`` ...).
        history: Operasyon geçmişi (kronolojik).
        charts: Grafik tanımları.
        insights: Kural tabanlı insight listesi (Insights bölümü için).
        include_statistics: Statistical Analysis bölümünü ekle.
        include_relationships: Relationships bölümünü ekle.
        include_outliers: Outlier Analysis bölümünü ekle.
        numeric_columns: Analiz bölümlerinde kullanılacak sayısal sütunlar.
        ml_code: ML Preparation bölümüne eklenecek kod (opsiyonel).

    Returns:
        ``nbformat`` NotebookNode nesnesi.
    """
    history = history or []
    charts = charts or []
    cells: list[Any] = []

    dataset_name = meta.get("filename") or "dataset"
    cells.append(
        new_markdown_cell(
            "# Exploratory Data Analysis\n\n"
            f"- **Kaynak dosya:** `{dataset_name}`\n"
            f"- **Veri işleme adımı:** {len(history)}\n"
            f"- **Grafik sayısı:** {len(charts)}\n"
            f"- **Oluşturma tarihi:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            "> Bu notebook **No-Code EDA Studio** tarafından otomatik üretildi. "
            "Kodlar, uygulamada arayüzden yapılan işlemlerin Python/Pandas "
            "karşılığıdır. İşlemler uygulamadaki sırayla çalışır."
        )
    )

    cells.append(new_markdown_cell("## Imports"))
    cells.append(new_code_cell(generate_imports(include_charts=True)))

    cells.append(new_markdown_cell("## Data Loading"))
    cells.append(new_code_cell(generate_loading_section(meta) + "\n\ndf.head()"))

    cells.extend(_overview_cells())

    if history:
        cells.extend(_operation_cells(history))
    else:
        cells.append(new_markdown_cell("## Data Cleaning"))
        cells.append(
            new_markdown_cell("Bu oturumda veri temizleme işlemi uygulanmadı.")
        )

    if include_statistics:
        cells.extend(_statistics_cells())
    if include_relationships:
        cells.extend(_relationship_cells(numeric_columns))
    if include_outliers:
        cells.extend(_outlier_cells(numeric_columns))

    cells.append(new_markdown_cell("## Visualizations"))
    if charts:
        for index, chart in enumerate(charts, start=1):
            title = chart.get("title") or chart.get("label") or f"Grafik {index}"
            cells.append(new_markdown_cell(f"**Grafik {index}:** {title}"))
            cells.append(new_code_cell(chart.get("python_code", "")))
    else:
        cells.append(new_markdown_cell("Bu oturumda grafik oluşturulmadı."))

    cells.append(new_markdown_cell("## ML Preparation"))
    if ml_code:
        cells.append(
            new_markdown_cell(
                "Aşağıdaki kod; train/test ayrımı ve ön işleme pipeline'ını "
                "kurar. Model seçimi ve eğitimi size bırakılmıştır."
            )
        )
        cells.append(new_code_cell(ml_code))
    else:
        cells.append(
            new_markdown_cell(
                "ML hazırlığı yapılandırılmadı. Uygulamadaki **ML Hazırlığı** "
                "sayfasından hedef sütunu seçip bu bölümü üretebilirsiniz."
            )
        )

    cells.append(new_markdown_cell("## Insights"))
    cells.append(
        new_markdown_cell(
            "### Kural tabanlı veri kalitesi notları\n\n"
            + _insight_markdown(insights or [])
            + "\n\n---\n"
            "**Hatırlatmalar**\n\n"
            "- Korelasyon nedensellik göstermez.\n"
            "- Bir değerin istatistiksel olarak aykırı olması hatalı olduğu "
            "anlamına gelmez.\n"
            "- Veri kalite skoru bilimsel bir standart değildir."
        )
    )

    notebook = new_notebook(cells=cells)
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
        "eda_studio": {
            "source_file": str(dataset_name),
            "operations": len(history),
            "charts": len(charts),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
    }
    return notebook


def notebook_to_bytes(notebook: nbformat.NotebookNode) -> bytes:
    """Notebook'u indirilebilir UTF-8 bayt dizisine çevirir.

    Raises:
        nbformat.ValidationError: Notebook geçerli v4 biçiminde değilse.
    """
    nbformat.validate(notebook)
    return nbformat.writes(notebook, version=4).encode("utf-8")
