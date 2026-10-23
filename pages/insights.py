"""Smart Insights sayfası: kural tabanlı, deterministik bulgular."""

from __future__ import annotations

from typing import Any

import streamlit as st

from components.messages import (
    disclaimer,
    error_boundary,
    require_dataset,
    section_header,
    show_info,
    show_success,
)
from components.metric_cards import quality_badges
from components.tables import download_csv_button, show_dataframe
from core import session
from core.insights import (
    CATEGORIES,
    INSIGHT_RULES,
    SEVERITIES,
    filter_insights,
    insight_counts,
    insights_to_dataframe,
)
from utils import caching
from utils.helpers import format_number, safe_filename

__all__ = ["render"]

_SEVERITY_STYLE = {
    "critical": ("🔴", "error"),
    "warning": ("🟠", "warning"),
    "info": ("🔵", "info"),
}


def _render_insight(item: dict[str, Any]) -> None:
    """Tek bir insight kartını gösterir."""
    icon, _ = _SEVERITY_STYLE.get(str(item.get("severity")), ("🔵", "info"))
    column = item.get("column")
    header = f"{icon} **{item.get('title')}**"
    if column:
        header += f" · `{column}`"
    with st.container(border=True):
        st.markdown(header)
        st.markdown(item.get("message", ""))
        details = []
        if item.get("metric") is not None:
            details.append(f"Metrik: `{item['metric']}`")
        if item.get("rule"):
            details.append(f"Kural: `{item['rule']}`")
        if item.get("category"):
            details.append(f"Kategori: {CATEGORIES.get(item['category'], item['category'])}")
        if details:
            st.caption(" · ".join(details))


def render() -> None:
    """Smart Insights sayfasını çizer."""
    st.title("💡 Smart Insights")
    if not require_dataset("Otomatik bulgular"):
        return

    st.caption(
        "Bu bulgular **kural tabanlıdır**: hiçbir API veya dil modeli kullanılmaz, "
        "sonuçlar deterministiktir ve her bulgunun hangi kuralla tetiklendiği gösterilir."
    )

    df = session.get_working_df()
    with error_boundary("Insight üretimi"):
        with st.spinner("Veri seti kurallara göre taranıyor..."):
            insights = caching.generate_insights(df)

        counts = insight_counts(insights)
        quality_badges(counts)

        if not insights:
            show_success("Tanımlı kurallara göre öne çıkan bir sorun bulunamadı.")
            return

        st.divider()
        filters_row = st.columns([1, 1])
        selected_severities = filters_row[0].multiselect(
            "Önem derecesi",
            list(SEVERITIES),
            default=list(SEVERITIES),
            format_func=lambda key: SEVERITIES[key],
            key="ins_severity",
        )
        available_categories = sorted({str(i.get("category")) for i in insights})
        selected_categories = filters_row[1].multiselect(
            "Kategori",
            available_categories,
            default=available_categories,
            format_func=lambda key: CATEGORIES.get(key, key),
            key="ins_category",
        )

        visible = filter_insights(insights, selected_severities, selected_categories)
        section_header(
            f"Bulgular ({format_number(len(visible))})",
            "Önem derecesine göre sıralanmıştır.",
            icon="📋",
        )
        if not visible:
            show_info("Seçili filtrelere uyan bulgu yok.")
        for item in visible:
            _render_insight(item)

        st.divider()
        with st.expander("📊 Tablo görünümü ve indirme"):
            table = insights_to_dataframe(visible)
            show_dataframe(table)
            download_csv_button(
                table,
                "⬇️ Bulguları CSV indir",
                f"{safe_filename(session.dataset_meta().get('filename'))}_bulgular.csv",
                key="dl_insights",
            )

        with st.expander("⚙️ Kullanılan kurallar ve eşikler"):
            st.markdown(
                f"""
                | Kural | Eşik |
                |---|---|
                | Düşük boş değer | < %{INSIGHT_RULES['missing_low']:.0f} |
                | Orta boş değer | %{INSIGHT_RULES['missing_low']:.0f} - %{INSIGHT_RULES['missing_medium']:.0f} |
                | Yüksek boş değer | > %{INSIGHT_RULES['missing_medium']:.0f} |
                | Sabit sütun | unique == 1 |
                | Olası ID | tekillik oranı >= {INSIGHT_RULES['id_ratio']:.2f} |
                | Kategori dengesizliği | en büyük kategori oranı >= {INSIGHT_RULES['imbalance_ratio']:.2f} |
                | Çarpıklık | abs(skewness) > {INSIGHT_RULES['skewness']:.0f} |
                | Yüksek korelasyon | abs(Pearson r) >= {INSIGHT_RULES['correlation']:.2f} |
                | Tekrarlı satır | duplicate > 0 |
                | Aykırı değer | IQR aykırı oranı >= %{INSIGHT_RULES['outlier_ratio']:.0f} |
                | Yüksek kardinalite | tekillik >= {INSIGHT_RULES['cardinality_ratio']:.2f} ve unique >= {INSIGHT_RULES['cardinality_min_unique']:.0f} |
                """
            )

        disclaimer(
            "Bulgular yalnızca güncel çalışma verisi (working_df) üzerinden üretilir; "
            "veri temizleme işlemlerinden sonra tekrar kontrol etmeniz önerilir."
        )
