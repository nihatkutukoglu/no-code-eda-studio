"""Veri Kalitesi sayfası: analiz odaklı (veriyi değiştirmez) kontroller.

Arayüzü sade tutmak için analizler beş sekmede gruplanmıştır:
    1. Boş Değerler
    2. Tekrarlar
    3. Sütun Sağlığı (sabit, neredeyse sabit, olası ID, yüksek kardinalite)
    4. Tip ve Değer Sorunları (tip tutarsızlıkları, boşluk, inf, sütun adları)
    5. Kalite Skoru
"""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from components.messages import (
    disclaimer,
    error_boundary,
    require_dataset,
    section_header,
    show_info,
    show_success,
)
from components.metric_cards import metric_row, score_gauge
from components.tables import download_csv_button, show_dataframe, show_table_or_empty
from core import profiler, session
from core.chart_builder import COLOR_SEQUENCE, PLOTLY_TEMPLATE
from core.data_quality_score import SCORE_DISCLAIMER
from utils import caching, dataframe_utils as dfu
from utils.helpers import format_number, format_percent, safe_filename

__all__ = ["render"]


def _missing_tab() -> None:
    """Boş değer analizi sekmesi."""
    df = session.get_working_df()
    section_header("Boş Değerler", "Sütun bazında boş değer dağılımı.", icon="🕳️")

    with st.spinner("Boş değerler analiz ediliyor..."):
        table = profiler.missing_value_table(df)
    missing_only = table[table["Boş Değer"] > 0]

    metric_row(
        [
            ("Toplam boş hücre", format_number(dfu.total_missing(df))),
            ("Boş değer içeren sütun", format_number(len(missing_only))),
            ("Tamamen boş sütun", format_number(len(dfu.empty_columns(df)))),
        ]
    )

    if missing_only.empty:
        show_success("Harika. Bu veri setinde eksik değer bulunamadı.")
        return

    figure = px.bar(
        missing_only.head(40),
        x="Sütun",
        y="Boş %",
        title="Sütunlara göre boş değer oranı (%)",
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=COLOR_SEQUENCE,
        text="Boş %",
    )
    figure.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    figure.update_layout(
        yaxis_range=[0, min(100, float(missing_only["Boş %"].max()) * 1.2 + 5)],
        margin={"t": 60, "b": 40},
    )
    st.plotly_chart(figure, width="stretch")
    show_dataframe(table, height=min(420, 60 + 35 * len(table)))
    download_csv_button(
        table,
        "⬇️ Tabloyu indir",
        f"{safe_filename(session.dataset_meta().get('filename'))}_bos_degerler.csv",
        key="dl_missing",
    )
    show_info("Düzeltmek için: **Temizleme → Boş Değerler** sekmesi.")


def _duplicates_tab() -> None:
    """Tekrarlı satır analizi sekmesi."""
    df = session.get_working_df()
    section_header("Tekrarlı Satırlar", "Tam olarak aynı olan satırlar.", icon="👯")

    with st.spinner("Tekrarlı satırlar aranıyor..."):
        report = profiler.duplicate_report(df)

    metric_row(
        [
            ("Silinebilir tekrar", format_number(report["count"])),
            ("Tekrar oranı", format_percent(report["ratio"])),
            ("Tekrara karışan satır", format_number(report["affected_rows"])),
            ("Tekil satır", format_number(report["unique_rows"])),
        ],
        columns=4,
    )

    if report["count"] == 0:
        show_success("Tekrarlanan satır bulunamadı.")
    else:
        st.caption("İlk tekrarlı kayıtlar (aynı değerler bir arada gösterilir):")
        show_dataframe(report["sample"], hide_index=False)
        show_info("Silmek için: **Temizleme → Tekrarlar** sekmesi.")

    with st.expander("Seçili sütunlara göre tekrar kontrolü"):
        subset = st.multiselect(
            "Sütunlar", [str(c) for c in df.columns], key="dq_dup_subset"
        )
        if subset:
            from core.cleaner import duplicate_impact

            count = duplicate_impact(df, subset, "first")
            st.metric("Bu sütunlara göre tekrar", format_number(count))
            st.caption(
                "Aynı müşteri/ürün farklı satırlarda yer alıyorsa bu kontrol işe yarar."
            )


def _column_health_tab() -> None:
    """Sabit, neredeyse sabit, ID ve yüksek kardinalite analizleri."""
    df = session.get_working_df()
    section_header(
        "Sütun Sağlığı",
        "Bilgi taşımayan, kimlik gibi davranan veya aşırı çeşitli sütunlar.",
        icon="🩻",
    )

    constants = profiler.constant_column_table(df)
    st.markdown("**Sabit sütunlar** — tek bir değer taşıyor, analiz için bilgi yok.")
    show_table_or_empty(constants, "Sabit sütun bulunamadı.", icon="✅")

    st.divider()
    threshold = st.slider(
        "Baskınlık eşiği", 0.80, 0.999, 0.95, 0.005, key="dq_near_threshold"
    )
    st.markdown(
        f"**Neredeyse sabit sütunlar** — tek bir değer satırların "
        f"%{threshold * 100:.1f}'inden fazlasını kaplıyor."
    )
    show_table_or_empty(
        profiler.near_constant_table(df, threshold),
        "Neredeyse sabit sütun bulunamadı.",
        icon="✅",
    )

    st.divider()
    id_threshold = st.slider(
        "Tekillik eşiği (olası ID)", 0.50, 1.00, float(dfu.DEFAULT_ID_RATIO), 0.01,
        key="dq_id_threshold",
    )
    st.markdown("**Olası kimlik (ID) sütunları** — değerlerin neredeyse tamamı tekil.")
    id_table = profiler.potential_id_table(df, id_threshold)
    show_table_or_empty(
        id_table,
        "Olası ID sütunu bulunamadı.",
        f"Hiçbir sütunun tekillik oranı %{id_threshold * 100:.0f} eşiğini aşmıyor.",
        icon="✅",
    )
    if not id_table.empty:
        disclaimer(
            "Bu bir tahmindir: yüksek tekillik, sütunun gerçekten kimlik sütunu "
            "olduğunu kanıtlamaz."
        )

    st.divider()
    st.markdown(
        "**Yüksek kardinaliteli kategorik sütunlar** — grafiklerde ve gruplamada "
        "okunabilirliği düşürür."
    )
    high_card = profiler.high_cardinality_table(df)
    show_table_or_empty(
        high_card, "Yüksek kardinaliteli sütun bulunamadı.", icon="✅"
    )
    if not high_card.empty:
        show_info(
            "Öneri: **Dönüşümler → Gruplama (Binning)** veya kategori sadeleştirme."
        )


def _type_issue_tab() -> None:
    """Tip tutarsızlıkları ve değer düzeyi sorunlar."""
    df = session.get_working_df()
    section_header(
        "Tip ve Değer Sorunları",
        "Basit kurallarla olası tip ve içerik problemlerini arar.",
        icon="🧬",
    )

    with st.spinner("Veri tipleri kontrol ediliyor..."):
        findings = caching.dtype_check_table(df)
    st.markdown("**Veri tipi uyarıları**")
    show_table_or_empty(
        findings,
        "Belirgin bir tip sorunu bulunamadı.",
        "Sütun tipleri içerikleriyle tutarlı görünüyor.",
        icon="✅",
    )
    if not findings.empty:
        disclaimer("Dönüşümler: **Temizleme → Sütun İşlemleri → Veri Tipi Dönüştür**.")

    st.divider()
    st.markdown("**Değer düzeyi sorunlar** — boşluk, yer tutucu metin, sonsuz değer.")
    problems = profiler.value_problem_table(df)
    show_table_or_empty(problems, "Değer düzeyinde sorun bulunamadı.", icon="✅")

    st.divider()
    st.markdown("**Sütun adı çakışmaları**")
    name_issues = profiler.duplicate_column_names(df)
    show_table_or_empty(
        name_issues, "Sütun adlarında çakışma bulunamadı.", icon="✅"
    )

    with st.expander("Mevcut sütun profili"):
        show_dataframe(
            caching.column_profile(df)[
                ["Sütun", "Veri Tipi", "Rol (tahmin)", "Boş", "Tekil"]
            ]
        )


def _score_tab() -> None:
    """Data Quality Score sekmesi."""
    df = session.get_working_df()
    section_header(
        "Veri Kalite Skoru",
        "Veri kalitesi sorunlarını tek bakışta görmek için basit, açıklanabilir bir skor.",
        icon="🩺",
    )
    with st.spinner("Skor hesaplanıyor..."):
        score = caching.quality_score(df)

    left, right = st.columns([1, 2])
    with left:
        st.plotly_chart(score_gauge(score.overall, score.label), width="stretch")
    with right:
        st.markdown("**Alt bileşenler ve puan etkileri**")
        show_dataframe(score.to_dataframe())
        worst = max(score.components, key=lambda component: component.lost_points)
        if worst.lost_points > 0:
            show_info(
                f"En çok puan kaybı **{worst.label}** bileşeninden geliyor "
                f"({worst.lost_points} puan). {worst.detail}"
            )

    st.caption(f"📌 {SCORE_DISCLAIMER}")
    with st.expander("Skor nasıl hesaplanıyor?"):
        st.markdown(
            "Her bileşen 0-100 arası bir alt skor üretir ve ağırlığıyla çarpılarak "
            "toplanır:\n\n"
            "- **Completeness (35)** = 100 - boş hücre yüzdesi\n"
            "- **Duplicate Quality (20)** = 100 - tekrarlı satır yüzdesi\n"
            "- **Type Consistency (15)** = tip tutarsızlığı bulunan sütun oranına göre ceza\n"
            "- **Low Information Columns (15)** = sabit/boş sütun oranına göre ceza\n"
            "- **Outlier Indicator (15)** = sayısal sütunların ortalama IQR aykırı oranı\n\n"
            "Skor yalnızca güncel çalışma verisi üzerinden hesaplanır."
        )


def render() -> None:
    """Veri Kalitesi sayfasını çizer."""
    st.title("🩺 Veri Kalitesi")
    if not require_dataset("Veri kalitesi analizi"):
        return

    st.caption(
        "Bu sayfa **veriyi değiştirmez**; yalnızca analiz üretir. "
        "Düzeltmeleri Temizleme ve Dönüşümler sayfalarından uygulayabilirsiniz."
    )

    tabs = st.tabs(
        [
            "🕳️ Boş Değerler",
            "👯 Tekrarlar",
            "🩻 Sütun Sağlığı",
            "🧬 Tip & Değer",
            "📈 Kalite Skoru",
        ]
    )
    renderers = (
        _missing_tab,
        _duplicates_tab,
        _column_health_tab,
        _type_issue_tab,
        _score_tab,
    )
    for tab, renderer in zip(tabs, renderers):
        with tab, error_boundary("Analiz"):
            renderer()
