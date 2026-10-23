"""No-Code EDA Studio — uygulama giriş noktası.

Kod yazmadan keşifsel veri analizi yapın; yaptığınız işlemlerin Python
karşılığını görün ve `.py` / `.ipynb` olarak indirin.

Çalıştırma:
    streamlit run app.py

Mimari notlar:
    * Sayfalar ``pages/`` altında birer ``render()`` fonksiyonu olarak durur ve
      ``st.navigation`` ile callable page olarak bağlanır.
    * Veriye erişim yalnızca ``core.session`` üzerinden yapılır:
      ``original_df`` korunur, ``working_df`` üzerinde çalışılır ve her işlem
      ``operation_history`` içine Python koduyla birlikte yazılır.
    * Arayüz iki moda sahiptir: **Temel** (günlük EDA akışı) ve **Gelişmiş**
      (dönüşümler, pivot, ML hazırlığı). Amaç; güçlü özellikleri saklamadan
      arayüzü sade tutmaktır.
"""

from __future__ import annotations

import streamlit as st

from components.metric_cards import dataset_header
from core import session
from pages import (
    cleaning,
    code_viewer,
    data_quality,
    export_page,
    history,
    insights,
    ml_prep,
    outliers_page,
    overview,
    pivot_page,
    relationships,
    statistics_page,
    transform,
    upload,
    visualization,
)
from utils.helpers import format_number

APP_TITLE = "No-Code EDA Studio"
APP_TAGLINE = "Kod yazmadan analiz yap. Arkasındaki Python kodunu öğren."

#: Gelişmiş modda görünen sayfaların url yolları.
ADVANCED_PAGES = {"donusumler", "pivot", "ml-hazirligi", "iliskiler"}

#: Navigasyon tanımı: (modül, başlık, ikon, url yolu, grup)
_PAGE_DEFS = [
    (upload, "Veri Yükle", ":material/upload_file:", "veri-yukle", "VERİ"),
    (overview, "Genel Bakış", ":material/table_chart:", "genel-bakis", "VERİ"),
    (data_quality, "Veri Kalitesi", ":material/verified:", "veri-kalitesi", "VERİ"),
    (cleaning, "Temizleme", ":material/cleaning_services:", "temizleme", "VERİ"),
    (transform, "Dönüşümler", ":material/transform:", "donusumler", "VERİ"),
    (statistics_page, "İstatistik", ":material/functions:", "istatistik", "ANALİZ"),
    (visualization, "Görselleştir", ":material/bar_chart:", "grafikler", "ANALİZ"),
    (relationships, "İlişkiler", ":material/hub:", "iliskiler", "ANALİZ"),
    (outliers_page, "Aykırı Değerler", ":material/scatter_plot:", "aykiri-degerler", "ANALİZ"),
    (insights, "İçgörüler", ":material/lightbulb:", "icgoruler", "ANALİZ"),
    (pivot_page, "Pivot & GroupBy", ":material/pivot_table_chart:", "pivot", "GELİŞMİŞ"),
    (ml_prep, "ML Hazırlığı", ":material/model_training:", "ml-hazirligi", "GELİŞMİŞ"),
    (history, "İşlem Geçmişi", ":material/history:", "islem-gecmisi", "PROJE"),
    (code_viewer, "Python Kodu", ":material/code:", "python-kodu", "PROJE"),
    (export_page, "Dışa Aktar", ":material/download:", "disa-aktar", "PROJE"),
]

_CUSTOM_CSS = """
<style>
    /* Sade, ferah ve tutarlı bir yerleşim */
    .block-container { padding-top: 2.1rem; padding-bottom: 3rem; max-width: 1320px; }
    h1 { font-size: 1.75rem !important; letter-spacing: -0.02em; }
    h2 { font-size: 1.25rem !important; }
    h3 { font-size: 1.05rem !important; }
    [data-testid="stMetricValue"] { font-size: 1.4rem; }
    [data-testid="stMetricLabel"] { color: #667085; }
    /* Metrik kartlarına hafif çerçeve */
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e6e8ee;
        border-radius: 12px;
        padding: 12px 16px;
    }
    div[data-testid="stExpander"] details {
        border-radius: 12px;
        border-color: #e6e8ee;
    }
    section[data-testid="stSidebar"] { border-right: 1px solid #e6e8ee; }
    section[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }
    /* Sekmeleri biraz daha ferah yap */
    button[data-baseweb="tab"] { padding-top: 10px; padding-bottom: 10px; }
    /* Tablolarda yatay taşmayı yönet */
    [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
    .eda-hero {
        border: 1px solid #e6e8ee; background: #ffffff; border-radius: 16px;
        padding: 34px 32px; margin-bottom: 18px;
    }
    .eda-hero h2 { margin: 0 0 6px; font-size: 1.5rem !important; }
    .eda-hero p { color: #667085; margin: 0; }
    .eda-pill {
        display: inline-block; padding: 3px 11px; border-radius: 999px;
        background: #eef2ff; color: #2563eb; font-size: 0.78rem; font-weight: 600;
    }
</style>
"""


def _configure_page() -> None:
    """Sayfa başlığı, ikon ve genel görünüm ayarları."""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "about": f"**{APP_TITLE}**\n\n{APP_TAGLINE}\n\n"
            "Verileriniz yalnızca oturum belleğinde işlenir; harici bir AI "
            "servisine veya API'ye gönderilmez."
        },
    )
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


def _build_pages() -> dict[str, list[st.Page]]:
    """Gruplanmış navigasyon tanımını üretir.

    Temel modda gelişmiş sayfalar listeden **kaldırılmaz**, yalnızca gizlenir
    (``visibility="hidden"``). Böylece kullanıcı gelişmiş bir sayfadayken mod
    değiştirse bile uygulama hata vermez.
    """
    advanced_mode = bool(st.session_state.get("ui_advanced_mode", False))
    groups: dict[str, list[st.Page]] = {}
    for index, (module, title, icon, url, group) in enumerate(_PAGE_DEFS):
        hidden = (not advanced_mode) and url in ADVANCED_PAGES
        page = st.Page(
            module.render,
            title=title,
            icon=icon,
            url_path=url,
            default=(index == 0),
            visibility="hidden" if hidden else "visible",
        )
        groups.setdefault(group, []).append(page)
    return {group: pages for group, pages in groups.items() if pages}


def _render_sidebar() -> None:
    """Kenar çubuğu: marka, mod seçimi, veri durumu ve gizlilik notu."""
    with st.sidebar:
        st.markdown(f"### 📊 {APP_TITLE}")
        st.caption(APP_TAGLINE)
        st.divider()

        if session.has_dataset():
            df = session.get_working_df()
            st.caption("YÜKLÜ VERİ")
            st.markdown(f"**{session.dataset_meta().get('filename')}**")
            st.caption(
                f"{format_number(len(df))} satır · {format_number(df.shape[1])} sütun · "
                f"{format_number(len(session.operation_history()))} işlem"
            )
            if session.filters_active():
                st.caption("🔎 Geçici filtre aktif")
        else:
            st.caption("Henüz veri yüklenmedi.")

        st.divider()
        st.toggle(
            "Gelişmiş mod",
            key="ui_advanced_mode",
            help="Dönüşümler, Pivot & GroupBy, İlişkiler ve ML Hazırlığı "
            "sayfalarını gösterir.",
        )
        st.caption(
            "🔒 Dosyanız yalnızca bu oturumun belleğinde işlenir; harici bir "
            "AI servisine gönderilmez."
        )


def main() -> None:
    """Uygulamayı başlatır."""
    _configure_page()
    session.init_session_state()

    navigation = st.navigation(_build_pages(), position="sidebar")
    _render_sidebar()

    # Veri yüklüyse her sayfanın üstünde sabit bir dataset özeti göster.
    if session.has_dataset() and navigation.url_path != "veri-yukle":
        dataset_header()

    navigation.run()


if __name__ == "__main__":
    main()
