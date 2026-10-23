"""Veri Yükle sayfası: CSV / Excel / JSON / Parquet yükleme ve okuma ayarları.

Dataset yüklenmeden önce ekran bilinçli olarak sadedir: tek bir yükleme alanı,
kısa bir açıklama ve örnek veriyle hızlı başlangıç.
"""

from __future__ import annotations

import streamlit as st

from components.messages import (
    disclaimer,
    error_boundary,
    section_header,
    show_info,
    show_success,
    show_warning_list,
)
from components.tables import show_dataframe
from core import session
from core.loader import (
    CSV_ENCODINGS,
    EXTENSION_TYPES,
    detect_extension,
    excel_sheet_names,
    load_dataframe,
)
from utils.helpers import format_bytes, format_number
from utils.validators import LoaderError

__all__ = ["render"]

_SEPARATOR_CHOICES: dict[str, str | None] = {
    "Otomatik algıla": None,
    "Virgül ( , )": ",",
    "Noktalı virgül ( ; )": ";",
    "Tab": "\t",
    "Dikey çizgi ( | )": "|",
}

_SAMPLE_CSV = """musteri_id,ad,yas,sehir,segment,gelir,harcama,uyelik_tarihi,aktif
1,Ayse Yilmaz,34,Ankara,Bireysel,18500,4200,2021-03-14,True
2,Mehmet Kaya,41,Istanbul,Kurumsal,27500,9100,2020-11-02,True
3,Zeynep Demir,29,Izmir,Bireysel,,3100,2022-06-21,False
4,Ali Vural,52,Ankara,Bireysel,41000,15200,2019-01-09,True
5,Elif Sahin,23,Bursa,Bireysel,12500,2600,2023-02-17,True
6,Ali Vural,52,Ankara,Bireysel,41000,15200,2019-01-09,True
7,Canan Aydin,37,Istanbul,KOBI,33000,8700,2021-08-30,False
8,Burak Ozdemir,45,Antalya,Kurumsal,250000,61000,2018-05-12,True
9,Deniz Arslan,31,Izmir,Bireysel,21000,5400,2022-12-01,True
10,Fatma Celik,,Ankara,Bireysel,19500,,2023-07-04,True
11,Kemal Toprak,58,Bursa,KOBI,36000,11200,2020-02-19,True
12,Selin Ak,27,Istanbul,Bireysel,16800,3900,2023-09-23,False
"""


def _reset_uploader() -> None:
    """Yeni dosya yüklemek için session'ı temizler."""
    session.clear_dataset()


def _render_loaded_state() -> None:
    """Yüklü veri setinin özetini ve sonraki adımları gösterir."""
    df = session.get_working_df()
    original = session.get_original_df()
    meta = session.dataset_meta()

    show_success(
        f"**{meta.get('filename')}** yüklendi · {format_number(len(df))} satır × "
        f"{format_number(df.shape[1])} sütun"
    )
    show_warning_list(st.session_state.get("load_warnings") or [], "Okuma notları")

    columns = st.columns(4)
    columns[0].metric("Satır", format_number(len(df)))
    columns[1].metric("Sütun", format_number(df.shape[1]))
    columns[2].metric("Dosya tipi", str(meta.get("file_type", "-")).upper())
    columns[3].metric("Uygulanan işlem", format_number(len(session.operation_history())))

    section_header("Önizleme", "Yüklenen verinin ilk 10 satırı.")
    show_dataframe(df.head(10), hide_index=False)

    if original is not None and not original.equals(df):
        show_info(
            "Veri seti üzerinde işlemler uygulanmış durumda. Orijinal veri korunuyor; "
            "İşlem Geçmişi sayfasından geri alabilirsiniz."
        )

    st.divider()
    left, right = st.columns([3, 1])
    left.markdown(
        "**Sıradaki adım:** sol menüden _Genel Bakış_ ile veriyi tanıyın, ardından "
        "_Veri Kalitesi_ ve _Temizleme_ sayfalarına geçin."
    )
    right.button(
        "🗑️ Yeni dosya yükle",
        on_click=_reset_uploader,
        width="stretch",
        help="Mevcut veri setini, işlem geçmişini ve grafikleri temizler.",
    )


def _load_file(
    uploaded: object, separator: str | None, encoding: str | None, sheet: str | None
) -> None:
    """Yüklenen dosyayı okuyup session'a yazar."""
    data = uploaded.getvalue()
    with st.spinner("Dosya okunuyor..."):
        result = load_dataframe(
            data,
            uploaded.name,
            separator=separator,
            encoding=encoding,
            sheet_name=sheet,
        )
    session.set_dataset(result)
    st.rerun()


def _render_hero() -> None:
    """Dataset yüklenmeden önce gösterilen sade karşılama bloğu."""
    st.markdown(
        """
        <div class="eda-hero">
          <span class="eda-pill">No-Code EDA Studio</span>
          <h2>Verinizi kod yazmadan keşfedin.</h2>
          <p>CSV, Excel, JSON veya Parquet dosyanızı yükleyin; temizleyin,
          görselleştirin ve analizinizin Python kodunu indirin.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    columns = st.columns(3)
    for column, (icon, title, text) in zip(
        columns,
        [
            ("🔍", "Analiz et", "Kalite, istatistik, korelasyon ve aykırı değerler"),
            ("🧼", "Temizle", "Boş değer, tekrar, tip dönüşümü ve dönüşümler"),
            ("🐍", "Kodu al", "Python script veya Jupyter Notebook olarak indir"),
        ],
    ):
        with column:
            st.markdown(f"**{icon} {title}**")
            st.caption(text)


def render() -> None:
    """Veri Yükle sayfasını çizer."""
    if session.has_dataset():
        st.title("📁 Veri Yükle")
        _render_loaded_state()
        return

    _render_hero()
    st.divider()

    uploaded = st.file_uploader(
        "Dosya seçin",
        type=["csv", "txt", "tsv", "xlsx", "xls", "json", "jsonl", "parquet"],
        help="Desteklenen biçimler: CSV, TXT, TSV, XLSX, XLS, JSON, JSONL, Parquet.",
    )

    if uploaded is None:
        if st.button("🧪 Örnek veri setiyle başla", type="primary"):
            with error_boundary("Örnek veri yükleme"):
                result = load_dataframe(_SAMPLE_CSV.encode("utf-8"), "ornek_musteriler.csv")
                session.set_dataset(result)
                st.rerun()
        st.caption(
            "Örnek veri; boş değer, tekrarlı satır ve aykırı değer içerecek şekilde "
            "hazırlanmıştır. `sample_data/` klasöründe daha büyük örnekler de var."
        )
        disclaimer(
            "Dosyanız analiz için yalnızca uygulama oturumunda işlenir; harici bir "
            "AI servisine veya API'ye gönderilmez."
        )
        return

    st.caption(f"Seçilen dosya: **{uploaded.name}** · {format_bytes(uploaded.size)}")

    separator: str | None = None
    encoding: str | None = None
    sheet: str | None = None

    with error_boundary("Dosya tipi tespiti"):
        extension = detect_extension(uploaded.name)
        file_type = EXTENSION_TYPES[extension]

        if file_type in ("csv", "excel"):
            with st.expander("⚙️ Gelişmiş okuma ayarları (genelde gerekmez)"):
                if file_type == "csv":
                    separator_label = st.selectbox("Ayraç", list(_SEPARATOR_CHOICES))
                    separator = _SEPARATOR_CHOICES[separator_label]
                    encoding_label = st.selectbox(
                        "Encoding", ["Otomatik algıla", *CSV_ENCODINGS]
                    )
                    encoding = None if encoding_label == "Otomatik algıla" else encoding_label
                    st.caption(
                        "Otomatik algılama sırayla şu encoding'leri dener: "
                        + ", ".join(CSV_ENCODINGS)
                    )
                else:
                    try:
                        sheets = excel_sheet_names(uploaded.getvalue(), uploaded.name)
                    except LoaderError as exc:
                        sheets = []
                        st.warning(str(exc))
                    if len(sheets) > 1:
                        sheet = st.selectbox("Excel sayfası", sheets)
                    elif sheets:
                        sheet = sheets[0]
                        st.caption(f"Tek sayfa bulundu: **{sheet}**")
                    else:
                        st.caption("Sayfa listesi okunamadı; ilk sayfa kullanılacak.")
        elif file_type == "json":
            st.caption(
                "JSON: kayıt listesi, satır-satır JSON (JSONL) ve iç içe nesneler "
                "desteklenir; iç içe alanlar nokta ile düzleştirilir."
            )

        if st.button("📥 Dosyayı yükle", type="primary"):
            _load_file(uploaded, separator, encoding, sheet)
