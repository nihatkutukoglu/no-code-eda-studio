"""Veri Temizleme sayfası.

Tüm dönüşümler yalnızca ``working_df`` üzerinde uygulanır, ``original_df``
korunur ve her başarılı işlem ``operation_history`` içine Python kodu ile
birlikte kaydedilir.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from components.filter_builder import render_filter_builder
from components.messages import (
    disclaimer,
    error_boundary,
    require_dataset,
    section_header,
    show_info,
    show_success,
    show_warning,
)
from components.metric_cards import metric_row
from components.tables import show_dataframe
from core import cleaner, session
from core.filters import filters_to_code
from utils import dataframe_utils as dfu
from utils.helpers import format_number, format_percent

__all__ = ["render"]


def _apply(action: str, params: dict[str, Any], success_hint: str = "") -> None:
    """Operasyonu uygular, sonucu kullanıcıya bildirir ve sayfayı yeniler."""
    with error_boundary("Veri temizleme"):
        operation = session.apply_and_record(action, params)
        st.session_state["last_operation_message"] = (
            operation.get("description", "İşlem uygulandı.") + (f" {success_hint}" if success_hint else "")
        )
        st.rerun()


def _header_metrics() -> None:
    """Sayfa üstünde güncel durum kartları."""
    df = session.get_working_df()
    original = session.get_original_df()
    history = session.operation_history()
    metric_row(
        [
            (
                "Satır",
                f"{len(df):,}",
                f"Orijinal: {len(original):,}" if original is not None else None,
            ),
            (
                "Sütun",
                f"{df.shape[1]:,}",
                f"Orijinal: {original.shape[1]:,}" if original is not None else None,
            ),
            ("Boş hücre", f"{dfu.total_missing(df):,}"),
            ("Tekrarlı satır", f"{dfu.duplicate_count(df):,}"),
            ("Uygulanan işlem", f"{len(history):,}"),
        ],
        columns=5,
    )
    message = st.session_state.pop("last_operation_message", None)
    if message:
        show_success(message)


def _missing_tab() -> None:
    """Boş değer temizleme sekmesi."""
    df = session.get_working_df()
    section_header(
        "Boş Değerler", "Boş değerleri silebilir veya doldurabilirsiniz.", icon="🕳️"
    )

    missing_counts = df.isna().sum()
    missing_columns = [str(c) for c in df.columns if missing_counts[c] > 0]
    if not missing_columns:
        show_success("Veri setinde boş değer yok. Bu bölümde yapılacak bir şey kalmadı.")
        return

    st.caption("Boş değer içeren sütunlar: " + ", ".join(missing_columns))
    mode = st.radio(
        "İşlem",
        ["Boş satırları sil", "Boş değerleri doldur"],
        horizontal=True,
        key="clean_missing_mode",
    )

    if mode == "Boş satırları sil":
        scope = st.radio(
            "Kapsam",
            ["Seçili sütunlar", "Tüm sütunlar"],
            horizontal=True,
            key="clean_missing_scope",
        )
        columns: list[str] = []
        if scope == "Seçili sütunlar":
            columns = st.multiselect(
                "Sütunlar", missing_columns, default=missing_columns[:1], key="clean_missing_cols"
            )
        how = "any"
        if scope == "Tüm sütunlar":
            how = (
                "any"
                if st.radio(
                    "Silme kuralı",
                    ["Herhangi bir sütun boşsa sil", "Tüm sütunlar boşsa sil"],
                    key="clean_missing_how",
                )
                == "Herhangi bir sütun boşsa sil"
                else "all"
            )

        affected = cleaner.missing_row_impact(df, columns, how)
        ratio = affected / len(df) * 100 if len(df) else 0
        st.info(
            f"Bu işlem **{format_number(affected)}** satırı silecek "
            f"({format_percent(ratio)}). Kalan satır: "
            f"{format_number(len(df) - affected)}.",
            icon="ℹ️",
        )
        if ratio >= 30:
            show_warning(
                "Veri setinin üçte birinden fazlasını siliyorsunuz. "
                "Doldurma (fill) yöntemlerini değerlendirmek isteyebilirsiniz."
            )
        if affected == 0:
            show_info("Seçiminize göre silinecek satır yok.")
        elif st.button("🗑️ Boş satırları sil", type="primary", key="btn_drop_missing"):
            _apply("drop_missing_rows", {"columns": columns, "how": how})
        return

    # Doldurma
    column = st.selectbox("Sütun", missing_columns, key="clean_fill_col")
    kind = dfu.column_kind(df[column])
    available = dict(cleaner.FILL_METHODS)
    if kind != "numeric":
        available.pop("mean", None)
        available.pop("median", None)
    method = st.selectbox(
        "Yöntem",
        list(available),
        format_func=lambda key: available[key],
        key="clean_fill_method",
    )
    value = None
    if method == "value":
        value = st.text_input(
            "Doldurulacak değer",
            key="clean_fill_value",
            placeholder="Örn. 0, Bilinmiyor, 2024-01-01",
        )

    affected = cleaner.fill_impact(df, column)
    preview_text = {
        "mean": "ortalama",
        "median": "medyan",
        "mode": "en sık değer",
        "value": "girdiğiniz değer",
        "ffill": "önceki satırın değeri",
        "bfill": "sonraki satırın değeri",
    }.get(method, method)
    st.info(
        f"**{format_number(affected)}** boş hücre {preview_text} ile doldurulacak.",
        icon="ℹ️",
    )
    if kind != "numeric" and method in ("mean", "median"):
        show_warning("Ortalama/medyan yalnızca sayısal sütunlarda kullanılabilir.")
    if st.button("🧴 Boş değerleri doldur", type="primary", key="btn_fill_missing"):
        _apply("fill_missing", {"column": column, "method": method, "value": value})


def _duplicates_tab() -> None:
    """Tekrarlı satır silme sekmesi."""
    df = session.get_working_df()
    section_header("Tekrarlı Satırlar", "Aynı kayıtların fazlalıklarını siler.", icon="👯")

    subset = st.multiselect(
        "Karşılaştırılacak sütunlar (boş bırakırsanız tüm sütunlar)",
        [str(c) for c in df.columns],
        key="clean_dup_subset",
    )
    keep = st.radio(
        "Hangi kayıt kalsın?",
        ["first", "last"],
        format_func=lambda value: "İlk kayıt" if value == "first" else "Son kayıt",
        horizontal=True,
        key="clean_dup_keep",
    )

    affected = cleaner.duplicate_impact(df, subset, keep)
    st.info(
        f"Bu işlem **{format_number(affected)}** satırı silecek. "
        f"Kalan satır: {format_number(len(df) - affected)}.",
        icon="ℹ️",
    )
    if affected == 0:
        show_success("Seçilen kapsamda tekrarlı satır yok.")
        return
    if subset:
        show_warning(
            "Yalnızca seçili sütunlara göre tekrar siliyorsunuz; diğer sütunlardaki "
            "farklı değerler kaybolabilir."
        )
    if st.button("🗑️ Tekrarları sil", type="primary", key="btn_drop_dup"):
        _apply("drop_duplicates", {"subset": subset, "keep": keep})


def _rename_section() -> None:
    """Sütun adı değiştirme bölümü."""
    df = session.get_working_df()
    column = st.selectbox("Sütun", [str(c) for c in df.columns], key="clean_rename_col")
    new_name = st.text_input("Yeni ad", value=str(column), key="clean_rename_new")
    if st.button("✏️ Adı değiştir", key="btn_rename"):
        _apply("rename_column", {"column": column, "new_name": new_name})


def _drop_section() -> None:
    """Sütun silme bölümü."""
    df = session.get_working_df()
    candidates = [str(c) for c in df.columns]
    constants = set(dfu.constant_columns(df))
    empties = set(dfu.empty_columns(df))
    if constants or empties:
        show_info(
            "Bilgi taşımayan sütunlar: "
            + ", ".join(sorted(constants | empties))
            + " (sabit veya tamamen boş)"
        )
    columns = st.multiselect("Silinecek sütunlar", candidates, key="clean_drop_cols")
    if columns:
        st.info(
            f"**{len(columns)}** sütun silinecek. Kalan sütun: "
            f"{len(candidates) - len(columns)}.",
            icon="ℹ️",
        )
    if st.button("🗑️ Sütunları sil", type="primary", key="btn_drop_cols", disabled=not columns):
        _apply("drop_columns", {"columns": columns})


def _convert_section() -> None:
    """Veri tipi dönüştürme bölümü."""
    df = session.get_working_df()
    column = st.selectbox("Sütun", [str(c) for c in df.columns], key="clean_convert_col")
    st.caption(f"Mevcut tip: `{df[column].dtype}` · {dfu.describe_dtype(df[column])}")
    target = st.selectbox(
        "Hedef tip",
        list(cleaner.TYPE_TARGETS),
        format_func=lambda key: cleaner.TYPE_TARGETS[key],
        key="clean_convert_target",
    )
    datetime_format = None
    if target == "datetime":
        datetime_format = st.text_input(
            "Tarih biçimi (opsiyonel)",
            key="clean_convert_fmt",
            placeholder="Örn. %d/%m/%Y — boş bırakırsanız otomatik denenir",
        ) or None

    preview = cleaner.conversion_preview(df, column, target, datetime_format)
    if preview.get("error"):
        show_warning(preview["error"])
    else:
        left, right = st.columns(2)
        left.metric("Dönüşecek değer", format_number(preview["ok"]))
        right.metric(
            "Kaybolacak değer",
            format_number(preview["lost"]),
            help="Dönüştürülemeyen değerler boş (NaN) olur.",
        )
        if preview["lost"]:
            show_warning(
                "Dönüştürülemeyen örnek değerler: "
                + ", ".join(f"`{v}`" for v in preview["examples"])
            )
    if st.button("🔁 Tipi dönüştür", type="primary", key="btn_convert"):
        _apply(
            "convert_dtype",
            {"column": column, "target": target, "datetime_format": datetime_format},
        )


def _columns_tab() -> None:
    """Sütun işlemleri sekmesi."""
    section_header(
        "Sütun İşlemleri", "Sütun adını değiştirin, sütun silin veya tipini dönüştürün.", icon="🧱"
    )
    inner = st.tabs(["✏️ Yeniden adlandır", "🗑️ Sütun sil", "🔁 Veri tipi dönüştür"])
    with inner[0], error_boundary("Sütun adı değiştirme"):
        _rename_section()
    with inner[1], error_boundary("Sütun silme"):
        _drop_section()
    with inner[2], error_boundary("Tip dönüştürme"):
        _convert_section()


def _outlier_tab() -> None:
    """Aykırı değer silme kısayolu sekmesi."""
    df = session.get_working_df()
    section_header(
        "Aykırı Değerler",
        "IQR yöntemiyle bulunan aykırı satırları kalıcı olarak silin.",
        icon="🎯",
    )
    numeric = dfu.numeric_columns(df)
    if not numeric:
        show_info("Veri setinde sayısal sütun yok.")
        return
    column = st.selectbox("Sütun", numeric, key="clean_outlier_col")
    factor = st.slider("IQR çarpanı (k)", 1.0, 3.0, 1.5, 0.1, key="clean_outlier_factor")
    affected = cleaner.outlier_impact(df, column, factor)
    st.info(
        f"**{format_number(affected)}** satır aykırı olarak işaretlendi "
        f"({format_percent(affected / len(df) * 100 if len(df) else 0)}).",
        icon="ℹ️",
    )
    disclaimer(
        "Bir değerin istatistiksel olarak aykırı olması hatalı olduğu anlamına gelmez. "
        "Detaylı inceleme için Aykırı Değer Analizi sayfasını kullanın."
    )
    if st.button(
        "🗑️ Aykırı satırları sil", type="primary", key="btn_drop_outliers", disabled=not affected
    ):
        _apply("drop_outlier_rows", {"column": column, "factor": float(factor)})


def _filter_tab() -> None:
    """Filtreyi kalıcı uygulama sekmesi."""
    df = session.get_working_df()
    section_header(
        "Filtre ile Satır Seçimi",
        "Filtreyi veri setine kalıcı olarak uygulayarak satırları kalıcı şekilde seçin.",
        icon="🔎",
    )
    filters = render_filter_builder(
        df,
        key_prefix="cleaning",
        title="Kalıcı filtre",
        description=(
            "Buradaki filtre **veri setini değiştirir**. Yalnızca görüntüleme için "
            "filtrelemek istiyorsanız Grafik Oluşturucu sayfasını kullanın."
        ),
    )
    if not filters:
        return

    with st.expander("Üretilecek Python kodu"):
        st.code(filters_to_code(filters), language="python")
    show_warning(
        "Bu işlem filtreye uymayan satırları siler. Geri almak için İşlem Geçmişi "
        "sayfasındaki 'Son İşlemi Geri Al' butonunu kullanabilirsiniz."
    )
    if st.button("✅ Filtreyi veri setine uygula", type="primary", key="btn_apply_filter"):
        _apply("filter_rows", {"filters": filters})
        st.session_state["cleaning_filters"] = []


def render() -> None:
    """Veri Temizleme sayfasını çizer."""
    st.title("🧹 Veri Temizleme")
    if not require_dataset("Veri temizleme"):
        return

    st.caption(
        "Her işlem yalnızca çalışma kopyasına (working_df) uygulanır; orijinal veri "
        "korunur ve her adımın Python kodu kaydedilir."
    )
    _header_metrics()
    st.divider()

    tabs = st.tabs(
        [
            "🕳️ Boş Değerler",
            "👯 Tekrarlar",
            "🧱 Sütun İşlemleri",
            "🎯 Aykırı Değerler",
            "🔎 Filtre",
        ]
    )
    renderers = (_missing_tab, _duplicates_tab, _columns_tab, _outlier_tab, _filter_tab)
    for tab, renderer in zip(tabs, renderers):
        with tab, error_boundary("Veri temizleme"):
            renderer()

    st.divider()
    with st.expander("👀 Güncel veri önizlemesi (ilk 10 satır)"):
        show_dataframe(session.get_working_df().head(10), hide_index=False)
