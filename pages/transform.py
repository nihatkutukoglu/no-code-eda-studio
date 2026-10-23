"""Dönüşümler sayfası: metin temizleme, sayısal dönüşüm/ölçekleme,
kategorik encoding, tarih özellikleri, yeni sütun üretimi, binning ve düzen.

Tüm işlemler ``operation_history`` üzerinden uygulanır: geri alınabilir,
yeniden oynatılabilir ve Python koduna çevrilebilir.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from components.messages import (
    disclaimer,
    error_boundary,
    require_dataset,
    section_header,
    show_info,
    show_success,
    show_warning,
)
from components.tables import show_dataframe
from core import session, transformations as tf
from core.code_generator import build_operation_code
from utils import dataframe_utils as dfu
from utils.helpers import format_number

__all__ = ["render"]


def _apply(action: str, params: dict[str, Any]) -> None:
    """Dönüşümü uygular ve sayfayı yeniler."""
    with error_boundary("Dönüşüm"):
        operation = session.apply_and_record(action, params)
        st.session_state["transform_message"] = operation.get("description")
        st.rerun()


def _preview_and_apply(
    action: str,
    params: dict[str, Any],
    df: pd.DataFrame,
    button_label: str,
    key: str,
) -> None:
    """Dönüşümün ön izlemesini gösterir ve uygulama butonunu çizer.

    Ön izleme, işlemi **geçici bir kopya** üzerinde çalıştırır; veri seti
    yalnızca kullanıcı butona basınca değişir.
    """
    with st.expander("👁️ Ön izleme ve üretilecek kod", expanded=True):
        try:
            from core.operations import apply_operation

            preview_df, operation = apply_operation(df, action, params)
        except Exception as exc:  # kullanıcı dostu mesaj
            show_warning(str(exc))
            return

        source_column = params.get("column") or params.get("left")
        new_columns = [c for c in preview_df.columns if c not in df.columns]
        target = (
            new_columns[0]
            if new_columns
            else (params.get("new_column") or source_column)
        )
        if source_column in df.columns and target in preview_df.columns:
            comparison = tf.transform_preview(df, source_column, preview_df, target, rows=5)
            show_dataframe(comparison)
        elif new_columns:
            show_dataframe(preview_df[new_columns].head(5))
        st.caption(
            f"Sonuç: {format_number(len(preview_df))} satır × "
            f"{format_number(preview_df.shape[1])} sütun"
            + (f" · yeni sütun: {', '.join(new_columns)}" if new_columns else "")
        )
        st.code(build_operation_code(action, operation.params), language="python")

    if st.button(button_label, type="primary", key=key):
        _apply(action, params)


def _target_column_input(df: pd.DataFrame, source: str, key: str, suffix: str) -> str | None:
    """Sonucun yeni sütuna mı yazılacağını soran ortak kontrol.

    Not: Widget anahtarına sütun ve yöntem eki dahil edilir; böylece kullanıcı
    sütunu veya yöntemi değiştirdiğinde önerilen ad güncellenir (Streamlit
    widget değerleri anahtar bazında korunduğu için bu gerekli).
    """
    mode = st.radio(
        "Sonuç nereye yazılsın?",
        ["Yeni sütun oluştur", "Mevcut sütunu değiştir"],
        horizontal=True,
        key=f"{key}_mode",
    )
    if mode == "Mevcut sütunu değiştir":
        return None
    default = tf.suggest_feature_name(df, f"{source}_{suffix}")
    return st.text_input(
        "Yeni sütun adı", value=default, key=f"{key}_name_{source}_{suffix}"
    )


def _text_tab(df: pd.DataFrame) -> None:
    """Metin temizleme sekmesi."""
    section_header(
        "Metin Temizleme",
        "Boşluk kırpma, harf dönüşümü, bul/değiştir ve karakter silme.",
        icon="🔤",
    )
    candidates = [c for c in dfu.categorical_columns(df) if not dfu.column_kind(df[c]) == "boolean"]
    if not candidates:
        show_info("Veri setinde metin/kategorik sütun bulunamadı.")
        return

    column = st.selectbox("Sütun", candidates, key="tf_text_col")
    method = st.selectbox(
        "İşlem",
        list(tf.TEXT_METHODS),
        format_func=lambda key: tf.TEXT_METHODS[key],
        key="tf_text_method",
    )
    params: dict[str, Any] = {"column": column, "method": method}
    if method == "replace":
        columns = st.columns(2)
        params["find"] = columns[0].text_input("Aranacak metin", key="tf_text_find")
        params["replace_with"] = columns[1].text_input(
            "Yeni metin (boş = sil)", key="tf_text_replace"
        )
    elif method == "remove_chars":
        params["characters"] = st.text_input(
            "Silinecek karakterler",
            key="tf_text_chars",
            placeholder="Örn. .,-() — her karakter tek tek silinir",
        )
    params["new_column"] = _target_column_input(df, column, "tf_text", "temiz")

    _preview_and_apply("text_transform", params, df, "🔤 Metni dönüştür", "tf_text_apply")


def _numeric_tab(df: pd.DataFrame) -> None:
    """Sayısal dönüşüm ve ölçekleme sekmesi."""
    section_header(
        "Sayısal Dönüşüm ve Ölçekleme",
        "Log, karekök, yuvarlama ve Min-Max / Standart / Robust ölçekleme.",
        icon="🧮",
    )
    numeric = dfu.numeric_columns(df)
    if not numeric:
        show_info("Veri setinde sayısal sütun bulunamadı.")
        return

    columns = st.columns(2)
    column = columns[0].selectbox("Sütun", numeric, key="tf_num_col")
    method = columns[1].selectbox(
        "Dönüşüm",
        list(tf.NUMERIC_METHODS),
        format_func=lambda key: tf.NUMERIC_METHODS[key],
        key="tf_num_method",
    )
    params: dict[str, Any] = {"column": column, "method": method}
    if method == "round":
        params["decimals"] = st.number_input(
            "Ondalık basamak", min_value=0, max_value=8, value=2, key="tf_num_decimals"
        )

    series = pd.to_numeric(df[column], errors="coerce")
    if method in ("log", "sqrt") and (series <= 0).any():
        show_warning(
            f"'{column}' sütununda sıfır veya negatif değerler var; bunlar "
            "dönüşüm sonrası boş (NaN) olur. log1p alternatifini düşünün."
        )
    if method in ("minmax", "standard", "robust"):
        disclaimer(
            "Ölçekleme pandas formülüyle uygulanır (ek bağımlılık gerekmez). "
            "scikit-learn karşılığı ML Hazırlığı sayfasındaki pipeline kodunda üretilir."
        )

    suffix = {"minmax": "minmax", "standard": "z", "robust": "robust"}.get(method, method)
    params["new_column"] = _target_column_input(df, column, "tf_num", suffix)

    _preview_and_apply("numeric_transform", params, df, "🧮 Dönüşümü uygula", "tf_num_apply")


def _encoding_tab(df: pd.DataFrame) -> None:
    """Kategorik encoding sekmesi."""
    section_header(
        "Kategorik Encoding",
        "Kategorileri modellerin kullanabileceği sayısal temsile çevirin.",
        icon="🔢",
    )
    candidates = dfu.categorical_columns(df)
    if not candidates:
        show_info("Veri setinde kategorik sütun bulunamadı.")
        return

    columns = st.columns(2)
    column = columns[0].selectbox("Sütun", candidates, key="tf_enc_col")
    method = columns[1].selectbox(
        "Yöntem",
        list(tf.ENCODING_METHODS),
        format_func=lambda key: tf.ENCODING_METHODS[key],
        key="tf_enc_method",
    )

    nunique = dfu.safe_nunique(df[column])
    st.caption(f"'{column}' sütununda {format_number(nunique)} farklı değer var.")
    params: dict[str, Any] = {"column": column, "method": method}

    if method == "onehot":
        if nunique > 30:
            show_warning(
                f"{format_number(nunique)} kategori için One-Hot Encoding "
                f"{format_number(nunique)} yeni sütun üretir. Label Encoding veya "
                "önce kategori sadeleştirme önerilir."
            )
        params["drop_first"] = st.toggle(
            "İlk kategoriyi düşür (drop_first)",
            value=False,
            key="tf_enc_drop_first",
            help="Doğrusal modellerde kukla değişken tuzağını önler.",
        )
    elif method == "ordinal":
        values = [str(v) for v in pd.unique(df[column].dropna().astype(str))][:50]
        params["categories"] = st.multiselect(
            "Kategori sırası (küçükten büyüğe sürükleyerek seçin)",
            values,
            default=values,
            key="tf_enc_order",
        )
        if not params["categories"]:
            show_info("Sıralamayı belirlemek için en az bir kategori seçin.")
            return
    else:
        st.caption(
            "Label Encoding kategorileri alfabetik sıraya göre 0..n-1 arası "
            "sayılara çevirir; sıralama bilgisi taşımaz."
        )

    if method != "onehot":
        params["new_column"] = _target_column_input(df, column, "tf_enc", "kod")

    _preview_and_apply("encode_categorical", params, df, "🔢 Encoding uygula", "tf_enc_apply")


def _datetime_tab(df: pd.DataFrame) -> None:
    """Tarih özellikleri sekmesi."""
    section_header(
        "Tarih Özellikleri",
        "Tarih sütunundan yıl, ay, hafta günü gibi yeni sütunlar üretin.",
        icon="📅",
    )
    candidates = dfu.datetime_columns(df)
    if not candidates:
        show_info(
            "Tarih tipinde sütun bulunamadı. Metin olarak saklanan tarihleri "
            "Temizleme sayfasından `datetime` tipine dönüştürebilirsiniz."
        )
        return

    column = st.selectbox("Tarih sütunu", candidates, key="tf_dt_col")
    features = st.multiselect(
        "Üretilecek özellikler",
        list(tf.DATETIME_FEATURES),
        default=["year", "month", "weekday"],
        format_func=lambda key: tf.DATETIME_FEATURES[key],
        key="tf_dt_features",
    )
    if not features:
        show_info("En az bir özellik seçin.")
        return
    prefix = st.text_input("Sütun öneki", value=str(column), key="tf_dt_prefix")

    _preview_and_apply(
        "datetime_features",
        {"column": column, "features": features, "prefix": prefix},
        df,
        "📅 Özellikleri üret",
        "tf_dt_apply",
    )


def _feature_tab(df: pd.DataFrame) -> None:
    """Aritmetik feature üretimi sekmesi."""
    section_header(
        "Yeni Sütun (Feature Engineering)",
        "İki sayısal sütunu birleştirerek yeni bir sütun oluşturun.",
        icon="➕",
    )
    numeric = dfu.numeric_columns(df)
    if len(numeric) < 2:
        show_info("Bu işlem için en az 2 sayısal sütun gerekir.")
        return

    columns = st.columns(3)
    left = columns[0].selectbox("Birinci sütun (a)", numeric, key="tf_feat_left")
    operation = columns[1].selectbox(
        "İşlem",
        list(tf.ARITHMETIC_OPERATIONS),
        format_func=lambda key: tf.ARITHMETIC_OPERATIONS[key],
        key="tf_feat_op",
    )
    right = columns[2].selectbox(
        "İkinci sütun (b)",
        [c for c in numeric if c != left] or numeric,
        key="tf_feat_right",
    )
    default_name = tf.suggest_feature_name(df, f"{left}_{operation}_{right}")
    new_column = st.text_input("Yeni sütun adı", value=default_name, key="tf_feat_name")

    if operation in ("divide", "ratio"):
        disclaimer("Sıfıra bölme durumunda sonuç boş (NaN) olur; uygulama çökmez.")

    _preview_and_apply(
        "arithmetic_feature",
        {"left": left, "right": right, "operation": operation, "new_column": new_column},
        df,
        "➕ Sütunu oluştur",
        "tf_feat_apply",
    )


def _binning_tab(df: pd.DataFrame) -> None:
    """Binning sekmesi."""
    section_header(
        "Gruplama (Binning)",
        "Sayısal bir sütunu aralıklara ayırarak kategorik hale getirin.",
        icon="🪣",
    )
    numeric = dfu.numeric_columns(df)
    if not numeric:
        show_info("Veri setinde sayısal sütun bulunamadı.")
        return

    columns = st.columns(2)
    column = columns[0].selectbox("Sütun", numeric, key="tf_bin_col")
    method = columns[1].selectbox(
        "Yöntem",
        list(tf.BINNING_METHODS),
        format_func=lambda key: tf.BINNING_METHODS[key],
        key="tf_bin_method",
    )
    params: dict[str, Any] = {"column": column, "method": method}

    if method == "custom":
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        placeholder = (
            f"{series.min():.0f}, {series.median():.0f}, {series.max():.0f}"
            if not series.empty
            else "0, 18, 30, 60"
        )
        raw_edges = st.text_input(
            "Sınır değerleri (virgülle)", placeholder=placeholder, key="tf_bin_edges"
        )
        try:
            params["edges"] = [
                float(part.strip().replace(",", "."))
                for part in raw_edges.split(";")
                if part.strip()
            ] if ";" in raw_edges else [
                float(part.strip()) for part in raw_edges.split(",") if part.strip()
            ]
        except ValueError:
            show_warning("Sınır değerleri sayı olmalıdır (ör. 0, 18, 30, 60).")
            return
        if len(params.get("edges") or []) < 2:
            show_info("En az 2 sınır değeri girin.")
            return
    else:
        params["bins"] = st.slider("Aralık sayısı", 2, 20, 4, key="tf_bin_count")

    raw_labels = st.text_input(
        "Etiketler (opsiyonel, virgülle)",
        key="tf_bin_labels",
        placeholder="Örn. Düşük, Orta, Yüksek",
    )
    labels = [part.strip() for part in raw_labels.split(",") if part.strip()]
    if labels:
        params["labels"] = labels
    params["new_column"] = st.text_input(
        "Yeni sütun adı",
        value=tf.suggest_feature_name(df, f"{column}_grup"),
        key="tf_bin_name",
    )

    _preview_and_apply("bin_column", params, df, "🪣 Grupları oluştur", "tf_bin_apply")


def _layout_tab(df: pd.DataFrame) -> None:
    """Sıralama, sütun kopyalama ve sütun sırası sekmesi."""
    section_header(
        "Sıralama ve Sütun Düzeni",
        "Satırları sıralayın, sütun kopyalayın veya sütun sırasını değiştirin.",
        icon="🔀",
    )
    inner = st.tabs(["⬆️ Satır sıralama", "📑 Sütun kopyala", "↔️ Sütun sırası"])
    all_columns = [str(c) for c in df.columns]

    with inner[0], error_boundary("Sıralama"):
        sort_columns = st.multiselect("Sıralama sütunları", all_columns, key="tf_sort_cols")
        if sort_columns:
            ascending = []
            for column in sort_columns:
                ascending.append(
                    st.radio(
                        f"'{column}' yönü",
                        [True, False],
                        format_func=lambda value: "Artan (A→Z, 0→9)"
                        if value
                        else "Azalan (Z→A, 9→0)",
                        horizontal=True,
                        key=f"tf_sort_dir_{column}",
                    )
                )
            _preview_and_apply(
                "sort_rows",
                {"columns": sort_columns, "ascending": ascending},
                df,
                "⬆️ Satırları sırala",
                "tf_sort_apply",
            )
        else:
            show_info("Sıralamak için en az bir sütun seçin.")

    with inner[1], error_boundary("Sütun kopyalama"):
        column = st.selectbox("Kopyalanacak sütun", all_columns, key="tf_dup_col")
        new_name = st.text_input(
            "Yeni sütun adı",
            value=tf.suggest_feature_name(df, f"{column}_kopya"),
            key="tf_dup_name",
        )
        _preview_and_apply(
            "duplicate_column",
            {"column": column, "new_column": new_name},
            df,
            "📑 Sütunu kopyala",
            "tf_dup_apply",
        )

    with inner[2], error_boundary("Sütun sırası"):
        order = st.multiselect(
            "Öne alınacak sütunlar (seçim sırası korunur)",
            all_columns,
            key="tf_order_cols",
            help="Seçilmeyen sütunlar mevcut sıralarıyla sona eklenir.",
        )
        if order:
            _preview_and_apply(
                "reorder_columns",
                {"columns": order},
                df,
                "↔️ Sırayı uygula",
                "tf_order_apply",
            )
        else:
            show_info("Sıralamak için en az bir sütun seçin.")


def render() -> None:
    """Dönüşümler sayfasını çizer."""
    st.title("🧬 Dönüşümler")
    if not require_dataset("Veri dönüşümleri"):
        return

    message = st.session_state.pop("transform_message", None)
    if message:
        show_success(message)

    st.caption(
        "Dönüşümler çalışma kopyasına uygulanır, işlem geçmişine kaydedilir ve "
        "Python koduna çevrilir. Her adımı İşlem Geçmişi sayfasından geri alabilirsiniz."
    )

    df = session.get_working_df()
    tabs = st.tabs(
        [
            "🔤 Metin",
            "🧮 Sayısal",
            "🔢 Encoding",
            "📅 Tarih",
            "➕ Yeni Sütun",
            "🪣 Gruplama",
            "🔀 Düzen",
        ]
    )
    renderers = (
        _text_tab,
        _numeric_tab,
        _encoding_tab,
        _datetime_tab,
        _feature_tab,
        _binning_tab,
        _layout_tab,
    )
    for tab, renderer in zip(tabs, renderers):
        with tab, error_boundary("Dönüşüm"):
            renderer(df)
