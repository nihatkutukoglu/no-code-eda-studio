"""İşlem Geçmişi sayfası: geçmiş, undo ve reset.

Undo, snapshot saklamadan çalışır: ``original_df`` üzerinde kalan operasyonlar
yeniden oynatılarak (replay) ``working_df`` tekrar üretilir. Bu yaklaşım hem
RAM dostudur hem de notebook üretiminin temelini oluşturur.
"""

from __future__ import annotations

import streamlit as st

from components.messages import (
    disclaimer,
    error_boundary,
    require_dataset,
    section_header,
    show_info,
    show_success,
    show_warning,
    show_warning_list,
)
from components.metric_cards import metric_row
from components.tables import download_csv_button, show_dataframe
from core import session
from core.operations import ACTION_LABELS, Operation, history_table
from utils.helpers import format_number, py_literal, safe_filename

__all__ = ["render"]


def _undo() -> None:
    """Son işlemi geri alır."""
    with error_boundary("Geri alma"):
        removed = session.undo_last_operation()
        if removed:
            st.session_state["history_message"] = (
                "Geri alındı: " + str(removed.get("description", removed.get("action")))
            )
        st.rerun()


def _reset() -> None:
    """Veri setini orijinaline döndürür."""
    with error_boundary("Sıfırlama"):
        session.reset_dataset()
        st.session_state["history_message"] = (
            "Veri seti orijinal haline döndürüldü ve işlem geçmişi temizlendi."
        )
        st.rerun()


def render() -> None:
    """İşlem Geçmişi sayfasını çizer."""
    st.title("🕘 İşlem Geçmişi")
    if not require_dataset("İşlem geçmişi"):
        return

    message = st.session_state.pop("history_message", None)
    if message:
        show_success(message)
    show_warning_list(
        st.session_state.get("replay_errors") or [],
        "Yeniden uygulanamayan işlemler",
    )

    original = session.get_original_df()
    working = session.get_working_df()
    history = session.operation_history()

    metric_row(
        [
            ("Uygulanan işlem", format_number(len(history))),
            (
                "Satır",
                format_number(len(working)),
                f"Orijinal: {format_number(len(original))}",
            ),
            (
                "Sütun",
                format_number(working.shape[1]),
                f"Orijinal: {format_number(original.shape[1])}",
            ),
            ("Kaydedilen grafik", format_number(len(session.generated_charts()))),
        ],
        columns=4,
    )

    actions = st.columns([1, 1, 2])
    with actions[0]:
        if st.button(
            "↩️ Son İşlemi Geri Al",
            disabled=not history,
            width="stretch",
            help="Son operasyonu geçmişten çıkarır ve kalan adımları yeniden uygular.",
        ):
            _undo()
    with actions[1]:
        if st.button(
            "🔄 Orijinale Sıfırla",
            disabled=not history,
            width="stretch",
            type="secondary",
            help="Tüm işlemleri siler ve veri setini yüklenen haline döndürür.",
        ):
            _reset()
    with actions[2]:
        if history:
            st.caption(
                "Geri alma, veri kopyası saklamak yerine işlemleri yeniden "
                "oynatarak çalışır; bu yüzden bellek kullanımı düşüktür."
            )

    st.divider()

    if not history:
        show_info(
            "Henüz veri dönüşümü uygulanmadı. Veri Temizleme sayfasındaki işlemler "
            "burada sırayla listelenir."
        )
        return

    section_header("Uygulanan İşlemler", "Sıra, açıklama ve etkiler.", icon="📜")
    table = history_table(history)
    show_dataframe(table)
    download_csv_button(
        table,
        "⬇️ Geçmişi CSV indir",
        f"{safe_filename(session.dataset_meta().get('filename'))}_islem_gecmisi.csv",
        key="dl_history",
    )

    st.divider()
    section_header(
        "İşlem Detayları", "Her operasyonun parametreleri ve Python kodu.", icon="🔎"
    )
    for index, item in enumerate(history, start=1):
        operation = Operation.from_dict(item)
        with st.expander(f"{index}. {operation.label} — {operation.description}"):
            left, right = st.columns([1, 1])
            with left:
                st.markdown("**Operasyon**")
                st.markdown(
                    f"- Tip: `{operation.action}` ({ACTION_LABELS.get(operation.action, '-')})\n"
                    f"- ID: `{operation.id}`\n"
                    f"- Zaman: {operation.created_at.replace('T', ' ')}\n"
                    f"- Satır değişimi: {operation.rows_before} → {operation.rows_after}\n"
                    f"- Sütun değişimi: {operation.cols_before} → {operation.cols_after}"
                )
            with right:
                st.markdown("**Parametreler**")
                st.code(py_literal(operation.params), language="python")
            st.markdown("**Python kodu**")
            st.code(operation.python_code, language="python")

    if len(history) > 1:
        show_warning(
            "Yalnızca **son** işlem geri alınabilir. Aradaki bir adımı kaldırmak için "
            "veri setini sıfırlayıp işlemleri yeniden uygulamanız gerekir."
        )
    disclaimer(
        "Orijinal veri (original_df) hiçbir işlemde değiştirilmez; tüm dönüşümler "
        "çalışma kopyası üzerinde uygulanır."
    )
