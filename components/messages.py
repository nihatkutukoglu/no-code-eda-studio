"""Kullanıcı mesajları, boş durumlar (empty state) ve hata sınırı (error boundary).

Amaç: Uygulamanın hiçbir yerde traceback ile çökmemesi ve mesaj dilinin
tutarlı olması.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import streamlit as st

from utils.validators import EdaError

__all__ = [
    "show_error",
    "show_warning",
    "show_info",
    "show_success",
    "empty_state",
    "require_dataset",
    "error_boundary",
    "disclaimer",
    "section_header",
    "show_warning_list",
]


def show_error(message: object, icon: str = "🚫") -> None:
    """Hata mesajı gösterir."""
    st.error(str(message), icon=icon)


def show_warning(message: object, icon: str = "⚠️") -> None:
    """Uyarı mesajı gösterir."""
    st.warning(str(message), icon=icon)


def show_info(message: object, icon: str = "ℹ️") -> None:
    """Bilgi mesajı gösterir."""
    st.info(str(message), icon=icon)


def show_success(message: object, icon: str = "✅") -> None:
    """Başarı mesajı gösterir."""
    st.success(str(message), icon=icon)


def show_warning_list(messages: list[str], title: str = "Dikkat edilmesi gerekenler") -> None:
    """Birden fazla uyarıyı katlanabilir bir blokta gösterir."""
    if not messages:
        return
    with st.expander(f"⚠️ {title} ({len(messages)})", expanded=False):
        for message in messages:
            st.markdown(f"- {message}")


def section_header(title: str, description: str | None = None, icon: str = "") -> None:
    """Sayfa içi bölüm başlığı (tutarlı tipografi için)."""
    st.subheader(f"{icon} {title}".strip())
    if description:
        st.caption(description)


def empty_state(
    title: str,
    description: str = "",
    icon: str = "📂",
    hint: str | None = None,
) -> None:
    """Veri/sonuç olmadığında gösterilecek boş durum bloğu."""
    st.markdown(
        f"<div style='text-align:center; padding:2.5rem 1rem; "
        f"border:1px dashed rgba(128,128,128,.35); border-radius:12px;'>"
        f"<div style='font-size:2.5rem'>{icon}</div>"
        f"<div style='font-size:1.1rem; font-weight:600; margin-top:.5rem'>{title}</div>"
        f"<div style='opacity:.75; margin-top:.35rem'>{description}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    if hint:
        st.caption(hint)


def require_dataset(action: str = "Bu sayfayı kullanmak") -> bool:
    """Veri seti yoksa boş durum gösterir ve False döndürür.

    Args:
        action: Mesajda kullanılacak eylem ifadesi.

    Returns:
        Veri seti varsa True.
    """
    from core import session  # döngüsel import olmaması için fonksiyon içinde

    if session.has_dataset():
        return True
    empty_state(
        "Henüz veri seti yüklenmedi",
        f"{action} için sol menüden <b>Veri Yükle</b> sayfasına gidip "
        "bir CSV veya Excel dosyası yükleyin.",
        icon="📁",
    )
    return False


def disclaimer(message: str, icon: str = "📌") -> None:
    """Metodolojik uyarıları (korelasyon/aykırı değer/skor) gösterir."""
    st.caption(f"{icon} {message}")


@contextmanager
def error_boundary(context: str = "İşlem") -> Iterator[None]:
    """Blok içindeki hataları yakalayıp kullanıcı dostu mesaja çevirir.

    ``EdaError`` türevleri doğrudan gösterilir; beklenmeyen hatalar için
    genel bir mesaj ve teknik detay ayrı bir blokta sunulur.

    Example:
        >>> with error_boundary("Grafik oluşturma"):
        ...     build_chart(...)
    """
    try:
        yield
    except EdaError as exc:
        show_error(str(exc))
    except Exception as exc:  # son savunma hattı: uygulama çökmesin
        show_error(
            f"{context} sırasında beklenmeyen bir hata oluştu. "
            "Veri setiniz değiştirilmedi."
        )
        with st.expander("Teknik detay"):
            st.code(f"{type(exc).__name__}: {exc}")
