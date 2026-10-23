"""Veri dönüşümleri: metin temizleme, sayısal dönüşümler, ölçekleme,
kategorik encoding, tarih özellikleri, aritmetik feature üretimi ve binning.

Tüm fonksiyonlar **saf**tır: verilen DataFrame'i değiştirmez, yeni bir
DataFrame döndürür. Böylece ``operation_history`` üzerinden replay (undo)
deterministik çalışır.

Ölçekleme ve encoding işlemleri bilinçli olarak **pandas** ile yapılır
(scikit-learn değil): böylece üretilen kod ek bağımlılık gerektirmez ve
replay her ortamda aynı sonucu verir. scikit-learn karşılıkları ML Hazırlığı
sayfasındaki pipeline kodunda üretilir.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from utils import dataframe_utils as dfu
from utils.helpers import unique_name
from utils.validators import CleaningError

__all__ = [
    "TEXT_METHODS",
    "NUMERIC_METHODS",
    "ENCODING_METHODS",
    "DATETIME_FEATURES",
    "ARITHMETIC_OPERATIONS",
    "BINNING_METHODS",
    "text_transform",
    "numeric_transform",
    "encode_categorical",
    "datetime_features",
    "arithmetic_feature",
    "bin_column",
    "duplicate_column",
    "reorder_columns",
    "sort_rows",
    "transform_preview",
    "suggest_feature_name",
]

#: Metin temizleme yöntemleri.
TEXT_METHODS: dict[str, str] = {
    "trim": "Baştaki/sondaki boşlukları kırp",
    "lower": "küçük harfe çevir",
    "upper": "BÜYÜK HARFE çevir",
    "title": "Her Kelimeyi Büyük Harfle Başlat",
    "replace": "Metni değiştir (bul/değiştir)",
    "remove_chars": "Belirli karakterleri sil",
    "normalize_space": "Çoklu boşlukları tek boşluğa indir",
}

#: Sayısal dönüşüm yöntemleri.
NUMERIC_METHODS: dict[str, str] = {
    "log": "Log (doğal logaritma)",
    "log1p": "Log(1+x)",
    "sqrt": "Karekök",
    "abs": "Mutlak değer",
    "round": "Yuvarla",
    "minmax": "Min-Max ölçekleme (0-1)",
    "standard": "Standart ölçekleme (z-score)",
    "robust": "Robust ölçekleme (medyan/IQR)",
}

#: Kategorik encoding yöntemleri.
ENCODING_METHODS: dict[str, str] = {
    "label": "Label Encoding (tek sütun, 0..n-1)",
    "onehot": "One-Hot Encoding (her kategori için sütun)",
    "ordinal": "Ordinal Encoding (sıralamayı siz belirleyin)",
}

#: Tarih sütunundan üretilebilecek özellikler.
DATETIME_FEATURES: dict[str, str] = {
    "year": "Yıl",
    "quarter": "Çeyrek",
    "month": "Ay (1-12)",
    "month_name": "Ay adı",
    "day": "Gün",
    "weekday": "Haftanın günü (0=Pazartesi)",
    "weekday_name": "Gün adı",
    "hour": "Saat",
    "is_weekend": "Hafta sonu mu? (0/1)",
    "week": "Yılın haftası",
    "dayofyear": "Yılın günü",
}

#: Aritmetik feature işlemleri.
ARITHMETIC_OPERATIONS: dict[str, str] = {
    "add": "Topla (a + b)",
    "subtract": "Çıkar (a - b)",
    "multiply": "Çarp (a × b)",
    "divide": "Böl (a / b)",
    "ratio": "Oran (a / (a + b))",
}

#: Binning yöntemleri.
BINNING_METHODS: dict[str, str] = {
    "equal_width": "Eşit genişlikli aralıklar",
    "quantile": "Eşit sayıda kayıt (quantile)",
    "custom": "Özel sınırlar",
}


def _require_column(df: pd.DataFrame, column: Any) -> str:
    """Sütunun varlığını doğrular."""
    if column is None or column not in df.columns:
        raise CleaningError(
            f"'{column}' sütunu veri setinde bulunamadı. "
            "Sütun silinmiş veya adı değiştirilmiş olabilir."
        )
    return str(column)


def _require_numeric(df: pd.DataFrame, column: Any) -> str:
    """Sütunun sayısal olduğunu doğrular."""
    column = _require_column(df, column)
    if column not in dfu.numeric_columns(df):
        raise CleaningError(
            f"Bu dönüşüm sayısal sütun gerektirir. '{column}' sütunu "
            f"{df[column].dtype} tipinde. Önce tipi dönüştürün."
        )
    return column


def _target_name(df: pd.DataFrame, column: str, new_column: str | None) -> str:
    """Sonucun yazılacağı sütun adını belirler.

    ``new_column`` None ise mevcut sütun yerinde değiştirilir.
    """
    if not new_column:
        return column
    cleaned = str(new_column).strip()
    if not cleaned:
        return column
    if cleaned != column and cleaned in df.columns:
        raise CleaningError(
            f"'{cleaned}' adında bir sütun zaten var. Farklı bir ad girin."
        )
    return cleaned


def suggest_feature_name(df: pd.DataFrame, base: str) -> str:
    """Mevcut sütunlarla çakışmayan bir sütun adı önerir."""
    return unique_name(str(base), [str(c) for c in df.columns])


# ---------------------------------------------------------------------------
# Metin temizleme
# ---------------------------------------------------------------------------


def text_transform(
    df: pd.DataFrame,
    column: str,
    method: str,
    find: str | None = None,
    replace_with: str = "",
    characters: str | None = None,
    new_column: str | None = None,
) -> pd.DataFrame:
    """Metin sütununa temizleme dönüşümü uygular.

    Args:
        df: Kaynak DataFrame.
        column: Metin sütunu.
        method: ``TEXT_METHODS`` içindeki bir anahtar.
        find: ``replace`` yönteminde aranacak metin.
        replace_with: ``replace`` yönteminde yazılacak metin.
        characters: ``remove_chars`` yönteminde silinecek karakterler.
        new_column: Yeni sütun adı (None ise sütun yerinde değişir).

    Returns:
        Dönüşüm uygulanmış yeni DataFrame.

    Raises:
        CleaningError: Yöntem veya parametreler geçersizse.
    """
    column = _require_column(df, column)
    if method not in TEXT_METHODS:
        raise CleaningError(f"Desteklenmeyen metin dönüşümü: {method}")

    target = _target_name(df, column, new_column)
    series = df[column]
    if ptypes.is_numeric_dtype(series) or ptypes.is_datetime64_any_dtype(series):
        raise CleaningError(
            f"'{column}' metin sütunu değil ({series.dtype}). "
            "Metin dönüşümleri yalnızca metin/kategorik sütunlarda çalışır."
        )

    text = series.astype("string")
    if method == "trim":
        converted = text.str.strip()
    elif method == "lower":
        converted = text.str.lower()
    elif method == "upper":
        converted = text.str.upper()
    elif method == "title":
        converted = text.str.title()
    elif method == "normalize_space":
        converted = text.str.replace(r"\s+", " ", regex=True).str.strip()
    elif method == "replace":
        if find is None or find == "":
            raise CleaningError("Değiştirmek için aranacak metni girmelisiniz.")
        converted = text.str.replace(str(find), str(replace_with or ""), regex=False)
    else:  # remove_chars
        if not characters:
            raise CleaningError("Silinecek karakterleri girmelisiniz.")
        converted = text
        for char in str(characters):
            converted = converted.str.replace(char, "", regex=False)

    result = df.copy()
    result[target] = converted
    return result


# ---------------------------------------------------------------------------
# Sayısal dönüşümler ve ölçekleme
# ---------------------------------------------------------------------------


def numeric_transform(
    df: pd.DataFrame,
    column: str,
    method: str,
    decimals: int = 2,
    new_column: str | None = None,
) -> pd.DataFrame:
    """Sayısal sütuna matematiksel dönüşüm veya ölçekleme uygular.

    Args:
        df: Kaynak DataFrame.
        column: Sayısal sütun.
        method: ``NUMERIC_METHODS`` içindeki bir anahtar.
        decimals: ``round`` için ondalık basamak sayısı.
        new_column: Yeni sütun adı (None ise sütun yerinde değişir).

    Returns:
        Dönüşüm uygulanmış yeni DataFrame.

    Raises:
        CleaningError: Yöntem geçersizse veya dönüşüm tanımsızsa
            (ör. sabit sütunda min-max, tüm değerlerin negatif olduğu log).
    """
    column = _require_numeric(df, column)
    if method not in NUMERIC_METHODS:
        raise CleaningError(f"Desteklenmeyen sayısal dönüşüm: {method}")

    target = _target_name(df, column, new_column)
    series = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    clean = series.dropna()
    if clean.empty:
        raise CleaningError(f"'{column}' sütununda dönüştürülecek dolu değer yok.")

    if method == "log":
        if (clean <= 0).all():
            raise CleaningError(
                f"'{column}' sütunundaki tüm değerler sıfır veya negatif; "
                "logaritma tanımsız. Log(1+x) veya başka bir dönüşüm deneyin."
            )
        converted = np.log(series.where(series > 0))
    elif method == "log1p":
        if (clean <= -1).all():
            raise CleaningError(
                f"'{column}' için log(1+x) tanımsız (tüm değerler <= -1)."
            )
        converted = np.log1p(series.where(series > -1))
    elif method == "sqrt":
        if (clean < 0).all():
            raise CleaningError(
                f"'{column}' sütunundaki tüm değerler negatif; karekök tanımsız."
            )
        converted = np.sqrt(series.where(series >= 0))
    elif method == "abs":
        converted = series.abs()
    elif method == "round":
        converted = series.round(int(decimals))
    elif method == "minmax":
        low, high = float(clean.min()), float(clean.max())
        if high == low:
            raise CleaningError(
                f"'{column}' sabit bir sütun (min = max); Min-Max ölçekleme yapılamaz."
            )
        converted = (series - low) / (high - low)
    elif method == "standard":
        mean, std = float(clean.mean()), float(clean.std(ddof=0))
        if std == 0:
            raise CleaningError(
                f"'{column}' sütununun standart sapması 0; z-score hesaplanamaz."
            )
        converted = (series - mean) / std
    else:  # robust
        median = float(clean.median())
        iqr = float(clean.quantile(0.75) - clean.quantile(0.25))
        if iqr == 0:
            raise CleaningError(
                f"'{column}' sütununun IQR değeri 0; robust ölçekleme yapılamaz."
            )
        converted = (series - median) / iqr

    result = df.copy()
    result[target] = converted
    return result


# ---------------------------------------------------------------------------
# Kategorik encoding
# ---------------------------------------------------------------------------


def encode_categorical(
    df: pd.DataFrame,
    column: str,
    method: str = "label",
    categories: list[str] | None = None,
    drop_first: bool = False,
    new_column: str | None = None,
) -> pd.DataFrame:
    """Kategorik sütunu sayısal temsile çevirir.

    Args:
        df: Kaynak DataFrame.
        column: Kategorik sütun.
        method: ``label``, ``onehot`` veya ``ordinal``.
        categories: ``ordinal`` için kategori sırası (küçükten büyüğe).
        drop_first: ``onehot`` için ilk kategoriyi düşür (kukla değişken tuzağı).
        new_column: ``label``/``ordinal`` için hedef sütun adı.

    Returns:
        Encoding uygulanmış yeni DataFrame.

    Raises:
        CleaningError: Yöntem veya parametreler geçersizse.
    """
    column = _require_column(df, column)
    if method not in ENCODING_METHODS:
        raise CleaningError(f"Desteklenmeyen encoding yöntemi: {method}")

    series = df[column]
    if ptypes.is_datetime64_any_dtype(series):
        raise CleaningError(
            "Tarih sütunları encode edilemez. Önce Tarih Özellikleri bölümünden "
            "yıl/ay gibi alanlar üretin."
        )

    if method == "onehot":
        nunique = dfu.safe_nunique(series)
        if nunique > 100:
            raise CleaningError(
                f"'{column}' sütununda {nunique} farklı değer var. One-Hot "
                "Encoding çok fazla sütun üretir; Label Encoding kullanın veya "
                "önce kategorileri sadeleştirin."
            )
        dummies = pd.get_dummies(
            series.astype("string"), prefix=str(column), drop_first=bool(drop_first)
        )
        dummies = dummies.astype("int8")
        clashes = [c for c in dummies.columns if c in df.columns]
        if clashes:
            raise CleaningError(
                "Oluşacak sütun adları mevcut sütunlarla çakışıyor: "
                + ", ".join(map(str, clashes[:5]))
            )
        result = df.drop(columns=[column])
        return pd.concat([result, dummies], axis=1)

    target = _target_name(df, column, new_column)
    if method == "label":
        codes = series.astype("string").astype("category").cat.codes
        converted = codes.where(series.notna(), pd.NA).astype("Int64")
    else:  # ordinal
        if not categories:
            raise CleaningError(
                "Ordinal encoding için kategori sırasını belirlemelisiniz."
            )
        mapping = {str(value): index for index, value in enumerate(categories)}
        converted = series.astype("string").map(mapping).astype("Int64")
        if converted.notna().sum() == 0:
            raise CleaningError(
                "Belirlediğiniz sıralama sütundaki değerlerle eşleşmedi; "
                "işlem uygulanmadı."
            )

    result = df.copy()
    result[target] = converted
    return result


# ---------------------------------------------------------------------------
# Tarih özellikleri
# ---------------------------------------------------------------------------

_WEEKDAY_NAMES = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]
_MONTH_NAMES = [
    "Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran",
    "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık",
]


def datetime_features(
    df: pd.DataFrame, column: str, features: list[str], prefix: str | None = None
) -> pd.DataFrame:
    """Tarih sütunundan yeni özellik sütunları üretir.

    Args:
        df: Kaynak DataFrame.
        column: Tarih sütunu (gerekirse dönüştürülmeye çalışılır).
        features: ``DATETIME_FEATURES`` anahtarlarından oluşan liste.
        prefix: Yeni sütun öneki (None ise sütun adı kullanılır).

    Returns:
        Yeni özellik sütunları eklenmiş DataFrame.

    Raises:
        CleaningError: Sütun tarihe çevrilemezse veya özellik listesi boşsa.
    """
    column = _require_column(df, column)
    if not features:
        raise CleaningError("En az bir tarih özelliği seçmelisiniz.")
    unknown = [f for f in features if f not in DATETIME_FEATURES]
    if unknown:
        raise CleaningError("Bilinmeyen tarih özelliği: " + ", ".join(unknown))

    series = df[column]
    if not ptypes.is_datetime64_any_dtype(series):
        with warnings.catch_warnings():
            # Biçim verilmediğinde pandas satır bazında ayrıştırma uyarısı verir;
            # çevrilemeyen değerler zaten NaT olarak işaretlenir.
            warnings.simplefilter("ignore", UserWarning)
            converted = pd.to_datetime(series, errors="coerce")
        if converted.notna().sum() == 0:
            raise CleaningError(
                f"'{column}' sütunu tarihe çevrilemedi. Önce Veri Temizleme "
                "sayfasından datetime tipine dönüştürün."
            )
        series = converted

    base = str(prefix or column)
    result = df.copy()
    accessor = series.dt
    builders: dict[str, Any] = {
        "year": lambda: accessor.year.astype("Int64"),
        "quarter": lambda: accessor.quarter.astype("Int64"),
        "month": lambda: accessor.month.astype("Int64"),
        "month_name": lambda: accessor.month.map(
            lambda m: _MONTH_NAMES[int(m) - 1] if pd.notna(m) else pd.NA
        ).astype("string"),
        "day": lambda: accessor.day.astype("Int64"),
        "weekday": lambda: accessor.dayofweek.astype("Int64"),
        "weekday_name": lambda: accessor.dayofweek.map(
            lambda d: _WEEKDAY_NAMES[int(d)] if pd.notna(d) else pd.NA
        ).astype("string"),
        "hour": lambda: accessor.hour.astype("Int64"),
        "is_weekend": lambda: (accessor.dayofweek >= 5).astype("Int64"),
        "week": lambda: accessor.isocalendar().week.astype("Int64"),
        "dayofyear": lambda: accessor.dayofyear.astype("Int64"),
    }

    for feature in features:
        name = f"{base}_{feature}"
        if name in result.columns:
            raise CleaningError(
                f"'{name}' adında bir sütun zaten var. Farklı bir önek kullanın."
            )
        result[name] = builders[feature]()
    return result


# ---------------------------------------------------------------------------
# Aritmetik feature üretimi
# ---------------------------------------------------------------------------


def arithmetic_feature(
    df: pd.DataFrame,
    left: str,
    right: str,
    operation: str,
    new_column: str,
) -> pd.DataFrame:
    """İki sayısal sütundan yeni bir sütun üretir.

    Sıfıra bölme durumunda sonuç NaN olur (uygulama çökmez).

    Args:
        df: Kaynak DataFrame.
        left: Birinci sütun.
        right: İkinci sütun.
        operation: ``ARITHMETIC_OPERATIONS`` içindeki bir anahtar.
        new_column: Oluşturulacak sütunun adı.

    Returns:
        Yeni sütun eklenmiş DataFrame.

    Raises:
        CleaningError: Sütunlar sayısal değilse veya ad geçersizse.
    """
    left = _require_numeric(df, left)
    right = _require_numeric(df, right)
    if operation not in ARITHMETIC_OPERATIONS:
        raise CleaningError(f"Desteklenmeyen işlem: {operation}")
    name = str(new_column or "").strip()
    if not name:
        raise CleaningError("Yeni sütun için bir ad girmelisiniz.")
    if name in df.columns:
        raise CleaningError(f"'{name}' adında bir sütun zaten var.")

    a = pd.to_numeric(df[left], errors="coerce")
    b = pd.to_numeric(df[right], errors="coerce")
    if operation == "add":
        values = a + b
    elif operation == "subtract":
        values = a - b
    elif operation == "multiply":
        values = a * b
    elif operation == "divide":
        values = a / b.replace(0, np.nan)
    else:  # ratio
        total = (a + b).replace(0, np.nan)
        values = a / total

    result = df.copy()
    result[name] = values.replace([np.inf, -np.inf], np.nan)
    return result


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------


def bin_column(
    df: pd.DataFrame,
    column: str,
    method: str = "equal_width",
    bins: int = 4,
    edges: list[float] | None = None,
    labels: list[str] | None = None,
    new_column: str | None = None,
) -> pd.DataFrame:
    """Sayısal sütunu aralıklara (bin) ayırır.

    Args:
        df: Kaynak DataFrame.
        column: Sayısal sütun.
        method: ``equal_width``, ``quantile`` veya ``custom``.
        bins: Aralık sayısı (``equal_width`` / ``quantile``).
        edges: ``custom`` için sınır listesi (artan sırada).
        labels: Aralık etiketleri (opsiyonel).
        new_column: Hedef sütun adı (None ise ``<column>_bin``).

    Returns:
        Bin sütunu eklenmiş DataFrame.

    Raises:
        CleaningError: Parametreler geçersizse veya bin üretilemezse.
    """
    column = _require_numeric(df, column)
    if method not in BINNING_METHODS:
        raise CleaningError(f"Desteklenmeyen binning yöntemi: {method}")

    target = new_column or f"{column}_bin"
    target = str(target).strip()
    if target in df.columns and target != column:
        raise CleaningError(f"'{target}' adında bir sütun zaten var.")

    series = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if series.dropna().empty:
        raise CleaningError(f"'{column}' sütununda bin oluşturacak dolu değer yok.")

    clean_labels = [str(label) for label in labels] if labels else None
    try:
        if method == "custom":
            if not edges or len(edges) < 2:
                raise CleaningError(
                    "Özel binning için en az 2 sınır değeri girmelisiniz (ör. 0, 18, 30)."
                )
            ordered = sorted(float(edge) for edge in edges)
            if len(set(ordered)) != len(ordered):
                raise CleaningError("Sınır değerleri birbirinden farklı olmalıdır.")
            if clean_labels and len(clean_labels) != len(ordered) - 1:
                raise CleaningError(
                    f"{len(ordered)} sınır için {len(ordered) - 1} etiket girmelisiniz."
                )
            binned = pd.cut(
                series, bins=ordered, labels=clean_labels, include_lowest=True
            )
        elif method == "quantile":
            count = int(bins)
            if count < 2:
                raise CleaningError("Aralık sayısı en az 2 olmalıdır.")
            if clean_labels and len(clean_labels) != count:
                raise CleaningError(f"{count} aralık için {count} etiket girmelisiniz.")
            binned = pd.qcut(series, q=count, labels=clean_labels, duplicates="drop")
        else:  # equal_width
            count = int(bins)
            if count < 2:
                raise CleaningError("Aralık sayısı en az 2 olmalıdır.")
            if clean_labels and len(clean_labels) != count:
                raise CleaningError(f"{count} aralık için {count} etiket girmelisiniz.")
            binned = pd.cut(series, bins=count, labels=clean_labels)
    except CleaningError:
        raise
    except ValueError as exc:
        raise CleaningError(
            f"Bin oluşturulamadı: {exc}. Aralık sayısını azaltmayı deneyin."
        ) from exc

    if binned.dropna().empty:
        raise CleaningError(
            "Hiçbir değer verilen aralıklara girmedi; sınırları kontrol edin."
        )

    result = df.copy()
    result[target] = binned.astype("string") if clean_labels is None else binned.astype("string")
    return result


# ---------------------------------------------------------------------------
# Sütun / satır düzeni
# ---------------------------------------------------------------------------


def duplicate_column(df: pd.DataFrame, column: str, new_column: str | None = None) -> pd.DataFrame:
    """Bir sütunun kopyasını oluşturur."""
    column = _require_column(df, column)
    target = str(new_column or f"{column}_kopya").strip()
    if not target:
        raise CleaningError("Yeni sütun adı boş olamaz.")
    if target in df.columns:
        raise CleaningError(f"'{target}' adında bir sütun zaten var.")
    result = df.copy()
    result[target] = df[column]
    return result


def reorder_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Sütun sırasını değiştirir.

    Listede olmayan sütunlar mevcut sıralarıyla sona eklenir.
    """
    if not columns:
        raise CleaningError("Sıralama için en az bir sütun seçmelisiniz.")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise CleaningError("Şu sütunlar bulunamadı: " + ", ".join(map(str, missing)))
    rest = [str(c) for c in df.columns if c not in columns]
    return df[[*[str(c) for c in columns], *rest]]


def sort_rows(
    df: pd.DataFrame, columns: list[str], ascending: list[bool] | bool = True
) -> pd.DataFrame:
    """Satırları bir veya birden fazla sütuna göre sıralar.

    Args:
        df: Kaynak DataFrame.
        columns: Sıralama sütunları.
        ascending: Her sütun için artan/azalan bilgisi.

    Returns:
        Sıralanmış yeni DataFrame (indeks sıfırlanır).
    """
    if not columns:
        raise CleaningError("Sıralama için en az bir sütun seçmelisiniz.")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise CleaningError("Şu sütunlar bulunamadı: " + ", ".join(map(str, missing)))
    if isinstance(ascending, bool):
        order: Any = ascending
    else:
        if len(ascending) != len(columns):
            raise CleaningError("Sıralama yönü sayısı sütun sayısıyla aynı olmalıdır.")
        order = [bool(value) for value in ascending]
    try:
        result = df.sort_values(by=[str(c) for c in columns], ascending=order)
    except TypeError as exc:
        raise CleaningError(
            f"Sıralama yapılamadı: {exc}. Karışık tipli sütunlarda sıralama "
            "için önce tip dönüşümü uygulayın."
        ) from exc
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Ön izleme
# ---------------------------------------------------------------------------


def transform_preview(
    df: pd.DataFrame, before: str, after_df: pd.DataFrame, after: str, rows: int = 5
) -> pd.DataFrame:
    """Dönüşüm öncesi/sonrası değerleri yan yana gösteren küçük tablo üretir.

    Args:
        df: Dönüşüm öncesi DataFrame.
        before: Önceki sütun adı.
        after_df: Dönüşüm sonrası DataFrame.
        after: Sonraki sütun adı.
        rows: Gösterilecek satır sayısı.
    """
    sample = pd.DataFrame(
        {
            "Önce": df[before].head(rows).to_numpy()
            if before in df.columns
            else [None] * min(rows, len(df)),
            "Sonra": after_df[after].head(rows).to_numpy()
            if after in after_df.columns
            else [None] * min(rows, len(after_df)),
        }
    )
    return sample
