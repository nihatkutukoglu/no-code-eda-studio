"""CSV / Excel / JSON / Parquet dosyalarını dayanıklı biçimde okuyan katman.

Hedefler:
    * Farklı encoding (UTF-8, Windows-1254, ISO-8859-9, Latin-1) desteği.
    * Ayraç (``,`` ``;`` ``tab`` ``|``) otomatik tespiti.
    * JSON (kayıt listesi, satır-satır JSONL, iç içe yapılar) ve Parquet desteği.
    * Bozuk dosyalarda kullanıcı dostu Türkçe hata mesajı.
    * Okunan dosyanın Python kod karşılığının üretilebilmesi.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from utils.helpers import col_literal, py_literal, unique_name
from utils.validators import LoaderError

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "CSV_ENCODINGS",
    "CSV_SEPARATORS",
    "LoadResult",
    "detect_extension",
    "EXTENSION_TYPES",
    "excel_sheet_names",
    "load_dataframe",
    "loading_code",
]

#: Desteklenen dosya uzantıları.
SUPPORTED_EXTENSIONS = ("csv", "txt", "tsv", "xlsx", "xls", "json", "jsonl", "parquet")
#: Uzantıdan dosya tipine eşleme.
EXTENSION_TYPES = {
    "csv": "csv",
    "txt": "csv",
    "tsv": "csv",
    "xlsx": "excel",
    "xls": "excel",
    "json": "json",
    "jsonl": "json",
    "parquet": "parquet",
}
#: Denenecek encoding sırası.
CSV_ENCODINGS = ("utf-8-sig", "utf-8", "cp1254", "iso-8859-9", "latin-1")
#: Denenecek ayraç adayları.
CSV_SEPARATORS = (",", ";", "\t", "|")

_SEPARATOR_LABELS = {",": "virgül", ";": "noktalı virgül", "\t": "tab", "|": "dikey çizgi"}


@dataclass
class LoadResult:
    """Dosya okuma sonucunu ve okuma parametrelerini taşır."""

    df: pd.DataFrame
    filename: str
    file_type: str
    encoding: str | None = None
    separator: str | None = None
    sheet_name: str | int | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def separator_label(self) -> str:
        """Ayracın kullanıcıya gösterilecek adı."""
        if self.separator is None:
            return "-"
        return _SEPARATOR_LABELS.get(self.separator, self.separator)

    def summary(self) -> str:
        """Okuma parametrelerinin tek satırlık özeti."""
        parts = [f"{self.df.shape[0]:,} satır", f"{self.df.shape[1]:,} sütun"]
        if self.file_type == "csv":
            parts.append(f"ayraç: {self.separator_label}")
            parts.append(f"encoding: {self.encoding}")
        elif self.sheet_name is not None:
            parts.append(f"sayfa: {self.sheet_name}")
        return " · ".join(parts)


def detect_extension(filename: str) -> str:
    """Dosya adından uzantıyı küçük harf olarak çıkarır.

    Raises:
        LoaderError: Uzantı desteklenmiyorsa.
    """
    name = str(filename or "").strip()
    if "." not in name:
        raise LoaderError(
            "Dosya uzantısı anlaşılamadı. Desteklenen biçimler: "
            "CSV, XLSX, XLS, JSON, Parquet."
        )
    ext = name.rsplit(".", 1)[-1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise LoaderError(
            f"'.{ext}' uzantısı desteklenmiyor. Desteklenen biçimler: "
            "CSV, TXT, TSV, XLSX, XLS, JSON, JSONL, Parquet."
        )
    return ext


def _sniff_separator(sample: str) -> str:
    """Metin örneğinden en olası ayracı tahmin eder."""
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="".join(CSV_SEPARATORS))
        if dialect.delimiter in CSV_SEPARATORS:
            return str(dialect.delimiter)
    except (csv.Error, TypeError):
        pass

    lines = [line for line in sample.splitlines() if line.strip()][:20]
    best, best_score = ",", -1.0
    for sep in CSV_SEPARATORS:
        counts = [line.count(sep) for line in lines]
        if not counts or max(counts) == 0:
            continue
        # Tutarlı sayıda ayraç içeren ayracı tercih et.
        consistency = sum(1 for c in counts if c == counts[0]) / len(counts)
        score = counts[0] * consistency
        if score > best_score:
            best, best_score = sep, score
    return best


def _decode_sample(data: bytes) -> tuple[str, str]:
    """Baytları denenen encoding sırasına göre çözer.

    Returns:
        (çözülmüş_metin, encoding) ikilisi.

    Raises:
        LoaderError: Hiçbir encoding ile çözülemezse.
    """
    for enc in CSV_ENCODINGS:
        try:
            return data.decode(enc), enc
        except (UnicodeDecodeError, LookupError):
            continue
    raise LoaderError(
        "Dosyanın karakter kodlaması çözümlenemedi. "
        "Dosyayı UTF-8 olarak kaydedip tekrar deneyebilirsiniz."
    )


def _read_csv_bytes(
    data: bytes, separator: str | None, encoding: str | None
) -> tuple[pd.DataFrame, str, str, list[str]]:
    """CSV baytlarını okur; ayraç ve encoding tespitini yönetir.

    Returns:
        (df, kullanılan_ayraç, kullanılan_encoding, uyarılar)
    """
    warnings: list[str] = []

    if encoding:
        try:
            text = data.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            text, encoding = _decode_sample(data)
            warnings.append(
                f"Seçilen encoding ile dosya okunamadı, '{encoding}' ile okundu."
            )
    else:
        text, encoding = _decode_sample(data)

    if not text.strip():
        raise LoaderError("Dosya boş görünüyor. İçinde veri olan bir dosya yükleyin.")

    sample = text[:65_536]
    sniffed = separator or _sniff_separator(sample)
    candidates = [sniffed] if separator else [sniffed] + [s for s in CSV_SEPARATORS if s != sniffed]

    best: tuple[pd.DataFrame, str] | None = None
    last_error: Exception | None = None
    for sep in candidates:
        try:
            candidate = pd.read_csv(
                io.StringIO(text),
                sep=sep,
                engine="python",
                skip_blank_lines=True,
                on_bad_lines="warn",
            )
        except Exception as exc:  # pandas çok çeşitli hata tipi atabilir
            last_error = exc
            continue
        # En çok sütun üreten ayraç en olası doğru ayraçtır.
        if best is None or candidate.shape[1] > best[0].shape[1]:
            best = (candidate, sep)
        if best[0].shape[1] > 1 and sep == sniffed:
            break

    if best is None:
        raise LoaderError(
            "CSV dosyası ayrıştırılamadı. Dosyanın bozuk olup olmadığını kontrol edin."
            + (f" (Teknik detay: {last_error})" if last_error else "")
        )

    df, chosen = best
    if df.shape[1] == 1 and not separator:
        warnings.append(
            "Dosya tek sütun olarak okundu. Ayraç yanlış algılanmış olabilir; "
            "gelişmiş ayarlardan ayracı elle seçebilirsiniz."
        )
    return df, chosen, encoding, warnings


def excel_sheet_names(data: bytes, filename: str) -> list[str]:
    """Excel dosyasındaki sayfa adlarını döndürür.

    Raises:
        LoaderError: Dosya okunamazsa.
    """
    ext = detect_extension(filename)
    if ext not in ("xlsx", "xls"):
        return []
    engine = "openpyxl" if ext == "xlsx" else "xlrd"
    try:
        with pd.ExcelFile(io.BytesIO(data), engine=engine) as book:
            return [str(name) for name in book.sheet_names]
    except ImportError as exc:
        raise LoaderError(
            f"Excel okuma motoru bulunamadı ({engine}). "
            "requirements.txt içindeki paketleri kurduğunuzdan emin olun."
        ) from exc
    except Exception as exc:
        raise LoaderError(
            "Excel dosyası açılamadı. Dosya bozuk veya şifreli olabilir. "
            f"(Teknik detay: {exc})"
        ) from exc


def _read_excel_bytes(
    data: bytes, ext: str, sheet_name: str | int | None
) -> tuple[pd.DataFrame, str | int, list[str]]:
    """Excel baytlarını okur.

    Returns:
        (df, kullanılan_sayfa, uyarılar)
    """
    engine = "openpyxl" if ext == "xlsx" else "xlrd"
    target: str | int = 0 if sheet_name is None else sheet_name
    try:
        df = pd.read_excel(io.BytesIO(data), sheet_name=target, engine=engine)
    except ImportError as exc:
        raise LoaderError(
            f"Excel okuma motoru bulunamadı ({engine}). "
            "requirements.txt içindeki paketleri kurun."
        ) from exc
    except ValueError as exc:
        raise LoaderError(
            f"Excel sayfası okunamadı: {exc}. Sayfa adını kontrol edin."
        ) from exc
    except Exception as exc:
        raise LoaderError(
            "Excel dosyası okunamadı. Dosya bozuk, şifreli veya desteklenmeyen "
            f"bir biçimde olabilir. (Teknik detay: {exc})"
        ) from exc

    if isinstance(df, dict):  # sheet_name=None verilmişse
        first = next(iter(df))
        return df[first], first, [f"Birden fazla sayfa bulundu, '{first}' sayfası okundu."]
    return df, target, []


def _read_json_bytes(data: bytes) -> tuple[pd.DataFrame, str, list[str]]:
    """JSON / JSONL baytlarını okur.

    Desteklenen yapılar:
        * Kayıt listesi: ``[{"a": 1}, {"a": 2}]``
        * Satır-satır JSON (JSONL): her satırda bir nesne
        * Sütun sözlüğü: ``{"a": [1, 2]}``
        * İç içe nesneler (``json_normalize`` ile düzleştirilir)

    Returns:
        ``(df, kullanılan_encoding, uyarılar)``
    """
    text, encoding = _decode_sample(data)
    if not text.strip():
        raise LoaderError("JSON dosyası boş görünüyor.")

    warnings: list[str] = []
    payload: Any = None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        # JSONL denemesi: her satır ayrı bir JSON nesnesi
        records: list[Any] = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise LoaderError(
                    "JSON dosyası ayrıştırılamadı. "
                    f"{line_no}. satırda sorun var: {exc.msg}"
                ) from exc
        if not records:
            raise LoaderError("JSON dosyasında okunabilir kayıt bulunamadı.")
        payload = records
        warnings.append("Dosya satır-satır JSON (JSONL) olarak okundu.")

    if isinstance(payload, dict):
        # Tek bir kayıt mı, sütun sözlüğü mü?
        values = list(payload.values())
        if values and all(isinstance(v, (list, tuple)) for v in values):
            df = pd.DataFrame(payload)
        else:
            df = pd.json_normalize(payload)
            warnings.append("JSON nesnesi tek satırlık tablo olarak okundu.")
    elif isinstance(payload, list):
        if payload and all(isinstance(item, dict) for item in payload):
            df = pd.json_normalize(payload)
            if any("." in str(column) for column in df.columns):
                warnings.append(
                    "İç içe JSON alanları nokta ile birleştirilerek düzleştirildi."
                )
        else:
            df = pd.DataFrame(payload)
            if df.shape[1] == 1:
                df.columns = ["value"]
    else:
        raise LoaderError(
            "JSON içeriği tabloya çevrilemedi. Kayıt listesi veya sütun "
            "sözlüğü biçiminde bir dosya yükleyin."
        )
    return df, encoding, warnings


def _read_parquet_bytes(data: bytes) -> tuple[pd.DataFrame, list[str]]:
    """Parquet baytlarını okur.

    Returns:
        ``(df, uyarılar)``
    """
    try:
        df = pd.read_parquet(io.BytesIO(data))
    except ImportError as exc:
        raise LoaderError(
            "Parquet okumak için 'pyarrow' paketi gerekli. "
            "requirements.txt içindeki paketleri kurun."
        ) from exc
    except Exception as exc:
        raise LoaderError(
            "Parquet dosyası okunamadı. Dosya bozuk veya desteklenmeyen bir "
            f"sürümde olabilir. (Teknik detay: {exc})"
        ) from exc
    return df, []


def _normalize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Sütun adlarını temizler: boşlukları kırpar, boş/tekrarlı adları düzeltir.

    Returns:
        (yeni_df, uyarılar)
    """
    warnings: list[str] = []
    new_names: list[str] = []
    renamed = 0
    for idx, col in enumerate(df.columns, start=1):
        name = str(col).strip()
        if not name or name.lower().startswith("unnamed:"):
            name = f"Sutun_{idx}"
            renamed += 1
        if name != str(col):
            renamed += 1
        new_names.append(unique_name(name, new_names))
    if renamed:
        warnings.append(
            f"{renamed} sütun adı otomatik olarak düzeltildi (boşluk kırpma, "
            "adsız sütunların yeniden adlandırılması)."
        )
    df = df.copy()
    df.columns = new_names
    return df, warnings


def load_dataframe(
    data: bytes,
    filename: str,
    separator: str | None = None,
    encoding: str | None = None,
    sheet_name: str | int | None = None,
) -> LoadResult:
    """Yüklenen dosya baytlarını DataFrame'e çevirir.

    Args:
        data: Dosya içeriği (bayt).
        filename: Orijinal dosya adı (uzantı tespiti için).
        separator: CSV için ayraç; None ise otomatik tespit edilir.
        encoding: CSV/JSON için encoding; None ise otomatik tespit edilir.
        sheet_name: Excel için sayfa adı/indeksi.

    Returns:
        Okunan DataFrame ve okuma parametrelerini içeren LoadResult.

    Raises:
        LoaderError: Dosya okunamaz, boş veya desteklenmiyorsa.
    """
    if not data:
        raise LoaderError("Dosya içeriği boş. Lütfen geçerli bir dosya yükleyin.")

    ext = detect_extension(filename)
    warnings: list[str] = []

    file_type = EXTENSION_TYPES[ext]
    used_sep: str | None = None
    used_enc: str | None = None
    used_sheet: str | int | None = None

    if file_type == "csv":
        df, used_sep, used_enc, warns = _read_csv_bytes(data, separator, encoding)
    elif file_type == "excel":
        df, used_sheet, warns = _read_excel_bytes(data, ext, sheet_name)
    elif file_type == "json":
        df, used_enc, warns = _read_json_bytes(data)
    else:  # parquet
        df, warns = _read_parquet_bytes(data)
    warnings.extend(warns)

    if df is None or df.shape[1] == 0:
        raise LoaderError("Dosyada okunabilir sütun bulunamadı.")

    df, col_warns = _normalize_columns(df)
    warnings.extend(col_warns)

    # Tamamen boş satırları at (dosya sonundaki boş satırlar yaygın).
    before = len(df)
    df = df.dropna(how="all")
    if len(df) < before:
        warnings.append(f"{before - len(df)} tamamen boş satır okuma sırasında atlandı.")

    df = df.reset_index(drop=True)

    if df.empty:
        raise LoaderError(
            "Dosya okundu ancak hiç veri satırı bulunamadı. "
            "Başlık satırı dışında veri olduğundan emin olun."
        )

    empty_cols = [str(c) for c in df.columns if df[c].isna().all()]
    if empty_cols:
        warnings.append(
            "Tamamen boş sütun(lar) bulundu: " + ", ".join(empty_cols[:5])
            + (" ..." if len(empty_cols) > 5 else "")
        )
    if df.shape[0] == 1:
        warnings.append(
            "Veri setinde tek satır var. İstatistik ve korelasyon analizleri "
            "anlamlı sonuç vermeyecektir."
        )

    return LoadResult(
        df=df,
        filename=str(filename),
        file_type=file_type,
        encoding=used_enc,
        separator=used_sep,
        sheet_name=used_sheet,
        warnings=warnings,
    )


def loading_code(result_meta: dict[str, Any]) -> str:
    """Dataset yükleme adımının Python kodunu üretir.

    Args:
        result_meta: ``file_type``, ``filename``, ``separator``, ``encoding``,
            ``sheet_name`` anahtarlarını içeren sözlük.

    Returns:
        ``pd.read_csv(...)`` / ``pd.read_excel(...)`` çağrısını içeren kod.
    """
    filename = result_meta.get("filename") or "dataset.csv"
    file_type = result_meta.get("file_type") or "csv"

    if file_type == "json":
        return "df = pd.read_json(" + col_literal(filename) + ")"
    if file_type == "parquet":
        return "df = pd.read_parquet(" + col_literal(filename) + ")"

    if file_type == "csv":
        args = [col_literal(filename)]
        sep = result_meta.get("separator")
        if sep and sep != ",":
            args.append(f"sep={py_literal(sep)}")
        enc = result_meta.get("encoding")
        if enc and enc not in ("utf-8", "utf-8-sig"):
            args.append(f"encoding={py_literal(enc)}")
        return "df = pd.read_csv(" + ", ".join(args) + ")"

    args = [col_literal(filename)]
    sheet = result_meta.get("sheet_name")
    if sheet is not None and sheet != 0:
        args.append(f"sheet_name={py_literal(sheet)}")
    return "df = pd.read_excel(" + ", ".join(args) + ")"
