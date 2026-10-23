"""Test fixture'ları: proje kökünü sys.path'e ekler ve örnek veri üretir."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def messy_df() -> pd.DataFrame:
    """Boş değer, tekrar, sabit sütun, aykırı değer ve metin sayı içeren veri."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 6],
            "ad": ["Ali", "Ayse", "Veli", "Zeynep", None, "Can", "Can"],
            "gelir": [1000.0, 2000.0, np.nan, 2500.0, 3000.0, 999999.0, 999999.0],
            "sehir": ["Ankara", "Izmir", "Ankara", "Ankara", "Izmir", "Bursa", "Bursa"],
            "sabit": ["X"] * 7,
            "tarih_metin": [
                "2024-01-01",
                "2024-02-01",
                "2024-03-01",
                "2024-04-01",
                "2024-05-01",
                "2024-06-01",
                "2024-06-01",
            ],
            "sayi_metin": ["1,5", "2,5", "3", "4", "5", "6", "6"],
            "bos": [None] * 7,
        }
    )


@pytest.fixture
def numeric_df() -> pd.DataFrame:
    """Korelasyon ve istatistik testleri için sayısal veri."""
    rng = np.random.default_rng(7)
    x = rng.normal(100, 15, 200)
    return pd.DataFrame(
        {
            "x": x,
            "y": x * 2 + rng.normal(0, 1, 200),  # x ile neredeyse tam korelasyon
            "z": rng.normal(50, 5, 200),
            "kategori": rng.choice(["A", "B", "C"], 200),
        }
    )


@pytest.fixture
def dataset_meta() -> dict[str, object]:
    """Kod üretimi testleri için dosya bilgisi."""
    return {
        "filename": "musteriler.csv",
        "file_type": "csv",
        "separator": ";",
        "encoding": "utf-8",
        "sheet_name": None,
    }
