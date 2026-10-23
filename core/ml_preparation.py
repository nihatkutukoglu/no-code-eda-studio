"""ML Hazırlığı: dataset'in modellemeye uygunluğunu analiz eder ve
scikit-learn kodu (train/test split + preprocessing pipeline) üretir.

Önemli: Bu modül **model eğitmez**. Amaç, kullanıcının veri setini modellemeye
hazırlarken karşılaşacağı sorunları (boş değerler, yüksek kardinalite, olası
sızıntı/ID sütunları, dengesiz hedef) önceden görmesi ve hazır bir pipeline
kodu alabilmesidir.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from utils import dataframe_utils as dfu
from utils.helpers import py_literal
from utils.validators import ValidationError

__all__ = [
    "TASK_LABELS",
    "SCALERS",
    "NUMERIC_IMPUTERS",
    "CATEGORICAL_IMPUTERS",
    "CATEGORICAL_ENCODERS",
    "MLReadiness",
    "detect_task_type",
    "analyze_readiness",
    "train_test_code",
    "pipeline_code",
    "full_ml_code",
]

#: Görev tipi etiketleri.
TASK_LABELS: dict[str, str] = {
    "binary_classification": "İkili sınıflandırma (binary classification)",
    "multiclass_classification": "Çok sınıflı sınıflandırma (multiclass)",
    "regression": "Regresyon (sayısal tahmin)",
    "unknown": "Belirsiz",
}

#: Ölçekleyici seçenekleri.
SCALERS: dict[str, str] = {
    "none": "Ölçekleme yok",
    "standard": "StandardScaler (z-score)",
    "minmax": "MinMaxScaler (0-1)",
    "robust": "RobustScaler (medyan/IQR)",
}

#: Sayısal imputation seçenekleri.
NUMERIC_IMPUTERS: dict[str, str] = {
    "none": "Doldurma yok",
    "mean": "Ortalama ile doldur",
    "median": "Medyan ile doldur",
    "constant": "Sabit değer (0) ile doldur",
}

#: Kategorik imputation seçenekleri.
CATEGORICAL_IMPUTERS: dict[str, str] = {
    "none": "Doldurma yok",
    "most_frequent": "En sık değer ile doldur",
    "constant": "'Bilinmiyor' ile doldur",
}

#: Kategorik encoder seçenekleri.
CATEGORICAL_ENCODERS: dict[str, str] = {
    "onehot": "One-Hot Encoding",
    "ordinal": "Ordinal Encoding",
}

#: Sınıflandırma sayılabilmesi için hedefteki maksimum tekil değer sayısı.
MAX_CLASSES = 20


@dataclass
class MLReadiness:
    """ML hazırlık analizi sonucu."""

    target: str
    task_type: str
    n_rows: int
    n_features: int
    numeric_features: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)
    datetime_features: list[str] = field(default_factory=list)
    features_with_missing: list[tuple[str, float]] = field(default_factory=list)
    high_cardinality: list[tuple[str, int]] = field(default_factory=list)
    constant_features: list[str] = field(default_factory=list)
    id_like_features: list[str] = field(default_factory=list)
    leakage_suspects: list[tuple[str, float]] = field(default_factory=list)
    target_missing: int = 0
    target_classes: dict[str, int] = field(default_factory=dict)
    imbalance_ratio: float = 0.0
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def task_label(self) -> str:
        """Görev tipinin kullanıcıya gösterilecek etiketi."""
        return TASK_LABELS.get(self.task_type, self.task_type)

    @property
    def is_ready(self) -> bool:
        """Modellemeye engel bir durum olup olmadığını söyler."""
        return not self.blockers

    def recommended_drops(self) -> list[str]:
        """Modelden çıkarılması önerilen sütunları döndürür."""
        drops = {
            *self.constant_features,
            *self.id_like_features,
            *[column for column, _ in self.leakage_suspects],
        }
        return sorted(drops)

    def summary_table(self) -> pd.DataFrame:
        """Bulguları tablo olarak döndürür."""
        records: list[dict[str, Any]] = []
        for column, ratio in self.features_with_missing:
            records.append(
                {
                    "Sütun": column,
                    "Bulgu": "Boş değer içeriyor",
                    "Detay": f"%{ratio:.1f} boş",
                    "Öneri": "Imputation uygulayın veya sütunu çıkarın",
                }
            )
        for column, nunique in self.high_cardinality:
            records.append(
                {
                    "Sütun": column,
                    "Bulgu": "Yüksek kardinalite",
                    "Detay": f"{nunique} farklı değer",
                    "Öneri": "One-Hot yerine Ordinal/Target encoding veya gruplama",
                }
            )
        for column in self.constant_features:
            records.append(
                {
                    "Sütun": column,
                    "Bulgu": "Sabit sütun",
                    "Detay": "Tek değer",
                    "Öneri": "Modelden çıkarın",
                }
            )
        for column in self.id_like_features:
            records.append(
                {
                    "Sütun": column,
                    "Bulgu": "Olası kimlik (ID) sütunu",
                    "Detay": "Neredeyse tüm değerler tekil",
                    "Öneri": "Modelden çıkarın",
                }
            )
        for column, score in self.leakage_suspects:
            records.append(
                {
                    "Sütun": column,
                    "Bulgu": "Olası veri sızıntısı (leakage)",
                    "Detay": f"Hedefle ilişki: {score:.2f}",
                    "Öneri": "Bu sütunun hedefi içerip içermediğini kontrol edin",
                }
            )
        return pd.DataFrame.from_records(
            records, columns=["Sütun", "Bulgu", "Detay", "Öneri"]
        )


def detect_task_type(series: pd.Series) -> str:
    """Hedef sütununa göre olası problem tipini tahmin eder.

    Args:
        series: Hedef sütunu.

    Returns:
        ``binary_classification``, ``multiclass_classification``,
        ``regression`` veya ``unknown``.
    """
    clean = series.dropna()
    if clean.empty:
        return "unknown"
    nunique = dfu.safe_nunique(clean)
    if nunique <= 1:
        return "unknown"
    kind = dfu.column_kind(clean)
    if kind == "boolean" or nunique == 2:
        return "binary_classification"
    if kind == "numeric":
        # Az sayıda tekil tam sayı değeri genelde sınıf etiketidir.
        is_integer_like = bool((clean.dropna() % 1 == 0).all())
        if nunique <= MAX_CLASSES and is_integer_like and nunique / len(clean) < 0.05:
            return "multiclass_classification"
        return "regression"
    if kind == "datetime":
        return "unknown"
    if nunique <= MAX_CLASSES:
        return "multiclass_classification"
    return "unknown"


def _is_id_like(df: pd.DataFrame, column: str, ratio_threshold: float = 0.99) -> bool:
    """Sütunun gerçekten kimlik (ID) sütunu olup olmadığını sıkı kurallarla tahmin eder.

    ML bağlamında yanlış pozitif pahalıdır: neredeyse tekil olan sürekli bir
    ölçüm (gelir, satış) ID değildir. Bu yüzden eşik yükseltilir ve ondalıklı
    sayısal sütunlar ID sayılmaz.
    """
    series = df[column].dropna()
    if series.empty:
        return False
    ratio = dfu.safe_nunique(series) / len(series)
    if ratio < ratio_threshold:
        return False
    if column in dfu.numeric_columns(df):
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if numeric.empty:
            return False
        is_integer_like = bool((numeric % 1 == 0).all())
        if not is_integer_like:
            return False
        # Tam sayı ve neredeyse tüm değerler tekil: ardışık/benzersiz kimlik sinyali.
        return bool(ratio >= 0.999)
    return True


def _leakage_score(df: pd.DataFrame, feature: str, target: str, task_type: str) -> float:
    """Bir özelliğin hedefle anormal yüksek ilişkisini ölçer (0-1)."""
    numeric_columns = dfu.numeric_columns(df)
    if task_type == "regression" and feature in numeric_columns and target in numeric_columns:
        pair = df[[feature, target]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(pair) < 3:
            return 0.0
        value = pair[feature].corr(pair[target])
        return float(abs(value)) if pd.notna(value) else 0.0

    # Sınıflandırmada: özellik hedefi birebir belirliyorsa sızıntı şüphesi.
    grouped = df[[feature, target]].dropna()
    if grouped.empty:
        return 0.0

    nunique = dfu.safe_nunique(grouped[feature])
    # Her grupta ortalama en az 3 kayıt olmalı; aksi halde "saflık" ölçümü
    # anlamsızdır (sürekli değişkenlerde her değer kendi grubu olur ve
    # sonuç her zaman 1.0 çıkar).
    if nunique <= 1 or len(grouped) / nunique < 3:
        return 0.0

    try:
        purity = (
            grouped.groupby(feature, observed=True)[target]
            .agg(lambda values: values.value_counts(normalize=True).iloc[0])
            .mean()
        )
    except (TypeError, ValueError, IndexError):
        return 0.0
    return float(purity) if pd.notna(purity) else 0.0


def analyze_readiness(
    df: pd.DataFrame,
    target: str,
    features: list[str] | None = None,
    leakage_threshold: float = 0.98,
) -> MLReadiness:
    """Dataset'in ML'e hazırlık durumunu analiz eder.

    Args:
        df: Kaynak DataFrame (``working_df``).
        target: Hedef sütun.
        features: Kullanılacak özellikler (None ise hedef dışındaki tüm sütunlar).
        leakage_threshold: Sızıntı şüphesi için ilişki eşiği.

    Returns:
        Bulguları ve uyarıları içeren ``MLReadiness``.

    Raises:
        ValidationError: Hedef sütun yok veya veri yetersizse.
    """
    if df is None or df.empty:
        raise ValidationError("ML hazırlığı için veri seti gerekli.")
    if target not in df.columns:
        raise ValidationError(f"'{target}' hedef sütunu bulunamadı.")

    selected = [c for c in (features or [str(c) for c in df.columns]) if c != target]
    missing = [c for c in selected if c not in df.columns]
    if missing:
        raise ValidationError("Şu sütunlar bulunamadı: " + ", ".join(map(str, missing)))
    if not selected:
        raise ValidationError("En az bir özellik (feature) sütunu seçmelisiniz.")

    rows = int(len(df))
    target_series = df[target]
    task_type = detect_task_type(target_series)

    report = MLReadiness(
        target=str(target),
        task_type=task_type,
        n_rows=rows,
        n_features=len(selected),
        numeric_features=[c for c in selected if c in dfu.numeric_columns(df)],
        categorical_features=[c for c in selected if c in dfu.categorical_columns(df)],
        datetime_features=[c for c in selected if c in dfu.datetime_columns(df)],
        target_missing=int(target_series.isna().sum()),
    )

    # Hedef kontrolü
    if task_type == "unknown":
        report.blockers.append(
            f"'{target}' sütunundan problem tipi anlaşılamadı. Hedef; sayısal bir "
            "ölçüm ya da sınırlı sayıda sınıf içeren bir sütun olmalıdır."
        )
    if report.target_missing:
        ratio = report.target_missing / rows * 100
        message = (
            f"Hedef sütunda {report.target_missing} boş değer var (%{ratio:.1f}). "
            "Bu satırlar modellemede kullanılamaz."
        )
        (report.blockers if ratio >= 50 else report.warnings).append(message)

    if task_type in ("binary_classification", "multiclass_classification"):
        counts = target_series.dropna().astype("string").value_counts()
        report.target_classes = {str(k): int(v) for k, v in counts.items()}
        if not counts.empty:
            report.imbalance_ratio = float(counts.iloc[0] / counts.sum())
            if report.imbalance_ratio >= 0.90:
                report.warnings.append(
                    f"Hedef çok dengesiz: '{counts.index[0]}' sınıfı "
                    f"%{report.imbalance_ratio * 100:.1f} pay alıyor. "
                    "Stratify, class_weight veya örnekleme yöntemlerini değerlendirin."
                )
            smallest = int(counts.iloc[-1])
            if smallest < 2:
                report.warnings.append(
                    f"En küçük sınıfta yalnızca {smallest} örnek var; "
                    "stratified split yapılamaz."
                )

    if rows < 50:
        report.warnings.append(
            f"Veri setinde {rows} satır var. Modelleme için oldukça az; "
            "sonuçlar güvenilir olmayabilir."
        )

    # Özellik kontrolleri
    for column in selected:
        series = df[column]
        missing_count = int(series.isna().sum())
        if missing_count:
            report.features_with_missing.append((column, missing_count / rows * 100))
        if dfu.safe_nunique(series) <= 1:
            report.constant_features.append(column)

    report.id_like_features = [c for c in selected if _is_id_like(df, c)]

    for column, nunique, _ratio in dfu.high_cardinality_columns(df, 0.3, 20):
        if column in selected and column not in report.id_like_features:
            report.high_cardinality.append((column, nunique))

    if task_type != "unknown":
        for column in selected:
            if column in report.constant_features or column in report.id_like_features:
                continue
            score = _leakage_score(df, column, target, task_type)
            if score >= leakage_threshold:
                report.leakage_suspects.append((column, score))

    if report.datetime_features:
        report.notes.append(
            "Tarih sütunları doğrudan modele verilmez. Dönüşümler sayfasından "
            "yıl/ay/hafta günü gibi özellikler üretmeniz önerilir: "
            + ", ".join(report.datetime_features)
        )
    if not report.numeric_features and not report.categorical_features:
        report.blockers.append("Modellemeye uygun özellik sütunu bulunamadı.")
    report.notes.append(
        "Bu analiz kural tabanlıdır; model eğitimi yapılmaz ve performans "
        "garantisi vermez."
    )
    return report


# ---------------------------------------------------------------------------
# Kod üretimi
# ---------------------------------------------------------------------------


def train_test_code(
    target: str,
    features: list[str],
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = False,
    df_name: str = "df",
) -> str:
    """train_test_split kodunu üretir."""
    lines = [
        "from sklearn.model_selection import train_test_split",
        "",
        f"feature_columns = {py_literal([str(c) for c in features])}",
        f"target_column = {py_literal(str(target))}",
        "",
        f"model_df = {df_name}[feature_columns + [target_column]].dropna(subset=[target_column])",
        "X = model_df[feature_columns]",
        "y = model_df[target_column]",
        "",
    ]
    args = [
        "X",
        "y",
        f"test_size={float(test_size)}",
        f"random_state={int(random_state)}",
    ]
    if stratify:
        args.append("stratify=y")
    lines.append("X_train, X_test, y_train, y_test = train_test_split(")
    lines.append("    " + ", ".join(args))
    lines.append(")")
    lines.append("print(X_train.shape, X_test.shape)")
    return "\n".join(lines)


def pipeline_code(
    numeric_features: list[str],
    categorical_features: list[str],
    numeric_imputer: str = "median",
    numeric_scaler: str = "standard",
    categorical_imputer: str = "most_frequent",
    categorical_encoder: str = "onehot",
) -> str:
    """scikit-learn ``Pipeline`` + ``ColumnTransformer`` kodunu üretir.

    Args:
        numeric_features: Sayısal özellikler.
        categorical_features: Kategorik özellikler.
        numeric_imputer: ``NUMERIC_IMPUTERS`` anahtarı.
        numeric_scaler: ``SCALERS`` anahtarı.
        categorical_imputer: ``CATEGORICAL_IMPUTERS`` anahtarı.
        categorical_encoder: ``CATEGORICAL_ENCODERS`` anahtarı.

    Returns:
        Çalıştırılabilir pipeline kodu.

    Raises:
        ValidationError: Hiç özellik seçilmemişse veya seçenek bilinmiyorsa.
    """
    if not numeric_features and not categorical_features:
        raise ValidationError("Pipeline için en az bir özellik seçmelisiniz.")
    for value, options, label in (
        (numeric_imputer, NUMERIC_IMPUTERS, "sayısal imputation"),
        (numeric_scaler, SCALERS, "ölçekleyici"),
        (categorical_imputer, CATEGORICAL_IMPUTERS, "kategorik imputation"),
        (categorical_encoder, CATEGORICAL_ENCODERS, "kategorik encoder"),
    ):
        if value not in options:
            raise ValidationError(f"Desteklenmeyen {label}: {value}")

    imports = [
        "from sklearn.compose import ColumnTransformer",
        "from sklearn.pipeline import Pipeline",
    ]
    if numeric_imputer != "none" or categorical_imputer != "none":
        imports.append("from sklearn.impute import SimpleImputer")
    scaler_classes = {
        "standard": "StandardScaler",
        "minmax": "MinMaxScaler",
        "robust": "RobustScaler",
    }
    encoder_classes = {"onehot": "OneHotEncoder", "ordinal": "OrdinalEncoder"}
    preprocessing_imports = []
    if numeric_scaler != "none":
        preprocessing_imports.append(scaler_classes[numeric_scaler])
    preprocessing_imports.append(encoder_classes[categorical_encoder])
    imports.append(
        "from sklearn.preprocessing import " + ", ".join(sorted(set(preprocessing_imports)))
    )

    lines = [*imports, ""]
    lines.append(f"numeric_features = {py_literal([str(c) for c in numeric_features])}")
    lines.append(
        f"categorical_features = {py_literal([str(c) for c in categorical_features])}"
    )
    lines.append("")

    transformers: list[str] = []
    if numeric_features:
        steps = []
        if numeric_imputer == "constant":
            steps.append("('imputer', SimpleImputer(strategy='constant', fill_value=0))")
        elif numeric_imputer != "none":
            steps.append(f"('imputer', SimpleImputer(strategy='{numeric_imputer}'))")
        if numeric_scaler != "none":
            steps.append(f"('scaler', {scaler_classes[numeric_scaler]}())")
        if not steps:
            steps.append("('passthrough', 'passthrough')")
        lines.append("numeric_pipeline = Pipeline(steps=[")
        for step in steps:
            lines.append(f"    {step},")
        lines.append("])")
        lines.append("")
        transformers.append("('num', numeric_pipeline, numeric_features)")

    if categorical_features:
        steps = []
        if categorical_imputer == "constant":
            steps.append(
                "('imputer', SimpleImputer(strategy='constant', fill_value='Bilinmiyor'))"
            )
        elif categorical_imputer != "none":
            steps.append(f"('imputer', SimpleImputer(strategy='{categorical_imputer}'))")
        if categorical_encoder == "onehot":
            steps.append(
                "('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))"
            )
        else:
            steps.append(
                "('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', "
                "unknown_value=-1))"
            )
        lines.append("categorical_pipeline = Pipeline(steps=[")
        for step in steps:
            lines.append(f"    {step},")
        lines.append("])")
        lines.append("")
        transformers.append("('cat', categorical_pipeline, categorical_features)")

    lines.append("preprocessor = ColumnTransformer(")
    lines.append("    transformers=[")
    for transformer in transformers:
        lines.append(f"        {transformer},")
    lines.append("    ],")
    lines.append("    remainder='drop',")
    lines.append(")")
    lines.append("")
    lines.append("# Eğitim verisine uygula (model seçimi size ait)")
    lines.append("X_train_prepared = preprocessor.fit_transform(X_train)")
    lines.append("X_test_prepared = preprocessor.transform(X_test)")
    lines.append("print(X_train_prepared.shape, X_test_prepared.shape)")
    return "\n".join(lines)


def full_ml_code(
    target: str,
    features: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = False,
    numeric_imputer: str = "median",
    numeric_scaler: str = "standard",
    categorical_imputer: str = "most_frequent",
    categorical_encoder: str = "onehot",
    drop_columns: list[str] | None = None,
) -> str:
    """ML hazırlık bölümünün tam kodunu (split + pipeline) üretir."""
    blocks: list[str] = []
    if drop_columns:
        blocks.append(
            "# Modelden çıkarılması önerilen sütunlar\n"
            f"df = df.drop(columns={py_literal([str(c) for c in drop_columns])}, errors='ignore')"
        )
    blocks.append(
        train_test_code(target, features, test_size, random_state, stratify)
    )
    blocks.append(
        pipeline_code(
            numeric_features,
            categorical_features,
            numeric_imputer,
            numeric_scaler,
            categorical_imputer,
            categorical_encoder,
        )
    )
    return "\n\n".join(blocks)
