# 📊 No-Code EDA Studio

> **Kod yazmadan analiz yap. Arkasındaki Python kodunu öğren. Notebook olarak indir.**

CSV, Excel, JSON veya Parquet dosyanızı yükleyin; veriyi profilleyin, temizleyin,
dönüştürün, görselleştirin, istatistiksel olarak inceleyin, pivot tablolar kurun ve
ML'e hazırlayın. Uygulama yaptığınız **her işlemin gerçek Python/Pandas kodunu üretir**
ve analizinizi `.py`, `.ipynb` veya tek dosyalık HTML rapor olarak indirmenize izin verir.

Bu yüzden proje sadece bir otomatik EDA aracı değil, aynı zamanda bir
**EDA + veri hazırlama + öğrenme platformudur**.

| | |
|---|---|
| **Hedef** | Excel kadar erişilebilir, Pandas kadar güçlü, modern SaaS kadar sade |
| **Teknoloji** | Python · Streamlit · Pandas · NumPy · Plotly · SciPy · scikit-learn |
| **Harici API** | Yok. Hiçbir LLM/AI servisi kullanılmaz, tüm analizler lokal çalışır |
| **Test** | 407 test (`pytest`), uçtan uca kullanıcı akışı dahil |

---

## 📸 Ekran görüntüleri

> Aşağıdaki alanlar yer tutucudur. `streamlit run app.py` ile uygulamayı açıp
> ekran görüntülerini `docs/` klasörüne ekleyebilirsiniz.

| Ekran | Görsel |
|---|---|
| Karşılama / veri yükleme | `docs/01-upload.png` |
| Genel bakış + kalite skoru | `docs/02-overview.png` |
| Dönüşümler (ölçekleme, encoding) | `docs/03-transform.png` |
| Grafik oluşturucu + öneriler | `docs/04-visualize.png` |
| Python kodu & notebook export | `docs/05-code.png` |

---

## ✨ Özellikler

### Veri içe aktarma
- **CSV, TXT, TSV, XLSX, XLS, JSON, JSONL, Parquet**
- CSV'de encoding (UTF-8 / CP1254 / ISO-8859-9 / Latin-1) ve ayraç (`,` `;` `tab` `|`) otomatik tespiti
- Excel'de birden fazla sayfa varsa sayfa seçimi
- JSON'da kayıt listesi, JSONL ve iç içe nesneler (nokta ile düzleştirme)
- Bozuk dosya, boş dosya, hatalı encoding ve tekrarlı sütun adlarında kullanıcı dostu davranış

### Genel bakış ve veri kalitesi
- Satır / sütun / hücre / bellek, boş hücre, tekrarlı satır, sayısal–kategorik–tarih–mantıksal sütun sayıları
- Önizleme (10 / 20 / 50 / 100 satır) ve sütun profili: tip, **tahmini rol**, boş %, tekil %, örnek değer
- Boş değer analizi (+ Plotly grafiği), tekrar analizi, seçili sütunlara göre tekrar kontrolü
- Sabit sütunlar, **neredeyse sabit** sütunlar, olası ID sütunları, yüksek kardinalite
- Tip tutarsızlıkları (sayısal/tarih görünen metin, karışık tip, mantıksal görünen sütun)
- Değer düzeyi sorunlar: baştaki/sondaki boşluk, çoklu boşluk, `-`/`NA` gibi yer tutucular, `inf`
- Sütun adı çakışmaları
- **Veri Kalite Skoru** (0–100, açıklanabilir 5 alt bileşen ve puan etkileri)

### Temizleme ve dönüşümler
| Grup | İşlemler |
|---|---|
| Boş değerler | satır sil · mean · median · mode · özel değer · ffill · bfill |
| Tekrarlar | tam tekrar · seçili sütunlara göre · first/last |
| Sütun | yeniden adlandır · sil · kopyala · sırala (reorder) |
| Tip dönüşümü | integer · float · string · category · boolean · datetime (biçim opsiyonel) |
| Metin temizleme | trim · küçük/BÜYÜK/Başlık harf · bul-değiştir · karakter sil · çoklu boşluk |
| Sayısal dönüşüm | log · log1p · sqrt · abs · round |
| Ölçekleme | Min-Max · Standart (z-score) · Robust (medyan/IQR) |
| Encoding | Label · One-Hot (drop_first) · Ordinal (sıra sizin) |
| Tarih özellikleri | yıl · çeyrek · ay · ay adı · gün · hafta günü · gün adı · saat · hafta sonu · hafta · yılın günü |
| Feature engineering | topla · çıkar · çarp · böl · oran (sıfıra bölme güvenli) |
| Binning | eşit genişlik · quantile · özel sınırlar (+ etiketler) |
| Satır | çoklu sütuna göre sıralama (artan/azalan) |
| Aykırı değer | IQR veya Z-Score ile satır silme |
| Filtre | filtreyi veri setine kalıcı uygulama |

Her işlem öncesi **etkilenecek satır/hücre sayısı** ve dönüşüm ön izlemesi (önce/sonra)
gösterilir; riskli işlemlerde uyarı verilir.

### Analiz
- **İstatistik**: count, missing, mean, median, mode, std, variance, min, Q1, Q2, Q3, max,
  range, IQR, **varyasyon katsayısı**, skewness, kurtosis, Shapiro-Wilk;
  kategorik için unique, mode, frekans, yüzde, **entropi/dengelilik**;
  tarih için min/max, aralık, yıl–çeyrek–ay–hafta günü dağılımları
- **İlişkiler**: Pearson · Spearman · **Kendall**, matris + heatmap, eşik üstü güçlü ilişkiler,
  iki sütun dağılım grafiği, **kategorik ilişkiler (Chi-Square + Cramér's V)**
- **Aykırı değerler**: IQR ve **Z-Score**, sınırlar, oran, kayıt önizleme, CSV indirme, silme
- **Pivot & GroupBy**: group by (count, nunique, sum, mean, median, min, max, std),
  pivot table (satır/sütun/değer/aggregation, toplamlar, 0 ile doldurma),
  **crosstab** (adet, satır %, sütun %, toplam %) → tablo + CSV + grafik
- **Smart Insights**: kural tabanlı, deterministik ve açıklanabilir bulgular
  (boş değer seviyeleri, tekrar, sabit sütun, olası ID, kategori dengesizliği,
  yüksek kardinalite, çarpıklık, yüksek korelasyon, aykırı değer yoğunluğu, `inf`, veri yapısı)

### Görselleştirme
13 grafik tipi: **Bar** (gruplu/yığılmış/yatay), **Line**, **Area**, **Scatter**, **Bubble**,
**Histogram**, **Box**, **Violin**, **Pie**, **Donut**, **Correlation Heatmap**,
**Kategori Heatmap (çapraz tablo)**, **Density Heatmap**.

- **Smart Chart Recommender**: sütun tiplerine göre öneri + tek tuşla oluşturma (AI yok)
- **Smart defaults**: grafik tipi seçilince form uygun sütunlarla dolu gelir
- Zorunlu alanlar görünür, geri kalanı "Gelişmiş ayarlar" içinde: renk, boyut, aggregation,
  sıralama, Top N, başlık, eksen etiketleri, legend
- Büyük veri setlerinde otomatik örnekleme ve kullanıcı bilgilendirmesi
- Filtreler (sayısal / metin / tarih / boş-dolu, **VE/VEYA** mantığı) ve hızlı Group By

### ML Hazırlığı (model eğitmez)
- Hedef sütuna göre problem tipi tahmini (ikili / çok sınıflı / regresyon)
- Eksik özellik, yüksek kardinalite, sabit sütun, **olası ID**, **olası veri sızıntısı (leakage)**,
  hedef sınıf dengesizliği tespiti ve öneriler
- No-code train/test ayarları (test oranı, random_state, stratify)
- scikit-learn **Pipeline + ColumnTransformer** kodu (imputation, scaling, encoding)
- Üretilen kod Python script ve Notebook export'una eklenir

### Kod üretimi ve export
- **Python Kodu** sayfası: Tüm Akış · Temizleme · Dönüşümler · Analiz · Görselleştirme · ML Hazırlığı
- **`.py` script** ve **`.ipynb` notebook** (gerçek `nbformat` v4, Jupyter/Colab uyumlu)
- Notebook bölümleri: Imports → Data Loading → Dataset Overview → Data Cleaning /
  Transformations / Feature Engineering → Statistical Analysis → Relationships →
  Outlier Analysis → Visualizations → ML Preparation → Insights
- **Veri export**: CSV · Excel (işlem geçmişi + bulgular ek sayfa) · **Parquet** · **JSON**
- **Analiz sonuçları**: korelasyon matrisi CSV, bulgular CSV/JSON, işlem geçmişi CSV, gruplama CSV
- **Grafikler**: interaktif Plotly HTML
- **HTML EDA raporu**: tek dosya; özet, kalite, skor, istatistik, korelasyon, bulgular, grafikler
  (tarayıcıdan "Yazdır → PDF" ile PDF'e çevrilebilir)

---

## 🔁 Kullanıcı akışı

```text
Veri yükle
  → Genel bakış → Veri kalitesi → Temizleme → Dönüşümler
  → İstatistik → Görselleştir → İlişkiler → Aykırı değerler → İçgörüler
  → Pivot & GroupBy → ML hazırlığı
  → İşlem geçmişi (undo / reset) → Python kodu → Dışa aktar (.py / .ipynb / veri / rapor)
```

Kenar çubuğu bu akışı dört grupta toplar: **VERİ · ANALİZ · GELİŞMİŞ · PROJE**.
**Gelişmiş mod** kapalıyken Dönüşümler, İlişkiler, Pivot ve ML Hazırlığı gizlenir;
günlük EDA için arayüz altı sayfaya iner.

---

## 🚀 Kurulum

Python 3.10+ gerekir (3.12 ile geliştirildi ve test edildi).

```bash
git clone <repo-url>
cd veri-eda
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
pip install -r requirements.txt
```

## ▶️ Çalıştırma

```bash
streamlit run app.py
```

Uygulama `http://localhost:8501` adresinde açılır. Hemen denemek için
**Örnek veri setiyle başla** düğmesini kullanabilir veya `sample_data/` içindeki
dosyaları yükleyebilirsiniz:

| Dosya | İçerik |
|---|---|
| `sample_data/ornek_musteriler.csv` | 407 satır, `;` ayraçlı; boş değer, tekrar, aykırı değer ve boş sütun içerir |
| `sample_data/ornek_satislar.xlsx` | 250 satırlık Excel sipariş tablosu |
| `sample_data/ornek_musteriler.json` | 60 kayıtlık JSON (kayıt listesi) |
| `sample_data/ornek_musteriler.parquet` | Aynı veri setinin Parquet sürümü |

---

## 🧪 Testler

```bash
python -m pytest
```

```bash
python -m pyflakes app.py core/*.py pages/*.py components/*.py utils/*.py
```

**407 test** şu alanları kapsar:

| Dosya | Kapsam |
|---|---|
| `tests/test_loader.py` | Encoding/ayraç tespiti, bozuk/boş dosya, Excel, JSON, Parquet |
| `tests/test_cleaner.py` | Tüm temizleme dönüşümleri, saflık (orijinal veri değişmez) |
| `tests/test_operations.py` | Operasyon kaydı, **replay determinizmi**, undo, reset |
| `tests/test_transformations.py` | Metin/sayısal/encoding/tarih/feature/binning + **kod eşdeğerliği** |
| `tests/test_analysis.py` | Filtreler (VE/VEYA), group by, istatistik, korelasyon, outlier, 13 grafik tipi, öneriler |
| `tests/test_pivot_ml_report.py` | Pivot, crosstab, ML hazırlığı, pipeline kodu, HTML rapor, Parquet/JSON export |
| `tests/test_insights.py` | Insight kuralları, determinizm, kalite skoru |
| `tests/test_code_generator.py` | Üretilen kodun **çalıştırılıp** sonucun doğrulanması |
| `tests/test_notebook_generator.py` | Geçerli `.ipynb`, bölümler, export fonksiyonları |
| `tests/test_app_smoke.py` | Streamlit `AppTest` ile **her sayfanın** render'ı + gerçek akışlar |
| `tests/test_edge_cases.py` | Boş/tek satır/sütunsuz veri, `inf`, Unicode, tırnaklı/tekrarlı sütun adları, sıfıra bölme |
| `tests/test_end_to_end.py` | **Tüm ürün akışı tek testte**: yükle → temizle → dönüştür → analiz → kod → notebook → export → undo → reset |

---

## 🧠 Mimari

```text
veri-eda/
├── app.py                      # Giriş noktası: gruplu navigasyon, tema, dataset başlığı
├── requirements.txt / pytest.ini / .streamlit/config.toml
│
├── pages/                      # Her sayfa bir render() fonksiyonu (st.navigation)
│   ├── upload.py  overview.py  data_quality.py  cleaning.py  transform.py
│   ├── statistics_page.py  visualization.py  relationships.py  outliers_page.py
│   ├── insights.py  pivot_page.py  ml_prep.py
│   └── history.py  code_viewer.py  export_page.py
│
├── core/                       # İş mantığı — Streamlit'ten tamamen bağımsız
│   ├── session.py              # Session State'in tek sahibi
│   ├── loader.py               # CSV/Excel/JSON/Parquet okuma + loading kodu
│   ├── profiler.py             # Veri kalitesi analizleri
│   ├── cleaner.py              # Saf temizleme dönüşümleri
│   ├── transformations.py      # Metin/sayısal/encoding/tarih/feature/binning
│   ├── operations.py           # Operation kaydı, replay, undo
│   ├── filters.py              # Filtre (VE/VEYA) ve group by motoru
│   ├── pivot.py                # Pivot table ve crosstab
│   ├── chart_builder.py        # 13 grafik tipi, öneriler, grafik kodu
│   ├── statistics.py           # Sayısal/kategorik/tarih istatistikleri
│   ├── correlations.py         # Pearson/Spearman/Kendall + Cramér's V
│   ├── outliers.py             # IQR ve Z-Score
│   ├── insights.py             # Kural tabanlı insight motoru
│   ├── data_quality_score.py   # 0-100 açıklanabilir skor
│   ├── ml_preparation.py       # ML hazırlık analizi + sklearn kod üretimi
│   ├── code_generator.py       # Operasyon → Python kodu (bölümlü)
│   ├── notebook_generator.py   # nbformat ile .ipynb
│   ├── report.py               # Tek dosyalık HTML EDA raporu
│   └── exporter.py             # CSV/XLSX/Parquet/JSON/.py/.ipynb/HTML bytes
│
├── components/                 # Tekrar kullanılabilir UI parçaları
│   ├── metric_cards.py  tables.py  messages.py  filter_builder.py
│
├── utils/
│   ├── dataframe_utils.py  validators.py  helpers.py  caching.py
│
├── sample_data/                # Örnek veri setleri
└── tests/                      # pytest paketi (407 test)
```

### Temel kararlar

1. **`original_df` asla değişmez.** Tüm dönüşümler `working_df` üzerinde uygulanır.
2. **Operation History tek doğruluk kaynağıdır.** Her işlem
   `{id, action, params, description, python_code}` olarak saklanır ve
   Undo · Reset · Replay · Code Generator · Notebook Generator bu yapıyı kullanır.
3. **Undo = replay.** Snapshot saklanmaz; `original_df` üzerinde kalan operasyonlar
   yeniden oynatılır. Bellek dostudur ve deterministiktir (testlerle doğrulanır).
4. **`core` katmanı Streamlit import etmez.** İş mantığı doğrudan test edilebilir;
   cache yalnızca `utils/caching.py` içinde tanımlanır.
5. **Üretilen kod çalıştırılabilir olmak zorundadır.** Sütun adları `repr` ile
   kaçışlanır (boşluk, tırnak, ters bölü, Unicode güvenli), veri değerleri koda
   gömülmez ve testler üretilen script'i gerçekten çalıştırıp sonucu karşılaştırır.
6. **Bölüm başlıkları sırayı bozmaz.** Kod/notebook bölümleri (Temizleme,
   Dönüşümler, Feature Engineering) kronolojik sırayı korur; kategori değişince
   yeni başlık eklenir — böylece export edilen kod uygulamadakiyle aynı sonucu verir.
7. **Hiçbir hata uygulamayı çökertmez.** Tüm sayfa gövdeleri `error_boundary` ile
   sarılır; iş mantığı `EdaError` türevleriyle kullanıcı dostu Türkçe mesaj üretir.
8. **Filtre semantiği:** değer karşılaştırmaları (`!=`, `içermez` dahil) boş değerleri
   sonuca **dahil etmez**; boş değerler için `boş` / `boş değil` operatörleri kullanılır.
9. **Ölçekleme/encoding pandas ile yapılır** (replay determinizmi ve bağımsız kod için);
   scikit-learn karşılıkları ML Hazırlığı pipeline kodunda üretilir.

---

## 🐍 Kod üretimi — örnek çıktı

```python
# ==========================================================
# Bu script No-Code EDA Studio tarafından otomatik üretildi.
# Kaynak dosya      : ornek_musteriler.csv
# Veri işleme adımı : 5
# Grafik sayısı     : 2
# ==========================================================

import pandas as pd
import numpy as np
import plotly.express as px

# --- Dataset Yükleme ---
df = pd.read_csv('ornek_musteriler.csv', sep=';')

# --- Veri Temizleme ---

# 1. gelir sütunundaki boş değerler medyan ile dolduruldu (45 hücre).
df['gelir'] = df['gelir'].fillna(df['gelir'].median())

# 2. Tüm sütunlar bazında tekrarlı satırlar silindi (7 satır).
df = df.drop_duplicates().reset_index(drop=True)

# --- Dönüşümler ---

# 3. gelir sütununa Standart ölçekleme (z-score) uygulandı; 'gelir_z' sütununa yazıldı.
col_mean = df['gelir'].mean()
col_std = df['gelir'].std(ddof=0)
df['gelir_z'] = (df['gelir'] - col_mean) / col_std

# --- Feature Engineering ---

# 4. uyelik_tarihi tarih sütunundan yeni özellikler üretildi: Yıl, Ay.
dt_series = pd.to_datetime(df['uyelik_tarihi'], errors='coerce')
df['uyelik_tarihi_year'] = dt_series.dt.year.astype('Int64')
df['uyelik_tarihi_month'] = dt_series.dt.month.astype('Int64')

# --- Görselleştirmeler ---

# Grafik 1: Bar Chart (Çubuk) - sehir bazında gelir (mean)
plot_df = (df.groupby(['sehir'], dropna=False)['gelir'].agg('mean').reset_index())
plot_df = plot_df.sort_values('gelir', ascending=False)
fig = px.bar(plot_df, x='sehir', y='gelir', title='sehir bazında gelir (mean)')
fig.show()
```

---

## 🔒 Gizlilik / veri notu

- Yüklediğiniz dosya **yalnızca uygulama oturumunun belleğinde** işlenir; uygulama
  dosyanızı diske veya veritabanına yazmaz.
- Hiçbir veri üçüncü bir servise, API'ye veya dil modeline gönderilmez — uygulamada
  harici AI entegrasyonu **yoktur**.
- Üretilen script ve notebook **kullanıcı verisi içermez**, yalnızca işlem kodlarını içerir.
- Oturum kapandığında veya veri sıfırlandığında bellekteki veri kaybolur.
- **Dürüst uyarı:** uygulamayı bir bulut platformuna (Streamlit Community Cloud,
  Hugging Face Spaces, kendi sunucunuz) deploy ederseniz, dosya yükleme sırasında veri
  o platformun altyapısından geçer ve platformun geçici dosya/bellek davranışları
  geçerli olur. Hassas veriler için uygulamayı **lokal** veya kendi kontrolünüzdeki
  bir ortamda çalıştırın.

---

## ⚠️ Metodolojik uyarılar

Uygulama bu uyarıları ilgili ekranlarda da gösterir:

- **Korelasyon nedensellik göstermez.**
- **Bir değerin istatistiksel olarak aykırı olması hatalı olduğu anlamına gelmez.**
- **Veri Kalite Skoru bilimsel veya evrensel bir standart değildir**; uygulama içi,
  açıklanabilir bir heuristik göstergedir.
- Olası ID sütunu, tip tespiti, veri sızıntısı şüphesi ve problem tipi tahmini
  **kural tabanlı tahminlerdir**; nihai kararı veri bağlamını bilen siz verirsiniz.
- ML Hazırlığı sayfası **model eğitmez** ve performans garantisi vermez.

---

## 🚢 Deploy

En hızlı yol **Streamlit Community Cloud**:

1. Projeyi GitHub'a push edin.
2. share.streamlit.io üzerinden repoyu bağlayın, ana dosya olarak `app.py` seçin.
3. Bağımlılıklar `requirements.txt` üzerinden otomatik kurulur.

Alternatifler: Hugging Face Spaces (Streamlit SDK), Render/Railway, ya da Docker
(`python:3.12-slim` + `pip install -r requirements.txt` +
`streamlit run app.py --server.port $PORT --server.address 0.0.0.0`).
Yükleme boyutu sınırı `.streamlit/config.toml` içindeki `maxUploadSize` ile ayarlanır.

---

## 🗺️ Roadmap

- Operasyon geçmişini kaydet/yükle (JSON proje dosyası) ve farklı veri setinde tekrar oynat
- Tarih serisi analizi: resample, hareketli ortalama, mevsimsellik
- Çoklu dosya / tablo birleştirme (join & concat arayüzü)
- İsteğe bağlı model eğitimi (baseline sınıflandırma/regresyon + metrikler)
- PDF rapor çıktısı ve rapor şablonu seçimi
- Türkçe / İngilizce arayüz desteği
- Polars motoru ile çok büyük dosya desteği

---

## 📄 Lisans

Bu proje eğitim ve kişisel kullanım için hazırlanmıştır. Dilediğiniz lisansı
(örn. MIT) ekleyerek yayınlayabilirsiniz.
