# Pemodelan Prediksi Suhu Permukaan Jawa
## Regresi Linear Multivariat (Normal Equation + Ridge) | Data CORDEX-CMIP5

> **Sumber data**: [CDS Copernicus](https://cds.climate.copernicus.eu/)  
> **Pendekatan**: Dari scratch — NumPy, Pandas, Matplotlib (tanpa sklearn/pytorch)  
> **Struktur**: Terinspirasi dari [aminst/wits](https://github.com/aminst/wits)

---

## Struktur Proyek

```
jawa-suhu-cordex/
│
├── data_pipeline.sh              ← Pipeline utama (Linux/macOS)
├── data_pipeline.ps1             ← Pipeline utama (Windows/PowerShell)
│
├── data/
│   ├── raw/
│   │   ├── historical/           ← NetCDF CORDEX historis (1976–2005)
│   │   ├── rcp45/                ← NetCDF RCP4.5 (2006–2050)
│   │   └── rcp85/                ← NetCDF RCP8.5 (2006–2045)
│   └── processed/
│       ├── features/             ← X_historical.csv (108 fitur hasil engineering)
│       ├── targets/              ← y_historical.csv (suhu target)
│       └── validation/           ← X/y_train/val/test.csv
│
├── direktori_tools_cleaning_data/
│   ├── cds_downloader.py         ← Download dari CDS API
│   ├── nc_extractor.py           ← Ekstrak NetCDF → CSV (subset Jawa)
│   ├── bias_corrector.py         ← EDCDF bias correction (NumPy scratch)
│   ├── feature_merger.py         ← Gabung CSV prediktor (ala wits)
│   ├── feature_engineer.py       ← Lag, rolling, seasonal encoding
│   ├── target_extractor.py       ← Ekstrak kolom target
│   └── data_splitter.py          ← Temporal train/val/test split
│
├── notebooks/
│   ├── 01_main_analysis.ipynb    ← Notebook lama (referensi/arsip)
│   └── 02_main_analysis.ipynb    ← Notebook utama (Multivariat, bersih)
│
├── results/
│   ├── figures/                  ← PNG output semua plot
│   ├── metrics/                  ← CSV evaluation report + prediksi
│   └── models/                   ← weights.npz, normalizer, meta.json
│
├── tmp_folder/                   ← Data temporari (raw CSV)
│
├── requirements.txt
└── README.md
```

---

## Cara Menjalankan

### 1. Setup Lingkungan

```bash
python -m venv AI
source AI/bin/activate         # Windows: AI\Scripts\activate
pip install -r requirements.txt
```

### 2. Konfigurasi CDS API

Daftar di [cds.climate.copernicus.eu](https://cds.climate.copernicus.eu/), lalu buat file:

```ini
# ~/.cdsapirc
url: https://cds.climate.copernicus.eu/api
key: YOUR_UID:YOUR_API_KEY
```

Pastikan sudah menerima **Terms of Use** dataset di website CDS sebelum download.

### 3. Jalankan Pipeline Penuh

```bash
# Linux / macOS
chmod +x data_pipeline.sh
./data_pipeline.sh

# Windows / PowerShell
.\data_pipeline.ps1
```

### 4. Jalankan Notebook Interaktif

```bash
cd notebooks
jupyter notebook 02_main_analysis.ipynb
```

---

## Metodologi

### 1. Pendekatan (Methodology & Rationale)

Proyek ini memecahkan masalah prediksi suhu dengan **Regresi Linear Multivariat** menggunakan solusi **closed-form (Normal Equation)** dengan regularisasi **Ridge (L2)**, diimplementasikan murni dengan NumPy tanpa bergantung pada framework tingkat tinggi (seperti scikit-learn).

**Mengapa pendekatan ini?**
- **Solusi Analitik (Deterministik)**: Normal Equation menghasilkan solusi optimal dalam satu langkah komputasi — tidak memerlukan iterasi, epoch, learning rate, atau seed acak. Hasil selalu sama.
- **Rumus**: `θ = (XᵀX + λI)⁻¹ Xᵀy`
- **Ridge Regularization**: Parameter `λ = 10⁻⁴` mencegah overfitting pada 108 fitur yang diengineering (lag, rolling, seasonal encoding).
- **Transparansi Prediktor**: Persamaan regresi linear menawarkan interpretasi langsung (Feature Importance) atas besaran pengaruh relatif masing-masing variabel prediktor terhadap target.
- **EDCDF Bias Correction**: Equidistant CDF Matching (Li et al., 2010) mengoreksi distribusi suhu simulasi agar sejalan dengan observasi.

### 2. Feature Engineering (108 Fitur)

Dari data mentah CORDEX-SEA, dilakukan engineering fitur secara komprehensif:

| Kategori | Detail |
|---|---|
| **Variabel dasar** | 15 variabel (suhu, curah hujan, tekanan, angin, radiasi, kelembaban, dll) |
| **Lag features** | Lag t-1, t-2, t-3, t-6, t-12 untuk setiap variabel |
| **Rolling mean** | Rolling average 3, 6, 12 bulan |
| **Seasonal encoding** | sin/cos dari bulan (menangkap siklus musiman) |
| **Anomali suhu** | Deviasi dari rata-rata klimatologi |

### 3. Validasi Hasil (Temporal Split)

Pada data deret waktu yang berurutan (*non-iid*), k-Fold Cross Validation standar akan menciptakan **data leakage**. Sebagai gantinya, digunakan **Walk-forward Temporal Split**:

| Set | Periode | Fungsi |
|---|---|---|
| **Training** | ≤ Tahun 2000 | Melatih bobot model |
| **Validation** | 2001–2003 | Evaluasi unseen terdekat |
| **Test** | 2004–2005 | Inferensi murni (out-of-sample) |

### 4. Metrik Evaluasi

Karena ini adalah tugas regresi (bukan klasifikasi), metrik yang digunakan:

- **RMSE (Root Mean Squared Error)** — Deviasi rata-rata dalam °C, memberi penalti pada outlier
- **MAE (Mean Absolute Error)** — Rerata simpangan absolut
- **R² (R-Squared)** — Proporsi varians yang dijelaskan model
- **MSE (Mean Squared Error)** — Kuadrat rata-rata error

*(Catatan: Nilai metrik aktual ada di `results/metrics/evaluation_report.csv`)*

---

### Data

| Aspek | Detail |
|---|---|
| Dataset | CORDEX South-East Asia (`projections-cordex-domains-single-levels`) |
| GCM Driver | MOHC-HadGEM2-ES (CMIP5) |
| RCM | SMHI-RCA4 |
| Resolusi | 0.22° × 0.22° (~25 km) |
| Periode | 1976–2005 (hist) + 2006–2050 (RCP4.5, RCP8.5) |
| Area | Pulau Jawa: lat[-8.8, -5.9] lon[105.1, 115.7] |

### Model: Regresi Linear Multivariat

```
y = X · w + b
Loss = MSE + λ‖w‖²  (Ridge regularization)
Solusi: θ = (XᵀX + λI)⁻¹ Xᵀy
```

**Keunggulan*:
- Satu langkah komputasi (tidak iteratif)
- Deterministik — hasil selalu reprodusibel
- Optimal secara matematis untuk ridge regression
- Cocok untuk dataset berukuran sedang (~216 sampel, 108 fitur)

### Preprocessing
1. **EDCDF Bias Correction** — Equidistant CDF Matching (Li et al., 2010)
2. **Z-score Normalization** — dari scratch NumPy
3. **Temporal Split** — Train ≤2000, Val 2001–2003, Test 2004–2005

---

## Referensi Utama (arXiv)

| Kode | Judul | Link |
|---|---|---|
| [CLIMATE-INV-ML] | Climate-Invariant Machine Learning | [arXiv:2112.08440](https://arxiv.org/abs/2112.08440) |
| [CORDEX-DL-PRED] | Regional Climate Model + DL Bias Correction | [arXiv:2504.19145](https://arxiv.org/html/2504.19145v2) |
| [TEMP-DOWNSCALING] | ML for Statistical Downscaling | [arXiv:1902.02865](https://arxiv.org/abs/1902.02865) |
| [CLIMATEBENCH] | ClimateBench: Benchmark for Climate Projections | [arXiv:2110.11676](https://arxiv.org/abs/2110.11676) |

Referensi lengkap → [`references/REFERENCES.md`](references/REFERENCES.md)

---

## Output yang Dihasilkan

```
results/figures/
  ├── eda_timeseries.png              ← EDA: time series variabel iklim
  ├── eda_correlation_matrix.png      ← Matriks korelasi Pearson
  ├── evaluation_result.png           ← Pred vs aktual, scatter, residual
  ├── feature_importance.png          ← Top 15 fitur
  └── proyeksi_suhu_jawa_2050.png     ← Proyeksi 1986–2050

results/metrics/
  ├── evaluation_report.csv           ← MSE, RMSE, MAE, R²
  └── test_predictions.csv            ← Aktual vs prediksi per bulan

results/models/
  ├── weights.npz                     ← Bobot model
  ├── normalizer_X.npz                ← Parameter Z-score
  ├── target_stats.npy                ← [mean, std] target
  ├── feature_importance.csv          ← Ranking fitur
  └── meta.json                       ← Metadata model
```
