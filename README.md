# Pemodelan Prediksi Suhu Permukaan Jawa
## Regresi Multivariat + EXDM Optimizer | Data CORDEX-CMIP5

> **Sumber data**: [CDS Copernicus](https://cds.climate.copernicus.eu/)  
> **Pendekatan**: Dari scratch — NumPy, Pandas, Matplotlib (tanpa sklearn/pytorch)  
> **Struktur**: Terinspirasi dari [aminst/wits](https://github.com/aminst/wits)

---

## Struktur Proyek

```
jawa-suhu-cordex/
│
├── data_pipeline.sh              ← Pipeline utama (jalankan ini)
│
├── data/
│   ├── raw/
│   │   ├── historical/           ← NetCDF CORDEX historis (1976–2005)
│   │   ├── rcp45/                ← NetCDF RCP4.5 (2006–2100)
│   │   └── rcp85/                ← NetCDF RCP8.5 (2006–2100)
│   └── processed/
│       ├── features/             ← X_historical.csv (fitur hasil engineering)
│       ├── targets/              ← y_historical.csv (suhu target)
│       └── validation/           ← X/y_train/val/test.csv
│
├── tools/
│   ├── cds_downloader.py         ← Download dari CDS API
│   ├── nc_extractor.py           ← Ekstrak NetCDF → CSV (subset Jawa)
│   ├── bias_corrector.py         ← EDCDF bias correction (NumPy scratch)
│   ├── feature_merger.py         ← Gabung CSV prediktor (ala wits)
│   ├── feature_engineer.py       ← Lag, rolling, seasonal encoding
│   ├── target_extractor.py       ← Ekstrak kolom target
│   ├── data_splitter.py          ← Temporal train/val/test split
│   ├── train_model.py            ← EXDM Optimizer + Regresi (NumPy scratch)
│   ├── evaluator.py              ← Metrik: RMSE, NSE, R², MAE, MBE
│   ├── predictor.py              ← Prediksi RCP masa depan
│   └── visualizer.py             ← Semua plot matplotlib
│
├── notebooks/
│   └── 01_main_analysis.ipynb    ← Notebook interaktif lengkap
│
├── results/
│   ├── figures/                  ← PNG output semua plot
│   ├── metrics/                  ← CSV evaluation report + prediksi
│   └── models/                   ← weights.npz, normalizer, meta.json
│
├── references/
│   └── REFERENCES.md             ← Semua sitasi: arXiv, jurnal, software
│
├── requirements.txt
└── README.md
```

---

## Cara Menjalankan

### 1. Setup Lingkungan

```bash
python -m venv env
source env/bin/activate         # Windows: env\Scripts\activate
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
chmod +x data_pipeline.sh
./data_pipeline.sh
```

### 4. Atau Jalankan Notebook Interaktif

```bash
cd notebooks
jupyter notebook 01_main_analysis.ipynb
```

---

## Metodologi

### 1. Metodologi (Methodology & Rationale)
Proyek ini memecahkan masalah prediksi suhu dengan **Regresi Multivariat** dan perhitungan aljabar linear murni berbasis NumPy, tanpa bergantung pada framework tingkat tinggi (seperti scikit-learn). Alasan pemilihan arsitektur ini:
- **Transparansi Prediktor**: Persamaan regresi linear menawarkan interpretasi langsung (Feature Importance) atas besaran pengaruh relatif masing-masing variabel prediktor (angin, radiasi, suhu masa lalu) terhadap target. Model yang *explainable* sangat penting dalam klimatologi.
- **EDCDF Bias Correction**: Output simulasi iklim global dan regional (GCM/RCM) sering berdeviasi secara sistematis dari titik stasiun meteorologi observasi. *Equidistant CDF Matching* (Li et al., 2010) digunakan. Metode ini mentransformasi distribusi peluang suhu raw-model agar secara keseragaman (*equidistant*) sejalan dengan observasi asli (ERA5), menjaga ekspektasi dan rentang pada proyeksi skenario masa depan RCP.
- **EXDM Optimizer**: Modul optimasi kustom *Exponential Decay Momentum* (EXDM). Data suhu iklim memiliki siklus musiman, *noise* tingkat tinggi, dan fitur yang meluruh. EXDM menggabungkan *momentum* (mencegah osilasi), *squared gradient accumulator* (skalasi gradien dinamis ala RMSProp), dan *exponential learning rate decay* untuk pembaruan gradien yang stabil, menghindari permasalahan numerik saat *gradient descent* berhadapan dengan *lag* fitur panjang.

### 2. Validasi Hasil (Result Validation)
Lebih baik dari k-Fold Cross Validation biasa? Pada data deret waktu yang saling berurutan (*non-iid data*), menggunakan instrumen statistik konvensional seperti *K-Fold Cross Validation* secara serampangan akan menciptakan kebocoran informasi masa depan ke masa lalu (*data leakage* melalui *lag/rolling features*). Sebagai solusinya, pipeline ini menggunakan **Walk-forward Temporal Split**:
- **Traning Set (≤ Tahun 2000):** Algoritma melatih bobot *w* berdasarkan pola masa lampau.
- **Validation Set (Tahun 2001–2003):** Mengevaluasi titik temu (*convergence*) pada skenario unseen terdekat. Fitur *Early Stopping* diaktifkan di sini—optimasi EXDM terhenti secara prematur apabila `val_loss` tidak membaik selama sekian *epoch* (kesabaran 30 iterasi), untuk menghindari *overfitting*.
- **Test Set (Tahun 2004–2005):** Inferensi komplit, mengekstraksi validasi metrik murni (out-of-sample).

### 3. Hasil & Metrik Evaluasi (Results & Metrics)
Dikarenakan objektif utama adalah memprediksi fluktuasi variabel temperatur dalam skala Celcius yang kontinu (Tugas Regresi), metrik bernuansa klasifikasi (seperti Akurasi, Presisi, atau Recall) secara esensi tidak kompatibel. Sebagai pengganti, model memvalidasi keandalannya menggunakan armada metrik regresi berstandar geofisika iklim:
- **RMSE (Root Mean Squared Error):** Metrik primer yang mengukur deviasi rata-rata hasil regresi dalam besaran derajat aktual (°C). Menguadratkan galat memberi penalti lebih kuat pada outlier (prediksi salah arah ekstrem).
- **MAE (Mean Absolute Error):** Rerata jarak deviasi absolut harian, merefleksikan nilai mutlak simpangan secara stabil tanpa terlalu dipengaruhi observasi yang ganjil.
- **R² (R-Squared):** Mengukur seberapa besar proporsi dan tendensi dari total keragaman temperatur rata-rata berhasil dijelaskan oleh model fitur yang disusun.
- **NSE (Nash-Sutcliffe Efficiency):** Indikator esensial yang distandardisasi bagi skenario hidrologi/iklim. Apabila `NSE > 0`, kemampuan prediksi model jauh lebih cerdas ketimbang sekadar menebak memakai konstan nilai rata-rata empiris masa lampau (klimatologi statis). 
- **MBE (Mean Bias Error):** Diagnosa tendensi arah, menentukan apakah performa jaringan cenderung memprediksi secara konsisten melampaui ukuran sesungguhnya (*overpredicting / bias positif*) atau meremehkan (*underpredicting*). 

*(Catatan: Nilai metrik kuantitatif aktual dilaporkan otomatis ke dalam format CSV pada `results/metrics/evaluation_report.csv` usai menjalankan skrip `./data_pipeline.sh`)*

---

### Data
| Aspek | Detail |
|---|---|
| Dataset | CORDEX South-East Asia (`projections-cordex-domains-single-levels`) |
| GCM Driver | MOHC-HadGEM2-ES (CMIP5) |
| RCM | SMHI-RCA4 |
| Resolusi | 0.22° × 0.22° (~25 km) |
| Periode | 1976–2005 (hist) + 2006–2100 (RCP4.5, RCP8.5) |
| Area | Pulau Jawa: lat[-8.8, -5.9] lon[105.1, 115.7] |

### Variabel Prediktor (X)
- Suhu 2m lag (t-1, t-2, t-3, t-6, t-12)
- Rolling mean suhu (3, 6, 12 bulan)
- Curah hujan + lag (t-1, t-3, t-6)
- Tekanan permukaan + lag
- Kelembaban relatif + lag
- Kecepatan dan arah angin
- Radiasi surya
- Seasonal encoding (sin/cos bulan)
- Anomali suhu (deviasi dari klimatologi)

### Preprocessing
1. **EDCDF Bias Correction** — Equidistant CDF Matching (Li et al., 2010)
2. **Z-score Normalization** — dari scratch NumPy
3. **Temporal Split** — Train ≤2000, Val 2001–2003, Test 2004–2005

### Model: Regresi Linear Multivariat
```
y = X · w + b
Loss = MSE + λ‖w‖²  (Ridge regularization)
```

### EXDM Optimizer
Exponential Decay Momentum — menggabungkan:
- **Momentum** (β₁ = 0.9) untuk stabilitas arah gradien
- **Adaptive accumulator** (β₂ = 0.999) — seperti RMSProp
- **Exponential decay** learning rate: ηₜ = η₀ · exp(−λt)
- **Bias correction** seperti Adam

```
vₜ  = β₁·vₜ₋₁ + (1-β₁)·∇L
sₜ  = β₂·sₜ₋₁ + (1-β₂)·∇L²
ηₜ  = η₀ · exp(−λ·t)
wₜ  = wₜ₋₁ − ηₜ · v̂ₜ / (√ŝₜ + ε)
```

### Metrik Evaluasi
- **RMSE** — Root Mean Squared Error (°C)
- **NSE** — Nash-Sutcliffe Efficiency (standard hidrologi/iklim)
- **R²** — Koefisien determinasi
- **MAE** — Mean Absolute Error
- **MBE** — Mean Bias Error
- **Pearson r** — Korelasi


---

## Referensi Utama (arXiv)

| Kode | Judul | Link |
|---|---|---|
| [CLIMATE-INV-ML] | Climate-Invariant Machine Learning | [arXiv:2112.08440](https://arxiv.org/abs/2112.08440) |
| [CORDEX-DL-PRED] | Regional Climate Model + DL Bias Correction | [arXiv:2504.19145](https://arxiv.org/html/2504.19145v2) |
| [TEMP-DOWNSCALING] | ML for Statistical Downscaling | [arXiv:1902.02865](https://arxiv.org/abs/1902.02865) |
| [CLIMATEBENCH] | ClimateBench: Benchmark for Climate Projections | [arXiv:2110.11676](https://arxiv.org/abs/2110.11676) |
| [ADAM] | Adam Optimizer | [arXiv:1412.6980](https://arxiv.org/abs/1412.6980) |
| [LR-DECAY] | Optimization for Large-Scale ML | [arXiv:1606.04838](https://arxiv.org/abs/1606.04838) |

Referensi lengkap → [`references/REFERENCES.md`](references/REFERENCES.md)

---

## Output yang Dihasilkan

```
results/figures/
  ├── eda_timeseries.png              ← EDA: time series 7 variabel
  ├── eda_correlation_matrix.png      ← Matriks korelasi Pearson
  ├── bias_correction_edcdf.png       ← CDF sebelum/sesudah koreksi
  ├── training_convergence.png        ← Kurva loss EXDM
  ├── evaluation_result.png           ← Pred vs aktual, scatter, residual
  ├── feature_importance.png          ← Top 15 fitur
  └── proyeksi_suhu_jawa_2100.png     ← Proyeksi 1976–2100

results/metrics/
  ├── evaluation_report.csv           ← RMSE, NSE, R², MAE, MBE, r
  ├── test_predictions.csv            ← Aktual vs prediksi per bulan
  ├── pred_rcp45.csv                  ← Proyeksi RCP4.5
  └── pred_rcp85.csv                  ← Proyeksi RCP8.5

results/models/
  ├── weights.npz                     ← Bobot model + loss history
  ├── normalizer_X.npz                ← Parameter Z-score
  ├── target_stats.npy                ← [mean, std] target
  ├── feature_importance.csv          ← Ranking fitur
  ├── val_metrics.csv                 ← Metrik validasi
  └── meta.json                       ← Hyperparameter model
```
