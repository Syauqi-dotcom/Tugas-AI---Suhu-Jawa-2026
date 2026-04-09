# Pemodelan Prediksi Suhu Permukaan Jawa
## Regresi Linear Multivariat (Normal Equation + Ridge)

> **Sumber data**: [CDS Copernicus](https://cds.climate.copernicus.eu/)  
> **Penjelasan Singkat**: Proyek ini memecahkan masalah prediksi iklim dengan solusi analitik (deterministik) menggunakan Regresi Linear Multivariat (Normal Equation + Ridge) pada data CORDEX-CMIP5.

---

## Struktur Proyek

```
jawa-suhu-cordex/
│
├── data_pipeline.sh              ← Pipeline data
│
├── data/
│   ├── raw/
│   │   ├── historical/           ← NetCDF CORDEX historis (1986–2005)
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
│   ├── workspace.ipynb    ← Notebook Model
│
├── results/
│   ├── figures/                  ← PNG output semua plot
│   ├── metrics/                  ← CSV evaluation report + prediksi
│   └── models/                   ← weights.npz, normalizer, meta.json
│
│
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
```

### 4. Jalankan Notebook Interaktif

```bash
cd notebooks
jupyter notebook workspace.ipynb
```

---

## Metodologi

### **1. Bias Handling EDCDF Bias Correction (Equi-Distant CDF Matching)**

Untuk mengatasi bias pada data CORDEX mentah. Saya mengelaborasi penelitian [Subramani](https://arxiv.org/abs/2504.19145) yang menggunakan model EDCDF untuk data climate change untuk mengatasi model bias.

$$x_{corrected} = F_{obs,hist}^{-1}\left(F_{mod,hist}(x_{proj})\right) + \left(x_{proj} - F_{mod,hist}^{-1}(F_{mod,hist}(x_{proj}))\right)$$

### **2. Featuring Engineering**

Dari data mentah CORDEX-SEA, dilakukan engineering fitur untuk mendapat sinyal cuaca non-linear dan time-series:

| Kategori | Detail | 
|---|---|
| **Variabel dasar** | 15 variabel (suhu, curah hujan, tekanan, angin, radiasi, kelembaban, dll) |
| **Lag features** | Lag t-1, t-2, t-3, t-6, t-12 untuk setiap variabel |
| **Rolling mean** | Rolling average 3, 6, 12 bulan |
| **Seasonal encoding** | menangkap siklus musiman |
| **Anomali suhu** | Deviasi dari rata-rata klimatologi |

### 3. Validasi Hasil (Temporal Split)

Pada data deret waktu yang berurutan (*non-iid*), k-Fold Cross Validation standar akan menciptakan **data leakage**. Sebagai gantinya, digunakan **Walk-forward Temporal Split**:

| Set | Periode | Fungsi |
|---|---|---|
| **Training** | ≤ Tahun 2000 | Melatih bobot model |
| **Validation** | 2001–2003 | Evaluasi unseen terdekat |
| **Test** | 2004–2005 | Inferensi murni (out-of-sample) |

### **4. Normalisasi Data (Z-Score)**

Kami melakukan normalisasi terhadap hasil dengan Z-Score Normalization

$$z = \frac{x - \mu}{\sigma}$$

### **5. Pemodelan: Regresi Linear Multivariat**

#### **5.1 Loss Function**
Kita memodelkan fungsi kerugian untuk multivariabel menggunakan [Shrinkage Methods dan Ridge Regression (halaman 61)](https://reader.z-library.im/read/f6273edb1db3a353e6d083291814f34688fd5df1712256585ba17b26be838f66/book/nY5V6QyA53/the-elements-of-statistical-learning-data-mining-inference-and-prediction-2nd-edition-12print.html?client_key=1fFLi67gBrNRP1j1iPy1&extension=pdf&signature=54478af843003a0f309cbd115227023a991ed8bb325d08b940f98ea0ca6b4b99&file_access_token=eyJleHAiOjE3NzU0OTA2NjQsInZlciI6IjEuMCIsInR5cCI6IkpXVCIsImFsZyI6IkhTMjU2In0.eyJzdWIiOiJmaWxlX2FjY2VzcyIsImZpbGVfc2hhMjU2IjoiZjYyNzNlZGIxZGIzYTM1M2U2ZDA4MzI5MTgxNGYzNDY4OGZkNWRmMTcxMjI1NjU4NWJhMTdiMjZiZTgzOGY2NiIsImFjY2Vzc19sZXZlbCI6ImxpbWl0ZWQiLCJpc19wdWJsaWMiOnRydWV9.DrvRn_dFw8YHccsmS5pK9xaWbSmgS0kYctfhAIrFwcU&download_location=https%3A%2F%2Fz-lib.sk%2Fdl%2Fr4Zdp5WNZX) 

$$J(\theta) = \underbrace{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}_{\text{Mean Squared Error (MSE)}} + \underbrace{\lambda \sum_{j=1}^{m} w_j^2}_{\text{Penalti L2 (Ridge)}}$$

| | |
|---|---|
| *Akurasi (MSE)* | memastikan akurasi model terhadap data asli |
| *Penalti* | mencegah overfitting berdasar pada bobot model ...|

#### **5.2 Solusi Analitik untuk Loss Function (Normal Equation)**

Kemudian kita cari posisi paling optimum dari loss function  dengan $\frac{\partial J_{(\theta)}}{\partial \theta} = 0 $

$$\theta = (X^T X + \lambda I')^{-1} X^T y$$

{[Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning: Data Mining, Inference, and Prediction (2nd Edition)](https://reader.z-library.im/read/f6273edb1db3a353e6d083291814f34688fd5df1712256585ba17b26be838f66/book/nY5V6QyA53/the-elements-of-statistical-learning-data-mining-inference-and-prediction-2nd-edition-12print.html?client_key=1fFLi67gBrNRP1j1iPy1&extension=pdf&signature=54478af843003a0f309cbd115227023a991ed8bb325d08b940f98ea0ca6b4b99&file_access_token=eyJleHAiOjE3NzU0OTA2NjQsInZlciI6IjEuMCIsInR5cCI6IkpXVCIsImFsZyI6IkhTMjU2In0.eyJzdWIiOiJmaWxlX2FjY2VzcyIsImZpbGVfc2hhMjU2IjoiZjYyNzNlZGIxZGIzYTM1M2U2ZDA4MzI5MTgxNGYzNDY4OGZkNWRmMTcxMjI1NjU4NWJhMTdiMjZiZTgzOGY2NiIsImFjY2Vzc19sZXZlbCI6ImxpbWl0ZWQiLCJpc19wdWJsaWMiOnRydWV9.DrvRn_dFw8YHccsmS5pK9xaWbSmgS0kYctfhAIrFwcU&download_location=https%3A%2F%2Fz-lib.sk%2Fdl%2Fr4Zdp5WNZX)}

### Model: Regresi Linear Multivariat

```
y = X · w + b
Solusi Analitik: θ = (XᵀX + λI)⁻¹ Xᵀy
```

### 6. Metrik Evaluasi

Tahap terakhir adalah menguji prediksi model ($\hat{y}$) pada data Test.

- **RMSE (Root Mean Squared Error)** — Deviasi rata-rata dalam °C, memberi penalti pada outlier
- **MAE (Mean Absolute Error)** — Rerata simpangan absolut
- **R² (R-Squared)** — Proporsi varians yang dijelaskan model
- **MSE (Mean Squared Error)** — Kuadrat rata-rata error

*(Catatan: Nilai metrik aktual ada di `results/metrics/evaluation_report.csv`)*

---

### Data diambil saat scrapping di website CORDEX

| Aspek | Detail |
|---|---|
| Dataset | CORDEX South-East Asia (`projections-cordex-domains-single-levels`) |
| GCM Driver | MOHC-HadGEM2-ES (CMIP5) |
| RCM | SMHI-RCA4 |
| Resolusi | 0.22° × 0.22° (~25 km) |
| Periode | 1976–2005 (hist) + 2006–2050 (RCP4.5, RCP8.5) |
| Area | Pulau Jawa: lat[-8.8, -5.9] lon[105.1, 115.7] |



---

## Referensi Utama (arXiv)

| Model | Judul | Link |
|---|---|---|
| [Book] | Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning: Data Mining, Inference, and Prediction | [(2nd edition)](https://z-library.im/book/nY5V6QyA53/the-elements-of-statistical-learning-data-mining-inference-and-prediction-2nd-edition-12print.html) |
| [CORDEX-DL-PRED] | Regional Climate Model + DL Bias Correction | [arXiv:2504.19145](https://arxiv.org/html/2504.19145) |

---

## Output yang Dihasilkan

```
results/figures/

results/metrics/
  └── evaluation_report.csv           ← MSE, RMSE, MAE, R²

results/models/
  ├── feature_importance.csv          ← Ranking fitur
```
