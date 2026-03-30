#!/bin/bash
# =============================================================================
# data_pipeline.sh
# Pipeline Data: Pemodelan Prediksi Suhu Permukaan Jawa
# CORDEX-CMIP5 | Regresi Multivariat + EXDM Optimizer
# Terinspirasi dari: github.com/aminst/wits/data_pipeline.sh
# =============================================================================

set -e  # Exit on any error

echo "============================================================"
echo " PIPELINE SUHU PERMUKAAN JAWA — CORDEX-CMIP5"
echo "============================================================"

# --- Direktori ---
RAW_HIST="data/raw/historical"
RAW_RCP85="data/raw/rcp85"
PROCESSED="data/processed"
RESULTS="results"
TMP="tmp"

mkdir -p "$TMP" "$RESULTS/figures" "$RESULTS/metrics" "$RESULTS/models"

# ─────────────────────────────────────────────────────────────────
# TAHAP 1: DOWNLOAD DATA DARI CDS COPERNICUS
# ─────────────────────────────────────────────────────────────────
echo ""
echo "[TAHAP 1] Download data CORDEX-SEA dari CDS Copernicus..."

#!/bin/bash

export CDSAPI_RC=$(pwd)/.cdsapirc

python3 direktori_tools_cleaning_data/cds_downloader.py \
    --scenario historical \
    --temporal_resolution monthly_mean \
    --output_dir "$RAW_HIST"

python3 direktori_tools_cleaning_data/cds_downloader.py \
    --scenario rcp85 \
    --temporal_resolution monthly_mean \
    --output_dir "$RAW_RCP85"

# ─────────────────────────────────────────────────────────────────
# TAHAP 2: EKSTRAKSI & SUBSET WILAYAH JAWA
# ─────────────────────────────────────────────────────────────────
echo ""
echo "[TAHAP 2] Ekstraksi dan subset wilayah Pulau Jawa..."
echo "         Bounding Box: lat [-8.8, -5.9] | lon [105.1, 115.7]"

for SCENARIO in historical rcp85; do
    echo "  -> Memproses skenario: $SCENARIO"
    python3 direktori_tools_cleaning_data/nc_extractor.py \
        --input    "data/raw/$SCENARIO" \
        --output   "$TMP/${SCENARIO}_raw.csv" \
        --scenario "$SCENARIO"
done

# ─────────────────────────────────────────────────────────────────
# TAHAP 3: BIAS CORRECTION (EDCDF)
# ─────────────────────────────────────────────────────────────────
echo ""
echo "[TAHAP 3a] Bias correction (EDCDF)..."

python3 direktori_tools_cleaning_data/bias_corrector.py \
    --historical  "$TMP/historical_raw.csv" \
    --rcp85       "$TMP/rcp85_raw.csv" \
    --output_dir  "$TMP"

# ─────────────────────────────────────────────────────────────────
# TAHAP 3b: FEATURE MERGER
# ─────────────────────────────────────────────────────────────────
echo ""
echo "[TAHAP 3b] Menggabungkan CSV prediktor..."

# Untuk data historis — gabungkan semua variabel dari satu file ekstraksi
python3 direktori_tools_cleaning_data/feature_merger.py \
    --inputs "$TMP/historical_corrected.csv" \
    --output "$TMP/hist_features.csv"

# Untuk RCP — jika ada file tambahan, tambahkan ke --inputs
python3 direktori_tools_cleaning_data/feature_merger.py \
    --inputs "$TMP/rcp85_corrected.csv" \
    --output "$TMP/rcp85_features.csv"

# ─────────────────────────────────────────────────────────────────
# TAHAP 3c: FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────
echo ""
echo "[TAHAP 3c] Feature engineering (lag, rolling, seasonal, anomali)..."

python3 direktori_tools_cleaning_data/feature_engineer.py \
    --input  "$TMP/hist_features.csv" \
    --output "$PROCESSED/features/X_historical.csv"

python3 direktori_tools_cleaning_data/feature_engineer.py \
    --input  "$TMP/rcp85_features.csv" \
    --output "$PROCESSED/features/X_rcp85.csv"

# ─────────────────────────────────────────────────────────────────
# TAHAP 3d: EKSTRAKSI TARGET
# ─────────────────────────────────────────────────────────────────
echo ""
echo "[TAHAP 3d] Ekstraksi kolom target (temp_2m)..."

python3 direktori_tools_cleaning_data/target_extractor.py \
    --input    "$PROCESSED/features/X_historical.csv" \
    --output_X "$PROCESSED/features/X_historical.csv" \
    --output_y "$PROCESSED/targets/y_historical.csv" \
    --target_col temp_2m

# ─────────────────────────────────────────────────────────────────
# TAHAP 3e: TRAIN/VAL/TEST SPLIT
# ─────────────────────────────────────────────────────────────────
echo ""
echo "[TAHAP 3e] Temporal split: train(≤2000) / val(2001-2003) / test(2004-2005)..."

python3 direktori_tools_cleaning_data/data_splitter.py \
    --features   "$PROCESSED/features/X_historical.csv" \
    --targets    "$PROCESSED/targets/y_historical.csv" \
    --output_dir "$PROCESSED/validation" \
    --train_end  2000 \
    --val_start  2001 \
    --val_end    2003 \
    --test_start 2004 \
    --test_end   2005

echo "  -> Fitur tersimpan di: $PROCESSED/features/"
echo "  -> Target tersimpan di: $PROCESSED/targets/"
echo "  -> Split tersimpan di: $PROCESSED/validation/"

echo ""
echo "============================================================"
echo " PIPELINE DATA SELESAI"
echo " Lanjutkan ke notebooks/01_main_analysis.ipynb untuk modeling."
echo "============================================================"

# ─────────────────────────────────────────────────────────────────
# SELESAI
# ─────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " PIPELINE SELESAI"
echo " Hasil: results/figures/ | results/metrics/ | results/models/"
echo "============================================================"
