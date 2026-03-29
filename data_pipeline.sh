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
RAW_RCP45="data/raw/rcp45"
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

# Beritahu cdsapi untuk melihat file .cdsapirc di folder tools
export CDSAPI_RC=$(pwd)/.cdsapirc

python3 tools/cds_downloader.py \
    --scenario historical \
    --output_dir "$RAW_HIST"

python3 tools/cds_downloader.py \
    --scenario rcp45 \
    --output_dir "$RAW_RCP45"

python3 tools/cds_downloader.py \
    --scenario rcp85 \
    --output_dir "$RAW_RCP85"

# ─────────────────────────────────────────────────────────────────
# TAHAP 2: EKSTRAKSI & SUBSET WILAYAH JAWA
# ─────────────────────────────────────────────────────────────────
echo ""
echo "[TAHAP 2] Ekstraksi dan subset wilayah Pulau Jawa..."
echo "         Bounding Box: lat [-8.8, -5.9] | lon [105.1, 115.7]"

for SCENARIO in historical rcp45 rcp85; do
    echo "  -> Memproses skenario: $SCENARIO"
    python3 tools/nc_extractor.py \
        --input    "data/raw/$SCENARIO" \
        --output   "$TMP/${SCENARIO}_raw.csv" \
        --scenario "$SCENARIO"
done

# ─────────────────────────────────────────────────────────────────
# TAHAP 3: BIAS CORRECTION (EDCDF)
# ─────────────────────────────────────────────────────────────────
echo ""
echo "[TAHAP 3a] Bias correction (EDCDF)..."

python3 tools/bias_corrector.py \
    --historical  "$TMP/historical_raw.csv" \
    --rcp45       "$TMP/rcp45_raw.csv" \
    --rcp85       "$TMP/rcp85_raw.csv" \
    --output_dir  "$TMP"

# ─────────────────────────────────────────────────────────────────
# TAHAP 3b: FEATURE MERGER
# ─────────────────────────────────────────────────────────────────
echo ""
echo "[TAHAP 3b] Menggabungkan CSV prediktor..."

# Untuk data historis — gabungkan semua variabel dari satu file ekstraksi
python3 tools/feature_merger.py \
    --inputs "$TMP/historical_corrected.csv" \
    --output "$TMP/hist_features.csv"

# Untuk RCP — jika ada file tambahan, tambahkan ke --inputs
python3 tools/feature_merger.py \
    --inputs "$TMP/rcp45_corrected.csv" \
    --output "$TMP/rcp45_features.csv"

python3 tools/feature_merger.py \
    --inputs "$TMP/rcp85_corrected.csv" \
    --output "$TMP/rcp85_features.csv"

# ─────────────────────────────────────────────────────────────────
# TAHAP 3c: FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────
echo ""
echo "[TAHAP 3c] Feature engineering (lag, rolling, seasonal, anomali)..."

python3 tools/feature_engineer.py \
    --input  "$TMP/hist_features.csv" \
    --output "$PROCESSED/features/X_historical.csv"

python3 tools/feature_engineer.py \
    --input  "$TMP/rcp45_features.csv" \
    --output "$PROCESSED/features/X_rcp45.csv"

python3 tools/feature_engineer.py \
    --input  "$TMP/rcp85_features.csv" \
    --output "$PROCESSED/features/X_rcp85.csv"

# ─────────────────────────────────────────────────────────────────
# TAHAP 3d: EKSTRAKSI TARGET
# ─────────────────────────────────────────────────────────────────
echo ""
echo "[TAHAP 3d] Ekstraksi kolom target (temp_2m)..."

python3 tools/target_extractor.py \
    --input    "$PROCESSED/features/X_historical.csv" \
    --output_X "$PROCESSED/features/X_historical.csv" \
    --output_y "$PROCESSED/targets/y_historical.csv" \
    --target_col temp_2m

# ─────────────────────────────────────────────────────────────────
# TAHAP 3e: TRAIN/VAL/TEST SPLIT
# ─────────────────────────────────────────────────────────────────
echo ""
echo "[TAHAP 3e] Temporal split: train(≤2000) / val(2001-2003) / test(2004-2005)..."

python3 tools/data_splitter.py \
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

# ─────────────────────────────────────────────────────────────────
# TAHAP 4: TRAINING MODEL (Regresi Multivariat + EXDM)
# ─────────────────────────────────────────────────────────────────
echo ""
echo "[TAHAP 4] Training model regresi multivariat dengan EXDM optimizer..."

python3 tools/train_model.py \
    --train_X "$PROCESSED/validation/X_train.csv" \
    --train_y "$PROCESSED/validation/y_train.csv" \
    --val_X   "$PROCESSED/validation/X_val.csv" \
    --val_y   "$PROCESSED/validation/y_val.csv" \
    --output  "$RESULTS/models" \
    --optimizer exdm \
    --epochs 500 \
    --lr 0.01 \
    --l2 1e-4

# ─────────────────────────────────────────────────────────────────
# TAHAP 5: EVALUASI & METRIK
# ─────────────────────────────────────────────────────────────────
echo ""
echo "[TAHAP 5] Evaluasi model pada data test..."

python3 tools/evaluator.py \
    --model_dir "$RESULTS/models" \
    --test_X    "$PROCESSED/validation/X_test.csv" \
    --test_y    "$PROCESSED/validation/y_test.csv" \
    --output    "$RESULTS/metrics/evaluation_report.csv" \
    --split     test

# ─────────────────────────────────────────────────────────────────
# TAHAP 6: PREDIKSI SKENARIO MASA DEPAN
# ─────────────────────────────────────────────────────────────────
echo ""
echo "[TAHAP 6] Prediksi suhu Jawa untuk RCP4.5 dan RCP8.5..."

python3 tools/predictor.py \
    --model_dir  "$RESULTS/models" \
    --rcp45      "$PROCESSED/features/X_rcp45.csv" \
    --rcp85      "$PROCESSED/features/X_rcp85.csv" \
    --output_dir "$RESULTS/metrics"

# ─────────────────────────────────────────────────────────────────
# TAHAP 7: VISUALISASI
# ─────────────────────────────────────────────────────────────────
echo ""
echo "[TAHAP 7] Membuat visualisasi hasil..."

python3 tools/visualizer.py \
    --mode       all \
    --output_dir "$RESULTS/figures" \
    --X_hist     "$PROCESSED/features/X_historical.csv" \
    --pred_csv   "$RESULTS/metrics/test_predictions.csv" \
    --metrics_csv "$RESULTS/metrics/evaluation_report.csv" \
    --fi_csv     "$RESULTS/models/feature_importance.csv" \
    --model_dir  "$RESULTS/models" \
    --hist_csv   "$PROCESSED/targets/y_historical.csv" \
    --rcp45_csv  "$RESULTS/metrics/pred_rcp45.csv" \
    --rcp85_csv  "$RESULTS/metrics/pred_rcp85.csv"

# ─────────────────────────────────────────────────────────────────
# SELESAI
# ─────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " PIPELINE SELESAI"
echo " Hasil: results/figures/ | results/metrics/ | results/models/"
echo "============================================================"
