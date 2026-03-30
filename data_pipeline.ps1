$ErrorActionPreference = "Stop"

Write-Host "============================================================"
Write-Host " PIPELINE SUHU PERMUKAAN JAWA - CORDEX-CMIP5"
Write-Host "============================================================"

# --- Deteksi Virtual Environment ---
$PYTHON_EXEC = "python"
if (Test-Path "..\AI\Scripts\python.exe") {
    $PYTHON_EXEC = "..\AI\Scripts\python.exe"
    Write-Host "[INFO] Menggunakan Python dari Virtual Environment: $PYTHON_EXEC `n" -ForegroundColor Green
} else {
    Write-Host "[WARN] Virtual Environment tidak terdeteksi! Menggunakan python global.`n" -ForegroundColor Yellow
}

# --- Direktori ---
$RAW_HIST = "data\raw\historical"
$RAW_RCP85 = "data\raw\rcp85"
$PROCESSED = "data\processed"
$RESULTS = "results"
$TMP_DIR = "tmp_folder"

New-Item -ItemType Directory -Force -Path $TMP_DIR, "$RESULTS\figures", "$RESULTS\metrics", "$RESULTS\models" | Out-Null

# --- TAHAP 1 ---
Write-Host "`n[TAHAP 1] Download data CORDEX-SEA dari CDS Copernicus..."

$env:CDSAPI_RC = "$PWD\.cdsapirc"

& $PYTHON_EXEC direktori_tools_cleaning_data\cds_downloader.py --scenario historical --temporal_resolution monthly_mean --output_dir $RAW_HIST
& $PYTHON_EXEC direktori_tools_cleaning_data\cds_downloader.py --scenario rcp85 --temporal_resolution monthly_mean --output_dir $RAW_RCP85

# --- TAHAP 2 ---
Write-Host "`n[TAHAP 2] Ekstraksi dan subset wilayah Pulau Jawa..."

$scenarios = @("historical", "rcp85")
foreach ($scenario in $scenarios) {
    Write-Host "  -> Memproses skenario: $scenario"
    & $PYTHON_EXEC direktori_tools_cleaning_data\nc_extractor.py --input "data\raw\$scenario" --output "$TMP_DIR\${scenario}_raw.csv" --scenario $scenario
}

# --- TAHAP 3a ---
Write-Host "`n[TAHAP 3a] Bias correction (EDCDF)..."
& $PYTHON_EXEC direktori_tools_cleaning_data\bias_corrector.py --historical "$TMP_DIR\historical_raw.csv" --rcp85 "$TMP_DIR\rcp85_raw.csv" --output_dir $TMP_DIR

# --- TAHAP 3b ---
Write-Host "`n[TAHAP 3b] Menggabungkan CSV prediktor..."
& $PYTHON_EXEC direktori_tools_cleaning_data\feature_merger.py --inputs "$TMP_DIR\historical_corrected.csv" --output "$TMP_DIR\hist_features.csv"
& $PYTHON_EXEC direktori_tools_cleaning_data\feature_merger.py --inputs "$TMP_DIR\rcp85_corrected.csv" --output "$TMP_DIR\rcp85_features.csv"

# --- TAHAP 3c ---
Write-Host "`n[TAHAP 3c] Feature engineering..."
& $PYTHON_EXEC direktori_tools_cleaning_data\feature_engineer.py --input "$TMP_DIR\hist_features.csv" --output "$PROCESSED\features\X_historical.csv"
& $PYTHON_EXEC direktori_tools_cleaning_data\feature_engineer.py --input "$TMP_DIR\rcp85_features.csv" --output "$PROCESSED\features\X_rcp85.csv"

# --- TAHAP 3d ---
Write-Host "`n[TAHAP 3d] Ekstraksi kolom target..."
& $PYTHON_EXEC direktori_tools_cleaning_data\target_extractor.py --input "$PROCESSED\features\X_historical.csv" --output_X "$PROCESSED\features\X_historical.csv" --output_y "$PROCESSED\targets\y_historical.csv" --target_col temp_2m

# --- TAHAP 3e ---
Write-Host "`n[TAHAP 3e] Temporal split..."
& $PYTHON_EXEC direktori_tools_cleaning_data\data_splitter.py --features "$PROCESSED\features\X_historical.csv" --targets "$PROCESSED\targets\y_historical.csv" --output_dir "$PROCESSED\validation" --train_end 2000 --val_start 2001 --val_end 2003 --test_start 2004 --test_end 2005


Write-Host "`n============================================================"
Write-Host " PIPELINE DATA SELESAI"
Write-Host " Lanjutkan ke notebooks\01_main_analysis.ipynb untuk modeling"
Write-Host "============================================================"
