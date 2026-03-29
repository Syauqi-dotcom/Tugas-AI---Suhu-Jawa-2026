"""
tools/feature_engineer.py
==========================
Feature engineering dari time series iklim Jawa.

Menyediakan dua API:
  1. Fungsi standalone (digunakan oleh quick_demo.py dan import langsung):
       seasonal_encoding(months) → (sin_arr, cos_arr)
       compute_climatology(series, times) → Series anomali
       add_lag_features(df, col, lags) → df
       add_rolling_features(df, col, windows) → df

  2. Pipeline lengkap melalui engineer_features(df) → df dengan semua
     fitur (lag, rolling, seasonal, anomali, wind speed)

Referensi:
  - Climate-Invariant ML  : Beucler et al. (2021), arXiv:2112.08440
  - Downscaling statistik : Vandal et al. (2019), arXiv:1902.02865

Penggunaan CLI:
  python tools/feature_engineer.py \\
         --input  data/processed/features/X_historical_raw.csv \\
         --output data/processed/features/X_historical.csv \\
         --target_col temp_2m
"""

import argparse
import os
import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════
#  FUNGSI STANDALONE  (dapat diimpor langsung oleh skrip lain)
# ══════════════════════════════════════════════════════════════════

def seasonal_encoding(months: np.ndarray) -> tuple:
    """
    Encode bulan sebagai pasangan nilai sin/cos untuk menangkap
    periodisitas tahunan tanpa diskontinuitas.

    Args:
        months : array integer bulan (1–12)

    Returns:
        (sin_month, cos_month) — masing-masing np.ndarray
    """
    months = np.asarray(months)
    return (
        np.sin(2 * np.pi * months / 12),
        np.cos(2 * np.pi * months / 12),
    )


def compute_climatology(series: pd.Series,
                        times:  pd.Series,
                        clim_start: int = 1976,
                        clim_end:   int = 2005) -> pd.Series:
    """
    Hitung anomali = nilai - rata-rata klimatologi per bulan.

    Periode klimatologi default: 1976–2005 (CORDEX historical baseline).

    Args:
        series     : pd.Series nilai (misal temp_2m)
        times      : pd.Series datetime yang sesuai dengan series
        clim_start : tahun awal periode klimatologi
        clim_end   : tahun akhir periode klimatologi

    Returns:
        pd.Series anomali (nilai dikurangi rata-rata bulan klimatologi)
    """
    dt   = pd.to_datetime(times)
    mask = (dt.dt.year >= clim_start) & (dt.dt.year <= clim_end)
    months = dt.dt.month

    clim_mean = (
        series[mask]
        .groupby(months[mask])
        .mean()
    )

    anomaly = series - months.map(clim_mean)
    return anomaly.reset_index(drop=True)


def add_lag_features(df: pd.DataFrame,
                     col: str,
                     lags: list = [1, 2, 3, 6, 12]) -> pd.DataFrame:
    """
    Tambahkan lag fitur untuk satu kolom.

    Args:
        df   : DataFrame input
        col  : nama kolom yang dibuat lag-nya
        lags : daftar nilai lag (dalam bulan)

    Returns:
        df dengan kolom baru `{col}_lag{lag}` untuk setiap lag
    """
    if col not in df.columns:
        return df
    for lag in lags:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame,
                         col: str,
                         windows: list = [3, 6, 12]) -> pd.DataFrame:
    """
    Tambahkan rolling mean untuk satu kolom.

    Args:
        df      : DataFrame input
        col     : nama kolom yang dihitung rolling mean-nya
        windows : daftar ukuran window (dalam bulan)

    Returns:
        df dengan kolom baru `{col}_roll{w}` untuk setiap window
    """
    if col not in df.columns:
        return df
    for w in windows:
        df[f"{col}_roll{w}"] = df[col].rolling(window=w, min_periods=1).mean()
    return df


# ══════════════════════════════════════════════════════════════════
#  FUNGSI MULTI-KOLOM  (digunakan oleh pipeline CLI)
# ══════════════════════════════════════════════════════════════════

def add_lag_features_multi(df: pd.DataFrame,
                           cols: list,
                           lags: list = [1, 2, 3, 6, 12]) -> pd.DataFrame:
    """Tambahkan lag untuk beberapa kolom sekaligus."""
    for col in cols:
        df = add_lag_features(df, col, lags)
    return df


def add_rolling_features_multi(df: pd.DataFrame,
                                cols: list,
                                windows: list = [3, 6, 12]) -> pd.DataFrame:
    """Tambahkan rolling mean untuk beberapa kolom sekaligus."""
    for col in cols:
        df = add_rolling_features(df, col, windows)
    return df


def add_wind_speed(df: pd.DataFrame,
                   u_col: str = "wind_u",
                   v_col: str = "wind_v") -> pd.DataFrame:
    """Hitung kecepatan angin total dari komponen u dan v."""
    if u_col in df.columns and v_col in df.columns:
        df["wind_speed"] = np.sqrt(df[u_col] ** 2 + df[v_col] ** 2)
    return df


def add_temperature_anomaly(df: pd.DataFrame,
                             temp_col:   str = "temp_2m",
                             time_col:   str = "time",
                             clim_start: int = 1976,
                             clim_end:   int = 2005) -> pd.DataFrame:
    """
    Hitung anomali suhu dan simpan sebagai kolom baru.
    Wrapper di atas compute_climatology untuk digunakan pipeline.
    """
    if temp_col not in df.columns:
        print(f"  [WARN] Kolom '{temp_col}' tidak ditemukan, anomali dilewati.")
        return df

    df[f"{temp_col}_anomaly"] = compute_climatology(
        df[temp_col], df[time_col], clim_start, clim_end
    ).values
    return df


# ══════════════════════════════════════════════════════════════════
#  PIPELINE LENGKAP
# ══════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame,
                      target_col: str = "temp_2m",
                      time_col:   str = "time") -> pd.DataFrame:
    """
    Jalankan seluruh tahap feature engineering secara berurutan.

    Tahapan:
      1. Kecepatan angin (wind_u + wind_v)
      2. Lag features   ( t-1, t-2, t-3, t-6, t-12 ) — semua kolom numerik
      3. Rolling mean   ( window 3, 6, 12 ) — kolom meteorologi utama
      4. Seasonal encoding (sin/cos bulan)
      5. Anomali suhu terhadap klimatologi 1976–2005
      6. Drop baris NaN akibat shift/lag

    Args:
        df         : DataFrame input (hasil nc_extractor / bias_corrector)
        target_col : kolom target (temp_2m)
        time_col   : kolom waktu

    Returns:
        DataFrame dengan fitur lengkap, tanpa baris NaN
    """
    print(f"  Input shape : {df.shape}")
    df = df.sort_values(time_col).reset_index(drop=True)

    # 1. Kecepatan angin
    df = add_wind_speed(df)
    print("  [OK] wind_speed = sqrt(wind_u² + wind_v²)")

    # 2. Lag features — semua kolom numerik kecuali waktu
    lag_cols = [c for c in df.columns if c != time_col and
                pd.api.types.is_numeric_dtype(df[c])]
    df = add_lag_features_multi(df, lag_cols, lags=[1, 2, 3, 6, 12])
    print("  [OK] Lag features: [1, 2, 3, 6, 12]")

    # 3. Rolling mean — variabel meteorologi utama
    roll_cols = [c for c in ["temp_2m", "precip", "humidity",
                              "pressure", "solar_rad"] if c in df.columns]
    df = add_rolling_features_multi(df, roll_cols, windows=[3, 6, 12])
    print("  [OK] Rolling mean: windows=[3, 6, 12]")

    # 4. Seasonal encoding
    months = pd.to_datetime(df[time_col]).dt.month.values
    df["month_sin"], df["month_cos"] = seasonal_encoding(months)
    print("  [OK] Seasonal encoding (sin/cos bulan)")

    # 5. Anomali suhu
    df = add_temperature_anomaly(df, target_col, time_col)
    print(f"  [OK] Anomali {target_col} terhadap klimatologi 1976-2005")

    # 6. Drop NaN
    n_before = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"  [OK] Drop NaN: {n_before} → {len(df)} baris")
    print(f"  Output shape: {df.shape} ({len(df.columns)} fitur)")
    return df


# ══════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Feature Engineering: lag, rolling, seasonal, anomali"
    )
    parser.add_argument("--input",      required=True,
                        help="CSV input (hasil nc_extractor/bias_corrector)")
    parser.add_argument("--output",     required=True,
                        help="CSV output dengan fitur lengkap")
    parser.add_argument("--target_col", default="temp_2m",
                        help="Kolom target (default: temp_2m)")
    parser.add_argument("--time_col",   default="time",
                        help="Kolom waktu (default: time)")
    args = parser.parse_args()

    print(f"\n=== Feature Engineer ===")
    print(f"  Input : {args.input}")
    print(f"  Output: {args.output}")

    df     = pd.read_csv(args.input, parse_dates=[args.time_col])
    df_out = engineer_features(df, args.target_col, args.time_col)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df_out.to_csv(args.output, index=False)
    print(f"\n  [SELESAI] → {args.output}")


if __name__ == "__main__":
    main()
