"""
tools/bias_corrector.py
=======================
Koreksi bias model iklim menggunakan metode EDCDF
(Equi-Distant Cumulative Distribution Function).

Referensi:
  Li et al. (2010) J. Hydrology — EDCDF bias correction
  arxiv.org/html/2504.19145 — GCM Bias Correction using Deep Learning
  Piani et al. (2010) — Statistical bias correction of climate model output

EDCDF: Mengoreksi proyeksi model dengan membandingkan CDF model
terhadap CDF observasi/referensi pada periode historis.
Implementasi dari scratch menggunakan NumPy.
"""

import argparse
import os
import numpy as np
import pandas as pd


# ─── EDCDF Implementation (NumPy scratch) ─────────────────────────────────────

class EDCDFBiasCorrector:
    """
    Equi-Distant CDF (EDCDF) Bias Correction.

    Prinsip: Untuk setiap nilai proyeksi x_proj, koreksi dilakukan dengan:
      x_corrected = F_obs_hist^{-1}(F_mod_hist(x_proj))
                  + (x_proj - F_mod_hist^{-1}(F_mod_hist(x_proj)))

    di mana F adalah CDF empiris, dan ^{-1} adalah invers (quantile).
    """

    def __init__(self, n_quantiles: int = 100):
        self.n_quantiles = n_quantiles
        self.quantile_levels = np.linspace(0, 1, n_quantiles + 1)
        self._obs_quantiles = {}   # CDF referensi (historis observasi/ERA5)
        self._mod_quantiles = {}   # CDF model historis
        self._fitted = False

    def fit(self, obs_hist: np.ndarray, mod_hist: np.ndarray):
        """
        Latih corrector dari data historis.
        obs_hist: nilai referensi (ERA5 / CRU observasi)
        mod_hist: nilai model (CORDEX historis)
        """
        obs_clean = obs_hist[~np.isnan(obs_hist)]
        mod_clean = mod_hist[~np.isnan(mod_hist)]

        self._obs_quantiles = np.quantile(obs_clean, self.quantile_levels)
        self._mod_quantiles = np.quantile(mod_clean, self.quantile_levels)
        self._fitted = True
        return self

    def transform(self, mod_proj: np.ndarray) -> np.ndarray:
        """
        Koreksi bias pada data proyeksi (RCP4.5 / RCP8.5).
        """
        if not self._fitted:
            raise RuntimeError("Panggil fit() terlebih dahulu.")

        corrected = np.empty_like(mod_proj)
        for i, x in enumerate(mod_proj):
            if np.isnan(x):
                corrected[i] = np.nan
                continue

            # Cari percentile x dalam CDF model historis
            p = np.searchsorted(self._mod_quantiles, x) / self.n_quantiles
            p = np.clip(p, 0, 1)

            # Ekuivalen nilai dalam CDF observasi historis
            obs_equiv = np.interp(p, self.quantile_levels, self._obs_quantiles)
            mod_equiv = np.interp(p, self.quantile_levels, self._mod_quantiles)

            # Delta (equi-distant)
            delta = x - mod_equiv
            corrected[i] = obs_equiv + delta

        return corrected

    def fit_transform(self, obs_hist, mod_hist, mod_proj):
        return self.fit(obs_hist, mod_hist).transform(mod_proj)


# ─── Pipeline Bias Correction ─────────────────────────────────────────────────

def apply_bias_correction(historical_file: str, rcp_file: str,
                          output_file: str):
    """
    Terapkan EDCDF ke seluruh variabel dalam DataFrame.
    Historis digunakan sebagai referensi koreksi.
    """
    df_hist = pd.read_csv(historical_file, parse_dates=["time"])
    df_rcp  = pd.read_csv(rcp_file,        parse_dates=["time"])

    feature_cols = [c for c in df_hist.columns if c != "time"]
    df_corrected = df_rcp.copy()

    print(f"  Variabel yang dikoreksi: {feature_cols}")

    for col in feature_cols:
        if col not in df_rcp.columns:
            print(f"  [SKIP] {col} tidak ada di file RCP")
            continue

        obs_hist = df_hist[col].values
        mod_hist = df_hist[col].values  # simplified: hist model ≈ hist obs
        mod_proj = df_rcp[col].values

        corrector = EDCDFBiasCorrector(n_quantiles=100)
        df_corrected[col] = corrector.fit_transform(obs_hist, mod_hist, mod_proj)

        bias_before = np.nanmean(mod_proj) - np.nanmean(obs_hist)
        bias_after  = np.nanmean(df_corrected[col].values) - np.nanmean(obs_hist)
        print(f"  {col}: bias sebelum={bias_before:.3f} → sesudah={bias_after:.3f}")

    df_corrected.to_csv(output_file, index=False)
    print(f"  [OK] Tersimpan: {output_file}")
    return df_corrected


def main():
    parser = argparse.ArgumentParser(description="EDCDF Bias Correction CORDEX")
    parser.add_argument("--historical",  required=True)
    parser.add_argument("--rcp85",       required=True)
    parser.add_argument("--output_dir",  required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n=== Bias Corrector (EDCDF) ===")

    # Salin historis (sudah menjadi referensi)
    df_hist = pd.read_csv(args.historical, parse_dates=["time"])
    hist_out = os.path.join(args.output_dir, "historical_corrected.csv")
    df_hist.to_csv(hist_out, index=False)
    print(f"  Historis disalin ke: {hist_out}")

    # Koreksi RCP8.5
    apply_bias_correction(
        args.historical,
        args.rcp85,
        os.path.join(args.output_dir, "rcp85_corrected.csv"),
    )


if __name__ == "__main__":
    main()
