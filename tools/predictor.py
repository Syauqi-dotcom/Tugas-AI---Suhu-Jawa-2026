"""
tools/predictor.py
==================
Prediksi suhu masa depan menggunakan model yang sudah dilatih.
Menghasilkan proyeksi untuk RCP4.5 dan RCP8.5 (2006–2100).

Input :
  - Model tersimpan (weights.npz, normalizer_X.npz, target_stats.npy)
  - Fitur RCP yang sudah di-engineer (X_rcp45.csv, X_rcp85.csv)

Output:
  - results/metrics/pred_rcp45.csv
  - results/metrics/pred_rcp85.csv

Referensi:
  - CORDEX-DL-PRED: Ghaemi et al. (2023), arXiv:2504.19145
  - ClimateBench: Watson-Parris et al. (2022), arXiv:2110.11676

Penggunaan:
  python tools/predictor.py \\
         --model_dir results/models/ \\
         --rcp45 data/processed/features/X_rcp45.csv \\
         --rcp85 data/processed/features/X_rcp85.csv \\
         --output_dir results/metrics/
"""

import argparse
import os
import numpy as np
import pandas as pd


# ─── Loader (dari train_model.py tanpa dependency circular) ──────────────────

def load_model_weights(model_dir: str) -> tuple:
    """
    Muat bobot model dari direktori.

    Returns:
        weights   : np.ndarray bobot model
        bias      : float bias
        meta      : dict metadata model
    """
    import json
    weights_path = os.path.join(model_dir, "weights.npz")
    meta_path    = os.path.join(model_dir, "meta.json")

    data = np.load(weights_path)
    weights = data["weights"]
    bias    = float(data["bias"][0])

    with open(meta_path) as f:
        meta = json.load(f)

    return weights, bias, meta


def load_normalizer(model_dir: str) -> tuple:
    """Muat parameter normalisasi Z-score."""
    norm_path = os.path.join(model_dir, "normalizer_X.npz")
    data      = np.load(norm_path)
    return data["mean"], data["std"]


def load_target_stats(model_dir: str) -> tuple:
    """Muat mean dan std target (suhu)."""
    stats = np.load(os.path.join(model_dir, "target_stats.npy"))
    return float(stats[0]), float(stats[1])  # mean, std


def predict_rcp(feature_csv:  str,
                model_dir:    str,
                output_csv:   str,
                scenario:     str,
                time_col:     str = "time") -> pd.DataFrame:
    """
    Jalankan prediksi untuk satu skenario RCP.

    Args:
        feature_csv : CSV fitur RCP (sudah di-engineer)
        model_dir   : direktori model tersimpan
        output_csv  : path simpan prediksi
        scenario    : label ('rcp45' atau 'rcp85')
        time_col    : nama kolom waktu

    Returns:
        DataFrame dengan kolom: time, predicted_temp_2m
    """
    print(f"\n  Prediksi skenario: {scenario}")
    df = pd.read_csv(feature_csv, parse_dates=[time_col])

    # Load model
    weights, bias, meta = load_model_weights(model_dir)
    norm_mean, norm_std = load_normalizer(model_dir)
    y_mean, y_std       = load_target_stats(model_dir)

    # Siapkan fitur numerik
    drop_cols = [time_col, "Unnamed: 0"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).values

    if X.shape[1] != len(weights):
        raise ValueError(
            f"Jumlah fitur tidak cocok: CSV={X.shape[1]}, model={len(weights)}.\n"
            f"Pastikan feature engineering konsisten antara historical dan RCP."
        )

    # Normalisasi
    X_norm = (X - norm_mean) / norm_std

    # Prediksi  (y = X·w + b), kemudian de-normalisasi
    y_pred_norm = X_norm @ weights + bias
    y_pred      = y_pred_norm * y_std + y_mean

    df_out = pd.DataFrame({
        time_col:             df[time_col],
        "predicted_temp_2m":  y_pred,
        "scenario":           scenario,
    })

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    print(f"  [OK] {len(df_out)} prediksi → {output_csv}")
    print(f"  Rentang prediksi: {y_pred.min():.2f}°C – {y_pred.max():.2f}°C")

    return df_out


def main():
    parser = argparse.ArgumentParser(
        description="Prediksi suhu RCP4.5 / RCP8.5 menggunakan model terlatih"
    )
    parser.add_argument("--model_dir",  required=True,
                        help="Direktori model (weights.npz, normalizer_X.npz, target_stats.npy)")
    parser.add_argument("--rcp45",      default=None,
                        help="CSV fitur X_rcp45.csv")
    parser.add_argument("--rcp85",      default=None,
                        help="CSV fitur X_rcp85.csv")
    parser.add_argument("--output_dir", required=True,
                        help="Direktori output prediksi")
    parser.add_argument("--time_col",   default="time")
    args = parser.parse_args()

    print("\n=== Predictor — Proyeksi Suhu Jawa 2006–2100 ===")

    if not args.rcp45 and not args.rcp85:
        raise ValueError("Berikan minimal --rcp45 atau --rcp85.")

    if args.rcp45:
        predict_rcp(
            args.rcp45, args.model_dir,
            os.path.join(args.output_dir, "pred_rcp45.csv"),
            "rcp45", args.time_col,
        )

    if args.rcp85:
        predict_rcp(
            args.rcp85, args.model_dir,
            os.path.join(args.output_dir, "pred_rcp85.csv"),
            "rcp85", args.time_col,
        )

    print("\n  [SELESAI]")


if __name__ == "__main__":
    main()
