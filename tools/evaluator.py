"""
tools/evaluator.py
==================
Evaluasi model prediksi suhu: RMSE, NSE, R², MAE, MBE, Pearson r.

Mendukung dua mode penggunaan:
  A. Mode pipeline (dari data_pipeline.sh):
       --model_dir  results/models/
       --test_X     data/processed/validation/X_test.csv
       --test_y     data/processed/targets/y_test.csv
       --output     results/metrics/evaluation_report.csv

  B. Mode post-predict (sudah ada CSV prediksi):
       --pred       results/metrics/test_predictions.csv
       --output     results/metrics/evaluation_report.csv

Metrik:
  RMSE  — Root Mean Squared Error (°C)
  NSE   — Nash-Sutcliffe Efficiency (standar hidrologi/iklim)
  R²    — Koefisien determinasi
  MAE   — Mean Absolute Error
  MBE   — Mean Bias Error (bias sistematik)
  r     — Korelasi Pearson

Referensi:
  - NSE    : Nash & Sutcliffe (1970), DOI: 10.1016/0022-1694(70)90255-6
  - Metrik : Wilks (2011) "Statistical Methods in Atmospheric Sciences"
  - Bench  : Watson-Parris et al. (2022), arXiv:2110.11676
"""

import argparse
import json
import os
import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════
#  IMPLEMENTASI METRIK  (NumPy scratch, tanpa sklearn)
# ══════════════════════════════════════════════════════════════════

def compute_rmse(y_true, y_pred): return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
def compute_mae(y_true, y_pred):  return float(np.mean(np.abs(y_true - y_pred)))
def compute_mbe(y_true, y_pred):  return float(np.mean(y_pred - y_true))

def compute_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

def compute_nse(y_true, y_pred):
    """Nash-Sutcliffe Efficiency: 1=perfect, 0=mean baseline, <0=worse than mean."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("-inf")

def compute_pearson_r(y_true, y_pred):
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray,
                 split: str = "test") -> dict:
    """Hitung semua metrik dan kembalikan sebagai dict."""
    return {
        "split":     split,
        "n_samples": int(len(y_true)),
        "RMSE":      compute_rmse(y_true, y_pred),
        "MAE":       compute_mae(y_true, y_pred),
        "MBE":       compute_mbe(y_true, y_pred),
        "R2":        compute_r2(y_true, y_pred),
        "NSE":       compute_nse(y_true, y_pred),
        "Pearson_r": compute_pearson_r(y_true, y_pred),
    }


def print_report(metrics: dict):
    """Cetak laporan evaluasi terformat ke console."""
    w = 42
    print(f"\n  ┌{'─'*w}┐")
    print(f"  │ Evaluasi split : {metrics.get('split','?'):22s} │")
    print(f"  │ n_samples      : {metrics.get('n_samples',0):<22d} │")
    print(f"  ├{'─'*w}┤")
    for key, unit in [("RMSE","°C"), ("MAE","°C"), ("MBE","°C"),
                      ("R2","  "), ("NSE","  "), ("Pearson_r","  ")]:
        v = metrics.get(key, float("nan"))
        print(f"  │ {key:12s} = {v:+10.4f} {unit:3s}             │")
    print(f"  └{'─'*w}┘")


# ══════════════════════════════════════════════════════════════════
#  HELPER: load & predict menggunakan model tersimpan
# ══════════════════════════════════════════════════════════════════

def _predict_from_model(model_dir: str,
                        X_csv: str,
                        time_col: str = "time") -> tuple:
    """
    Baca model dari disk dan buat prediksi dari CSV fitur.

    Returns:
        (weights, bias, meta, y_pred_original_scale, norm_info)
    """
    # Load weights
    weights_data = np.load(os.path.join(model_dir, "weights.npz"))
    weights = weights_data["weights"]
    bias    = float(weights_data["bias"][0])

    with open(os.path.join(model_dir, "meta.json")) as f:
        meta = json.load(f)

    # Load normalizer X
    nx = np.load(os.path.join(model_dir, "normalizer_X.npz"))
    x_mean, x_std = nx["mean"], nx["std"]

    # Load target stats
    y_stats = np.load(os.path.join(model_dir, "target_stats.npy"))
    y_mean, y_std = float(y_stats[0]), float(y_stats[1])

    # Load fitur
    df = pd.read_csv(X_csv)
    drop_cols = [time_col, "Unnamed: 0"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).values

    if X.shape[1] != len(weights):
        raise ValueError(
            f"Jumlah fitur tidak cocok: CSV={X.shape[1]}, model={len(weights)}."
        )

    # Normalize & predict
    X_norm = (X - x_mean) / x_std
    y_pred_norm = X_norm @ weights + bias
    y_pred = y_pred_norm * y_std + y_mean

    return y_pred, y_mean, y_std, meta


# ══════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluasi prediksi suhu. "
            "Mode A: --model_dir + --test_X + --test_y. "
            "Mode B: --pred (CSV prediksi)."
        )
    )
    # Mode A — pipeline
    parser.add_argument("--model_dir", default=None,
                        help="(Mode A) Direktori model tersimpan")
    parser.add_argument("--test_X",   default=None,
                        help="(Mode A) CSV fitur data test")
    parser.add_argument("--test_y",   default=None,
                        help="(Mode A) CSV target data test")
    # Mode B — post-predict
    parser.add_argument("--pred",         default=None,
                        help="(Mode B) CSV hasil prediksi (kolom actual & predicted)")
    parser.add_argument("--actual_col",    default="actual")
    parser.add_argument("--predicted_col", default="predicted")
    # Shared
    parser.add_argument("--output",  required=True,
                        help="Path output CSV laporan evaluasi")
    parser.add_argument("--split",   default="test",
                        help="Label split (default: test)")
    parser.add_argument("--time_col", default="time")
    args = parser.parse_args()

    print("\n=== Evaluator ===")

    # ── Mode A: gunakan model untuk prediksi langsung
    if args.model_dir and args.test_X and args.test_y:
        print(f"  Mode : A (model → prediksi langsung)")
        y_pred, _, y_std, _ = _predict_from_model(
            args.model_dir, args.test_X, args.time_col
        )
        df_y = pd.read_csv(args.test_y)
        target_col = next(
            (c for c in df_y.columns if c not in [args.time_col, "Unnamed: 0"]),
            df_y.columns[-1]
        )
        y_true = df_y[target_col].values

        # Simpan prediksi CSV
        pred_out = os.path.join(os.path.dirname(args.output), "test_predictions.csv")
        df_y_out = df_y[[args.time_col]].copy() if args.time_col in df_y.columns \
                    else pd.DataFrame()
        df_y_out["actual"]    = y_true
        df_y_out["predicted"] = y_pred
        os.makedirs(os.path.dirname(os.path.abspath(pred_out)), exist_ok=True)
        df_y_out.to_csv(pred_out, index=False)
        print(f"  [OK] Prediksi tersimpan: {pred_out}")

    # ── Mode B: baca dari CSV prediksi yang sudah ada
    elif args.pred:
        print(f"  Mode : B (CSV prediksi)")
        df = pd.read_csv(args.pred)
        if args.actual_col not in df.columns or args.predicted_col not in df.columns:
            raise ValueError(
                f"Kolom '{args.actual_col}'/'{args.predicted_col}' tidak ditemukan.\n"
                f"Kolom tersedia: {list(df.columns)}"
            )
        y_true = df[args.actual_col].values
        y_pred = df[args.predicted_col].values

    else:
        parser.error(
            "Gunakan Mode A (--model_dir + --test_X + --test_y) "
            "atau Mode B (--pred)."
        )
        return

    metrics = evaluate_all(y_true, y_pred, args.split)
    print_report(metrics)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    pd.DataFrame([metrics]).to_csv(args.output, index=False)
    print(f"\n  [OK] Laporan evaluasi → {args.output}")


if __name__ == "__main__":
    main()
