"""
tools/visualizer.py
===================
Semua visualisasi matplotlib untuk proyek prediksi suhu Jawa CORDEX.

Plot yang dihasilkan:
    1. eda_timeseries.png             – EDA: time series 7 variabel utama
    2. eda_correlation_matrix.png     – Matriks korelasi Pearson
    3. training_convergence.png       – Kurva loss training EXDM
    4. evaluation_result.png          – Pred vs aktual, scatter plot, residual
    5. feature_importance.png         – Top 15 fitur berdasarkan |bobot|
    6. proyeksi_suhu_jawa_2100.png    – Proyeksi historis + RCP4.5 + RCP8.5

Referensi:
  - Matplotlib: Hunter (2007), DOI: 10.1109/MCSE.2007.55
  - ClimateBench: Watson-Parris et al. (2022), arXiv:2110.11676

Penggunaan:
  python tools/visualizer.py --mode all --output_dir results/figures/
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless mode
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ─── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":       150,
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
})

PALETTE = ["#2196F3", "#F44336", "#4CAF50", "#FF9800",
           "#9C27B0", "#00BCD4", "#795548"]


# ─── 1. EDA Time Series ──────────────────────────────────────────────────────

def plot_eda_timeseries(csv_path: str, output_path: str,
                        time_col: str = "time"):
    """Plot time series 7 variabel utama dari data historis."""
    df = pd.read_csv(csv_path, parse_dates=[time_col])

    plot_cols = [c for c in ["temp_2m", "precip", "pressure", "humidity",
                              "wind_speed", "solar_rad", "temp_2m_anomaly"]
                 if c in df.columns][:7]
    n = len(plot_cols)
    if n == 0:
        print("  [WARN] Tidak ada kolom yang cocok untuk EDA time series.")
        return

    labels = {
        "temp_2m": "Suhu 2m (°C)", "precip": "Curah Hujan (mm/hari)",
        "pressure": "Tekanan (hPa)", "humidity": "Kelembaban (%)",
        "wind_speed": "Kec. Angin (m/s)", "solar_rad": "Radiasi Surya (W/m²)",
        "temp_2m_anomaly": "Anomali Suhu (°C)",
    }

    fig, axes = plt.subplots(n, 1, figsize=(14, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, col, color in zip(axes, plot_cols, PALETTE):
        ax.plot(df[time_col], df[col], color=color, linewidth=0.8, alpha=0.85)
        ax.set_ylabel(labels.get(col, col), fontsize=9)
        ax.fill_between(df[time_col], df[col], alpha=0.1, color=color)

    axes[-1].xaxis.set_major_locator(mdates.YearLocator(5))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30)

    fig.suptitle("EDA — Time Series Variabel CORDEX SEA (Jawa)", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] EDA time series → {output_path}")


# ─── 2. Matriks Korelasi ─────────────────────────────────────────────────────

def plot_correlation_matrix(csv_path: str, output_path: str,
                            time_col: str = "time", top_n: int = 20):
    """Plot matriks korelasi Pearson top-N fitur."""
    df = pd.read_csv(csv_path)
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    # Pilih top_n berdasarkan variance
    if len(num_cols) > top_n:
        variances = df[num_cols].var().nlargest(top_n)
        num_cols  = variances.index.tolist()

    corr = df[num_cols].corr()
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr, cmap="RdYlBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Pearson r")

    ax.set_xticks(range(len(num_cols)))
    ax.set_yticks(range(len(num_cols)))
    ax.set_xticklabels(num_cols, rotation=90, fontsize=7)
    ax.set_yticklabels(num_cols, fontsize=7)
    ax.set_title(f"Matriks Korelasi Pearson — Top {len(num_cols)} Fitur", fontsize=12)

    # Anotasi nilai
    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            val = corr.iloc[i, j]
            if abs(val) > 0.6:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=5, color="white" if abs(val) > 0.8 else "black")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Korelasi matrix → {output_path}")



# ─── 4. Training Convergence ─────────────────────────────────────────────────

def plot_training_convergence(loss_history: list, val_loss_history: list,
                              output_path: str):
    """Plot kurva loss EXDM selama training."""
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(loss_history) + 1)

    ax.plot(epochs, loss_history, color="#2196F3", linewidth=1.2,
            label="Train Loss (MSE + L2)")
    if val_loss_history:
        ax.plot(range(1, len(val_loss_history) + 1), val_loss_history,
                color="#F44336", linewidth=1.2, linestyle="--",
                label="Val Loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Kurva Konvergensi EXDM Optimizer")
    ax.legend()
    ax.set_yscale("log")

    best_epoch = int(np.argmin(val_loss_history)) + 1 if val_loss_history else None
    if best_epoch:
        ax.axvline(best_epoch, color="#4CAF50", linestyle=":", alpha=0.7,
                   label=f"Best epoch: {best_epoch}")
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Convergence plot → {output_path}")


# ─── 5. Evaluation Result ────────────────────────────────────────────────────

def plot_evaluation_result(y_true: np.ndarray, y_pred: np.ndarray,
                           times: pd.Series, output_path: str,
                           metrics: dict = None):
    """Plot: prediksi vs aktual + scatter + residual."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Plot 1: Time series
    ax = axes[0]
    ax.plot(times, y_true, color="#2196F3", linewidth=1.0, label="Aktual", alpha=0.9)
    ax.plot(times, y_pred, color="#F44336", linewidth=1.0, linestyle="--",
            label="Prediksi", alpha=0.9)
    ax.set_ylabel("Suhu 2m (°C)")
    ax.set_title("Prediksi vs Aktual — Test Set")
    ax.legend()

    # Plot 2: Scatter
    ax = axes[1]
    vmin = min(y_true.min(), y_pred.min())
    vmax = max(y_true.max(), y_pred.max())
    ax.scatter(y_true, y_pred, alpha=0.5, s=20, color="#9C27B0")
    ax.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=1, label="1:1 line")
    ax.set_xlabel("Suhu Aktual (°C)")
    ax.set_ylabel("Suhu Prediksi (°C)")
    ax.set_title("Scatter Plot: Aktual vs Prediksi")
    if metrics:
        txt = f"R²={metrics.get('R2',0):.3f}  RMSE={metrics.get('RMSE',0):.3f}°C  NSE={metrics.get('NSE',0):.3f}"
        ax.text(0.05, 0.92, txt, transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.legend()

    # Plot 3: Residual
    residuals = y_true - y_pred
    ax = axes[2]
    ax.plot(times, residuals, color="#FF9800", linewidth=0.8, alpha=0.8)
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax.fill_between(times, residuals, alpha=0.2, color="#FF9800")
    ax.set_ylabel("Residual (°C)")
    ax.set_xlabel("Waktu")
    ax.set_title("Residual = Aktual − Prediksi")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Evaluation plot → {output_path}")


# ─── 6. Feature Importance ───────────────────────────────────────────────────

def plot_feature_importance(fi_csv: str, output_path: str, top_n: int = 15):
    """Plot top-N feature importance berdasarkan magnitude bobot ternormalisasi."""
    df = pd.read_csv(fi_csv).head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.45)))
    colors = ["#2196F3" if w >= 0 else "#F44336"
              for w in df.get("weight", df.get("importance", [0] * len(df)))]
    y_pos  = range(len(df))

    col = "importance" if "importance" in df.columns else "weight"
    ax.barh(y_pos, df[col], color=colors, alpha=0.85, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["feature"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Importance (|w| ternormalisasi)")
    ax.set_title(f"Top {top_n} Feature Importance")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Feature importance → {output_path}")


# ─── 7. Proyeksi Suhu 2100 ───────────────────────────────────────────────────

def plot_proyeksi_suhu(hist_csv:  str, rcp45_csv: str, rcp85_csv: str,
                       output_path: str,
                       time_col: str = "time",
                       temp_col: str = "temp_2m"):
    """Plot proyeksi suhu Jawa 1976–2100: historis + RCP4.5 + RCP8.5."""
    fig, ax = plt.subplots(figsize=(16, 6))

    def load_and_smooth(path, col, window=12):
        df = pd.read_csv(path, parse_dates=[time_col])
        col_use = col if col in df.columns else \
                  "predicted_temp_2m" if "predicted_temp_2m" in df.columns else \
                  df.select_dtypes(include=np.number).columns[0]
        series = df[col_use].rolling(window, min_periods=1).mean()
        return df[time_col], series

    if os.path.exists(hist_csv):
        t, s = load_and_smooth(hist_csv, temp_col)
        ax.plot(t, s, color="#2196F3", linewidth=1.5, label="Historis (1976–2005)")

    if os.path.exists(rcp45_csv):
        t, s = load_and_smooth(rcp45_csv, temp_col)
        ax.plot(t, s, color="#4CAF50", linewidth=1.5, linestyle="--",
                label="RCP4.5 (2006–2100)")
        ax.fill_between(t, s - 0.2, s + 0.2, alpha=0.15, color="#4CAF50")

    if os.path.exists(rcp85_csv):
        t, s = load_and_smooth(rcp85_csv, temp_col)
        ax.plot(t, s, color="#F44336", linewidth=1.5, linestyle="-.",
                label="RCP8.5 (2006–2100)")
        ax.fill_between(t, s - 0.2, s + 0.2, alpha=0.15, color="#F44336")

    ax.axvline(pd.Timestamp("2006-01-01"), color="gray",
               linestyle=":", linewidth=1, alpha=0.7)
    ax.text(pd.Timestamp("2006-06-01"), ax.get_ylim()[0] + 0.1,
            "Hist | Proyeksi", fontsize=8, color="gray")

    ax.set_xlabel("Tahun")
    ax.set_ylabel("Suhu Rata-rata 2m Jawa (°C, rolling 12-bulan)")
    ax.set_title("Proyeksi Suhu Permukaan Jawa 1976–2100\n"
                 "CORDEX-SEA | MOHC-HadGEM2-ES / SMHI-RCA4 | EXDM Regression",
                 fontsize=12)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Proyeksi suhu → {output_path}")


# ─── Main CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualisasi semua plot untuk proyek CORDEX Jawa"
    )
    parser.add_argument("--mode", default="all",
                        choices=["all", "eda", "corr", "convergence",
                                 "evaluation", "importance", "proyeksi"],
                        help="Plot yang dibuat")
    parser.add_argument("--output_dir", default="results/figures/")

    # Path optional untuk masing-masing plot
    parser.add_argument("--X_hist",        default="data/processed/features/X_historical.csv")
    parser.add_argument("--pred_csv",      default="results/metrics/test_predictions.csv")
    parser.add_argument("--metrics_csv",   default="results/metrics/evaluation_report.csv")
    parser.add_argument("--fi_csv",        default="results/models/feature_importance.csv")
    parser.add_argument("--model_dir",     default="results/models/")
    parser.add_argument("--hist_csv",      default="results/metrics/pred_rcp45.csv")
    parser.add_argument("--rcp45_csv",     default="results/metrics/pred_rcp45.csv")
    parser.add_argument("--rcp85_csv",     default="results/metrics/pred_rcp85.csv")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("\n=== Visualizer — Proyek CORDEX Jawa ===")

    modes = [args.mode] if args.mode != "all" else \
            ["eda", "corr", "convergence", "evaluation", "importance", "proyeksi"]

    for mode in modes:
        if mode == "eda" and os.path.exists(args.X_hist):
            plot_eda_timeseries(args.X_hist,
                                os.path.join(args.output_dir, "eda_timeseries.png"))

        elif mode == "corr" and os.path.exists(args.X_hist):
            plot_correlation_matrix(args.X_hist,
                                    os.path.join(args.output_dir, "eda_correlation_matrix.png"))

        elif mode == "convergence":
            weight_file = os.path.join(args.model_dir, "weights.npz")
            if os.path.exists(weight_file):
                data = np.load(weight_file)
                plot_training_convergence(
                    list(data.get("loss_history", [])),
                    list(data.get("val_loss_history", [])),
                    os.path.join(args.output_dir, "training_convergence.png"),
                )

        elif mode == "evaluation" and os.path.exists(args.pred_csv):
            df = pd.read_csv(args.pred_csv, parse_dates=["time"] if "time" in
                             pd.read_csv(args.pred_csv, nrows=1).columns else None)
            metrics = None
            if os.path.exists(args.metrics_csv):
                metrics = pd.read_csv(args.metrics_csv).iloc[0].to_dict()
            plot_evaluation_result(
                df["actual"].values, df["predicted"].values,
                df.get("time", pd.Series(range(len(df)))),
                os.path.join(args.output_dir, "evaluation_result.png"),
                metrics,
            )

        elif mode == "importance" and os.path.exists(args.fi_csv):
            plot_feature_importance(args.fi_csv,
                                    os.path.join(args.output_dir, "feature_importance.png"))

        elif mode == "proyeksi":
            plot_proyeksi_suhu(
                args.hist_csv, args.rcp45_csv, args.rcp85_csv,
                os.path.join(args.output_dir, "proyeksi_suhu_jawa_2100.png"),
            )

    print("\n  [SELESAI] Semua plot disimpan ke:", args.output_dir)


if __name__ == "__main__":
    main()
