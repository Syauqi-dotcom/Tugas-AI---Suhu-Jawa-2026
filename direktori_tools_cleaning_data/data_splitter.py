"""
tools/data_splitter.py
======================
Temporal train/val/test split untuk data iklim.

Split strategy (temporal, bukan random):
  - Train : tahun ≤ 2000
  - Val   : 2001–2003
  - Test  : 2004–2005

Alasan temporal split:
  Data iklim bersifat time-ordered; random split akan menyebabkan
  data leakage karena lag features bergantung pada data masa lalu.

Output (di --output_dir):
  X_train.csv, X_val.csv, X_test.csv
  y_train.csv, y_val.csv, y_test.csv

Referensi:
  - Vandal et al. (2019), arXiv:1902.02865 — temporal CV downscaling
  - Bottou et al. (2018), arXiv:1606.04838 — stochastic optimization

Penggunaan CLI:
  python tools/data_splitter.py \\
         --features data/processed/features/X_historical.csv \\
         --targets  data/processed/targets/y_historical.csv \\
         --output_dir data/processed/validation/ \\
         --train_end 2000 --val_end 2003
"""

import argparse
import os
import pandas as pd


# ─── Default split boundaries ─────────────────────────────────────────────────
TRAIN_END  = 2000
VAL_START  = 2001
VAL_END    = 2003
TEST_START = 2004
TEST_END   = 2005


def temporal_split(df_X:  pd.DataFrame,
                   df_y:  pd.DataFrame,
                   time_col:   str = "time",
                   train_end:  int = TRAIN_END,
                   val_start:  int = VAL_START,
                   val_end:    int = VAL_END,
                   test_start: int = TEST_START,
                   test_end:   int = TEST_END) -> dict:
    """
    Lakukan temporal split dan kembalikan dict split DataFrames.

    Returns:
        {
          "X_train": df, "X_val": df, "X_test": df,
          "y_train": df, "y_val": df, "y_test": df
        }
    """
    year_X = pd.to_datetime(df_X[time_col]).dt.year
    year_y = pd.to_datetime(df_y[time_col]).dt.year

    boundaries = [
        ("train", year_X <= train_end,
                  year_y <= train_end),
        ("val",   (year_X >= val_start) & (year_X <= val_end),
                  (year_y >= val_start) & (year_y <= val_end)),
        ("test",  (year_X >= test_start) & (year_X <= test_end),
                  (year_y >= test_start) & (year_y <= test_end)),
    ]

    splits = {}
    for name, mask_X, mask_y in boundaries:
        splits[f"X_{name}"] = df_X[mask_X].reset_index(drop=True)
        splits[f"y_{name}"] = df_y[mask_y].reset_index(drop=True)

    return splits


def main():
    parser = argparse.ArgumentParser(
        description="Temporal train/val/test split data historis CORDEX"
    )
    # --features dan --targets sesuai data_pipeline.sh
    parser.add_argument("--features",   required=True,
                        help="CSV fitur  X_historical.csv")
    parser.add_argument("--targets",    required=True,
                        help="CSV target y_historical.csv")
    parser.add_argument("--output_dir", required=True,
                        help="Direktori output split CSV")
    parser.add_argument("--train_end",  type=int, default=TRAIN_END,
                        help=f"Tahun akhir train (default: {TRAIN_END})")
    parser.add_argument("--val_start",  type=int, default=VAL_START)
    parser.add_argument("--val_end",    type=int, default=VAL_END,
                        help=f"Tahun akhir val (default: {VAL_END})")
    parser.add_argument("--test_start", type=int, default=TEST_START)
    parser.add_argument("--test_end",   type=int, default=TEST_END)
    parser.add_argument("--time_col",   default="time")
    args = parser.parse_args()

    print("\n=== Data Splitter (Temporal) ===")
    print(f"  Train  : ≤ {args.train_end}")
    print(f"  Val    : {args.val_start}–{args.val_end}")
    print(f"  Test   : {args.test_start}–{args.test_end}")

    df_X = pd.read_csv(args.features, parse_dates=[args.time_col])
    df_y = pd.read_csv(args.targets,  parse_dates=[args.time_col])

    splits = temporal_split(
        df_X, df_y, args.time_col,
        args.train_end, args.val_start, args.val_end,
        args.test_start, args.test_end,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    n_total = len(df_X)

    for split_name, df_split in splits.items():
        path = os.path.join(args.output_dir, f"{split_name}.csv")
        df_split.to_csv(path, index=False)
        pct = len(df_split) / n_total * 100 if n_total > 0 else 0
        print(f"  [OK] {split_name:8s}: {len(df_split):4d} baris ({pct:.1f}%) → {path}")

    print("\n  [SELESAI]")


if __name__ == "__main__":
    main()
