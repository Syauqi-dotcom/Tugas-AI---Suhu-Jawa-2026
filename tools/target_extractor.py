"""
tools/target_extractor.py
==========================
Ekstrak kolom target (suhu 2m) dari DataFrame fitur ke file terpisah.
Memisahkan X (fitur) dan y (target) sesuai konvensi pipeline ML.

Output:
  - data/processed/features/X_historical.csv  → fitur (tanpa kolom target)
  - data/processed/targets/y_historical.csv   → target (kolom temp_2m)

Penggunaan:
  python tools/target_extractor.py \\
         --input  data/processed/features/X_historical.csv \\
         --output_X data/processed/features/X_historical.csv \\
         --output_y data/processed/targets/y_historical.csv \\
         --target_col temp_2m
"""

import argparse
import os
import pandas as pd


def extract_target(input_file: str,
                   output_X:   str,
                   output_y:   str,
                   target_col: str = "temp_2m",
                   time_col:   str = "time") -> tuple:
    """
    Pisahkan fitur (X) dan target (y) dari satu DataFrame.

    Args:
        input_file : CSV input lengkap (fitur + target)
        output_X   : path simpan X (fitur tanpa target)
        output_y   : path simpan y (target saja)
        target_col : nama kolom target
        time_col   : nama kolom waktu

    Returns:
        df_X, df_y : tuple DataFrame
    """
    df = pd.read_csv(input_file, parse_dates=[time_col])

    if target_col not in df.columns:
        raise ValueError(
            f"Kolom target '{target_col}' tidak ditemukan.\n"
            f"Kolom tersedia: {list(df.columns)}"
        )

    # y: waktu + target
    df_y = df[[time_col, target_col]].copy()

    # X: semua kolom kecuali target
    df_X = df.drop(columns=[target_col]).copy()

    # Simpan
    for path, df_out, label in [
        (output_X, df_X, "X (fitur)"),
        (output_y, df_y, "y (target)"),
    ]:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        df_out.to_csv(path, index=False)
        print(f"  [OK] {label}: {df_out.shape} → {path}")

    return df_X, df_y


def main():
    parser = argparse.ArgumentParser(
        description="Pisahkan fitur X dan target y dari CSV"
    )
    parser.add_argument("--input",      required=True,
                        help="CSV input (fitur + target)")
    parser.add_argument("--output_X",   required=True,
                        help="CSV output fitur X")
    parser.add_argument("--output_y",   required=True,
                        help="CSV output target y")
    parser.add_argument("--target_col", default="temp_2m",
                        help="Nama kolom target (default: temp_2m)")
    parser.add_argument("--time_col",   default="time",
                        help="Nama kolom waktu (default: time)")
    args = parser.parse_args()

    print("\n=== Target Extractor ===")
    extract_target(args.input, args.output_X, args.output_y,
                   args.target_col, args.time_col)
    print("\n  [SELESAI]")


if __name__ == "__main__":
    main()
