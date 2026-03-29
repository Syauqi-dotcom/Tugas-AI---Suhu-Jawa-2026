"""
tools/feature_merger.py
=======================
Gabungkan beberapa CSV prediktor menjadi satu DataFrame fitur (X).
Terinspirasi dari struktur pipeline aminst/wits.

Input : beberapa CSV dengan kolom 'time' + variabel
Output: X_historical.csv (atau X_rcp45.csv, X_rcp85.csv)

Referensi:
  - Pipeline wits: https://github.com/aminst/wits
  - Climate-Invariant ML: Beucler et al. (2021), arXiv:2112.08440

Penggunaan CLI:
  python tools/feature_merger.py \\
         --inputs data/processed/features/raw_hist.csv extra.csv \\
         --output data/processed/features/X_historical.csv
"""

import argparse
import os
import pandas as pd


def merge_features(input_files: list,
                   output_file:  str,
                   time_col:     str = "time") -> pd.DataFrame:
    """
    Gabungkan beberapa CSV berdasarkan kolom waktu (outer join).

    Args:
        input_files : list path ke CSV input
        output_file : path output CSV termerge
        time_col    : nama kolom waktu (default: 'time')

    Returns:
        DataFrame hasil merge
    """
    if not input_files:
        raise ValueError("Tidak ada file input yang diberikan.")

    print(f"  Merge {len(input_files)} file CSV:")
    df = None

    for fpath in input_files:
        if not os.path.exists(fpath):
            print(f"  [WARN] File tidak ditemukan: {fpath}, dilewati.")
            continue

        tmp = pd.read_csv(fpath, parse_dates=[time_col])
        print(f"    {os.path.basename(fpath):40s} → {tmp.shape}")

        if df is None:
            df = tmp
            continue

        # Kolom duplikat selain time → abaikan dari file berikutnya
        overlap = [c for c in tmp.columns if c in df.columns and c != time_col]
        if overlap:
            print(f"  [WARN] Kolom duplikat diabaikan: {overlap}")
            tmp = tmp.drop(columns=overlap)

        df = pd.merge(df, tmp, on=time_col, how="outer")

    if df is None:
        raise ValueError("Semua file input gagal dibaca.")

    df = df.sort_values(time_col).reset_index(drop=True)

    # Laporan missing values
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print(f"\n  [INFO] Missing values (ffill + bfill):")
        for col, cnt in missing.items():
            print(f"    {col}: {cnt} ({cnt/len(df)*100:.1f}%)")
        df = df.ffill().bfill()

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"\n  [OK] {df.shape} → {output_file}")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Gabungkan beberapa CSV prediktor (outer join pada waktu)"
    )
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Satu atau lebih file CSV input")
    parser.add_argument("--output", required=True,
                        help="Path CSV output hasil merge")
    parser.add_argument("--time_col", default="time",
                        help="Nama kolom waktu (default: time)")
    args = parser.parse_args()

    print("\n=== Feature Merger ===")
    merge_features(args.inputs, args.output, args.time_col)


if __name__ == "__main__":
    main()
