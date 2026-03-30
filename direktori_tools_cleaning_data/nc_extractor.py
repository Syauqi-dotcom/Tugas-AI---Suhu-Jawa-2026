"""
tools/nc_extractor.py
=====================
Ekstrak data NetCDF CORDEX → CSV untuk subset wilayah Jawa.
Merata-rata spatially seluruh grid Jawa → time series bulanan.

Variabel yang diekstrak:
  - tasmin/tasmax/tas      → suhu permukaan
  - pr                     → curah hujan
  - ps / hurs              → tekanan, kelembaban
  - uas / vas              → komponen angin
  - rsds                   → radiasi surya

Referensi:
  - netCDF4-python: Whitaker et al. (2023)
  - CORDEX-SEA domain: Tangang et al. (2020), DOI: 10.1007/s00382-020-05218-7

Penggunaan:
  python tools/nc_extractor.py --input data/raw/historical/ \\
         --output data/processed/features/X_historical.csv \\
         --scenario historical
"""

import argparse
import glob
import os
import zipfile
import tempfile
import uuid
import numpy as np
import pandas as pd
import netCDF4 as nc


# ─── Konfigurasi Variabel ─────────────────────────────────────────────────────

# Peta nama variabel CDS → nama kolom CSV
VAR_MAP = {
    "tas":    "temp_2m",
    "pr":     "precip",
    "ps":     "pressure",
    "hurs":   "humidity",
    "huss":   "specific_humidity",
    "uas":    "wind_u",
    "vas":    "wind_v",
    "sfcWind":"wind_speed",
    "tasmax": "temp_max_24h",
    "tasmin": "temp_min_24h",
    "evspsbl":"evaporation",
    "psl":    "sea_level_pressure",
    "rsds":   "solar_rad",
    "rlds":   "thermal_rad",
    "clt":    "cloud_cover"
}

# Faktor konversi (ke satuan standar)
CONVERSION = {
    "tas":     lambda x: x - 273.15,   # K → °C
    "tasmax":  lambda x: x - 273.15,   # K → °C
    "tasmin":  lambda x: x - 273.15,   # K → °C
    "pr":      lambda x: x * 86400,    # kg/m²/s → mm/hari
    "evspsbl": lambda x: x * 86400,    # kg/m²/s → mm/hari
    "ps":      lambda x: x / 100,      # Pa → hPa
    "psl":     lambda x: x / 100,      # Pa → hPa
    "hurs":    lambda x: x,            # % 
    "huss":    lambda x: x * 1000,     # kg/kg → g/kg
    "clt":     lambda x: x,            # %
    "uas":     lambda x: x,            # m/s
    "vas":     lambda x: x,            # m/s
    "sfcWind": lambda x: x,            # m/s
    "rsds":    lambda x: x,            # W/m²
    "rlds":    lambda x: x,            # W/m²
}

# Batas Jawa
LAT_MIN, LAT_MAX = -8.8, -5.9
LON_MIN, LON_MAX = 105.1, 115.7


# ─── Fungsi Ekstrak ──────────────────────────────────────────────────────────

def extract_java_mean(nc_file: str, var_name: str) -> tuple:
    """
    Ekstrak mean spasial area Jawa dari satu file NetCDF.

    Returns:
        times  : array datetime
        values : array float (mean spasial Jawa)
    """
    ds = nc.Dataset(nc_file)

    # Cari dimensi lat / lon
    lat = ds.variables.get("lat") or ds.variables.get("rlat")
    lon = ds.variables.get("lon") or ds.variables.get("rlon")
    if lat is None or lon is None:
        raise KeyError(f"Dimensi lat/lon tidak ditemukan di {nc_file}")

    lat_vals = lat[:].data
    lon_vals = lon[:].data

    # Mask wilayah Jawa
    lat_mask = (lat_vals >= LAT_MIN) & (lat_vals <= LAT_MAX)
    lon_mask = (lon_vals >= LON_MIN) & (lon_vals <= LON_MAX)

    # Waktu
    time_var = ds.variables["time"]
    times = nc.num2date(time_var[:], time_var.units,
                        calendar=getattr(time_var, "calendar", "standard"))
    times = [pd.Timestamp(t.year, t.month, getattr(t, "day", 1)) for t in times]

    # Variabel
    if var_name not in ds.variables:
        ds.close()
        return None, None

    data = ds.variables[var_name][:]  # shape: (time, lat, lon)
    if hasattr(data, "data"):
        data = np.ma.filled(data, np.nan)

    # Subset & mean spasial
    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]
    subset = data[:, lat_idx, :][:, :, lon_idx]  # (time, lat_J, lon_J)
    values = np.nanmean(subset, axis=(1, 2))      # (time,)

    # Konversi
    conv = CONVERSION.get(var_name, lambda x: x)
    values = conv(values)

    ds.close()
    return times, values


def extract_scenario(nc_dir: str, scenario: str) -> pd.DataFrame:
    """
    Proses semua file NetCDF di direktori dan gabungkan ke satu DataFrame.

    Args:
        nc_dir   : folder berisi file .nc atau .zip
        scenario : 'historical' atau 'rcp85'

    Returns:
        DataFrame dengan kolom: time + semua variabel
    """
    nc_files = sorted(glob.glob(os.path.join(nc_dir, "*.nc")) + glob.glob(os.path.join(nc_dir, "*.zip")))
    if not nc_files:
        raise FileNotFoundError(f"Tidak ada file .nc atau .zip di {nc_dir}")

    print(f"  Ditemukan {len(nc_files)} file untuk scenario={scenario}")

    dfs = {}

    def process_nc_file(target_nc_file):
        ds = nc.Dataset(target_nc_file)
        available_vars = [v for v in VAR_MAP if v in ds.variables]
        ds.close()
        for var in available_vars:
            times, values = extract_java_mean(target_nc_file, var)
            if times is None: continue
            col = VAR_MAP[var]
            tmp = pd.DataFrame({"time": times, col: values})
            if col not in dfs: dfs[col] = tmp
            else: dfs[col] = pd.concat([dfs[col], tmp], ignore_index=True)

    for file_path in nc_files:
        print(f"  Proses: {os.path.basename(file_path)}")
        if zipfile.is_zipfile(file_path):
            with zipfile.ZipFile(file_path, 'r') as zf:
                netcdf_names = [n for n in zf.namelist() if n.endswith('.nc')]
                for n in netcdf_names:
                    temp_nc = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.nc")
                    with open(temp_nc, "wb") as fout:
                        fout.write(zf.read(n))
                    try:
                        process_nc_file(temp_nc)
                    finally:
                        try: os.remove(temp_nc)
                        except: pass
        else:
            try:
                process_nc_file(file_path)
            except Exception as e:
                print(f"  [ERROR] {file_path}: {e}")

    if not dfs:
        raise ValueError("Tidak ada variabel yang berhasil diekstrak!")

    # Gabungkan semua variabel berdasarkan waktu (Resample daily -> monthly)
    df = None
    for col, tmp in dfs.items():
        # Agregasikan data temp_2m harian menjadi rata-rata bulanan
        tmp = tmp.set_index("time").resample("MS").mean().reset_index()
        tmp = tmp.dropna(subset=[col]).sort_values("time").reset_index(drop=True)
        
        if df is None:
            df = tmp
        else:
            df = pd.merge(df, tmp, on="time", how="outer")

    df = df.sort_values("time").reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Ekstrak NetCDF CORDEX → CSV (subset Jawa)"
    )
    parser.add_argument("--input",    required=True,
                        help="Direktori berisi file .nc")
    parser.add_argument("--output",   required=True,
                        help="Path output CSV")
    parser.add_argument("--scenario", required=True,
                        choices=["historical", "rcp85"],
                        help="Label skenario")
    args = parser.parse_args()

    print(f"\n=== NC Extractor — CORDEX subset Jawa ===")
    print(f"  Input   : {args.input}")
    print(f"  Output  : {args.output}")
    print(f"  Skenario: {args.scenario}")

    df = extract_scenario(args.input, args.scenario)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"  [OK] {len(df)} baris × {len(df.columns)} kolom → {args.output}")


if __name__ == "__main__":
    main()
