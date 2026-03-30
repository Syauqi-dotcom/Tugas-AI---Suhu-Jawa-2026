"""
tools/cds_downloader.py
=======================
Download data CORDEX South-East Asia dari Copernicus Climate Data Store (CDS)
menggunakan cdsapi.

Dataset: projections-cordex-domains-single-levels
GCM    : MOHC-HadGEM2-ES
RCM    : SMHI-RCA4
Resolusi: 0.22° × 0.22° (~25 km)
Area   : Jawa [-8.8, 105.1, -5.9, 115.7] (S, W, N, E)

Referensi:
  - CDS API: https://cds.climate.copernicus.eu/
  - CORDEX-SEA: Tangang et al. (2020), DOI: 10.1007/s00382-020-05218-7
"""

import argparse
import os
import time
import cdsapi


# ─── Konfigurasi Default ────────────────────────────────────────────────────────

DATASET = "projections-cordex-domains-single-levels"

# Bounding box Jawa: [lat_min, lon_min, lat_max, lon_max] → CDS format [N, W, S, E]
AREA_JAVA = [-5.9, 105.1, -8.8, 115.7]

VARIABLES = [
    "2m_air_temperature",
    "2m_relative_humidity",
    "10m_u_component_of_the_wind",
    "10m_v_component_of_the_wind",
    "mean_precipitation_flux",
    "surface_pressure",
    "surface_solar_radiation_downwards"
]

HISTORICAL_YEARS  = [str(y) for y in range(1986, 2006)] 
RCP85_YEARS       = [str(y) for y in range(2006, 2046)] 
# Sesuai contoh script pengguna, kita bisa meminta spesifik bulan:
MONTHS            = [f"{m:02d}" for m in range(1, 13)]

def download_cordex(scenario: str, years: list, output_dir: str,
                    domain: str = "south_east_asia", 
                    gcm: str    = "mohc_hadgem2_es",
                    rcm: str    = "ictp_regcm4_7",
                    temporal_resolution: str = "monthly_mean",
                    log_file: str = "download_log.txt"):
    """
    Download data CORDEX untuk skenario tertentu.

    Args:
        scenario   : 'historical' atau 'rcp_8_5'
        years      : list tahun yang diunduh
        output_dir : direktori simpan file NetCDF
        domain     : CORDEX domain (default: south_east_asia)
        gcm        : GCM driver (default: mohc_hadgem2_es)
        rcm        : RCM (default: ictp_regcm4_7)
        log_file   : File untuk melog tahun-tahun yang gagal diunduh
    """
    os.makedirs(output_dir, exist_ok=True)
    client = cdsapi.Client()

    for year in years:
        output_file = os.path.join(output_dir, f"cordex_{scenario}_{year}.nc")
        if os.path.exists(output_file):
            print(f"  [SKIP] {output_file} sudah ada.")
            continue

        max_retries = 5
        retry_delay = 10
        success = False

        for attempt in range(max_retries):
            try:
                print(f"  Downloading: scenario={scenario}, year={year} (Attempt {attempt+1}/{max_retries}) ...")
                client.retrieve(
                    DATASET,
                    {
                        "domain":         domain,
                        "experiment":     scenario,
                        "horizontal_resolution": "0_22_degree_x_0_22_degree",
                        "gcm_model":      gcm,
                        "rcm_model":      rcm,
                        "ensemble_member": "r1i1p1",
                        "variable":       VARIABLES,
                        "temporal_resolution": temporal_resolution,
                        "start_year":     [str(year)],
                        "end_year":       [str(year)],
                        "month":          MONTHS,
                        "format":         "zip",
                    },
                    output_file,
                )
                print(f"  [OK] Tersimpan: {output_file}")
                success = True
                break
            except Exception as e:
                print(f"  [ERROR] Gagal download year={year} pada attempt {attempt+1}: {e}")
                if attempt < max_retries - 1:
                    print(f"  [RETRY] Menunggu {retry_delay} detik sebelum mencoba lagi...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff

        if not success:
            print(f"  [FAILED] Melewati tahun {year} setelah {max_retries} percobaan.")
            with open(log_file, "a") as f:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{timestamp}] Gagal mengunduh: scenario={scenario}, year={year}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Download CORDEX SEA dari CDS API"
    )
    parser.add_argument(
        "--scenario", required=True,
        choices=["historical", "rcp85", "all"],
        help="Skenario yang diunduh"
    )
    parser.add_argument(
        "--output_dir", default="data/raw",
        help="Direktori output (default: data/raw)"
    )
    parser.add_argument(
        "--temporal_resolution", default="monthly_mean",
        choices=["monthly_mean", "daily_mean"],
        help="Resolusi temporal (default: monthly_mean)"
    )
    args = parser.parse_args()

    scenarios = {
        "historical": ("historical", HISTORICAL_YEARS, args.output_dir),
        "rcp85":      ("rcp_8_5", RCP85_YEARS, args.output_dir),
    }

    if args.scenario == "all":
        targets = list(scenarios.values())
    else:
        targets = [scenarios[args.scenario]]

    print("\n=== CDS Downloader — CORDEX South-East Asia ===")
    for cds_scenario, years, out_dir in targets:
        print(f"\n--- Skenario: {cds_scenario} ({len(years)} tahun) ---")
        download_cordex(cds_scenario, years, out_dir, temporal_resolution=args.temporal_resolution)

    print("\n[SELESAI] Download selesai.")


if __name__ == "__main__":
    main()
