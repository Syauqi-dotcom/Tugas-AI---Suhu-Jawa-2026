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
import cdsapi


# ─── Konfigurasi Default ────────────────────────────────────────────────────────

DATASET = "projections-cordex-domains-single-levels"

# Bounding box Jawa: [lat_min, lon_min, lat_max, lon_max] → CDS format [N, W, S, E]
AREA_JAVA = [-5.9, 105.1, -8.8, 115.7]

VARIABLES = [
    "2m_air_temperature",                
    # "mean_precipitation_flux",           
    # "surface_pressure",
    # "near_surface_relative_humidity",
    # "10m_u_component_of_wind",
    # "10m_v_component_of_wind",
    # "surface_solar_radiation_downwards",
]

HISTORICAL_YEARS  = [str(y) for y in range(1976, 2006)] 
RCP45_YEARS       = [str(y) for y in range(2006, 2101)] 
RCP85_YEARS       = [str(y) for y in range(2006, 2101)] 
MONTHS            = [f"{m:02d}" for m in range(1, 13)]

def download_cordex(scenario: str, years: list, output_dir: str,
                    domain: str = "southeast_asia", 
                    gcm: str    = "MOHC-HadGEM2-ES",
                    rcm: str    = "SMHI-RCA4"):
    """
    Download data CORDEX untuk skenario tertentu.

    Args:
        scenario   : 'historical', 'rcp_4_5', atau 'rcp_8_5'
        years      : list tahun yang diunduh
        output_dir : direktori simpan file NetCDF
        domain     : CORDEX domain (default: SEA-22)
        gcm        : GCM driver (default: MOHC-HadGEM2-ES)
        rcm        : RCM (default: SMHI-RCA4)
    """
    os.makedirs(output_dir, exist_ok=True)
    client = cdsapi.Client()

    for year in years:
        output_file = os.path.join(output_dir, f"cordex_{scenario}_{year}.nc")
        if os.path.exists(output_file):
            print(f"  [SKIP] {output_file} sudah ada.")
            continue

        print(f"  Downloading: scenario={scenario}, year={year} ...")
        client.retrieve(
            DATASET,
            {
                "domain":         domain,
                "experiment":     scenario,
                "gcm_model":      gcm,
                "rcm_model":      rcm,
                "variable":       VARIABLES,
                "temporal_resolution": "monthly_mean",
                "start_year":     year,
                "end_year":       year,
                "month":          MONTHS,
                "area":           AREA_JAVA,
                "format":         "netcdf",
            },
            output_file,
        )
        print(f"  [OK] Tersimpan: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Download CORDEX SEA dari CDS API"
    )
    parser.add_argument(
        "--scenario", required=True,
        choices=["historical", "rcp45", "rcp85", "all"],
        help="Skenario yang diunduh"
    )
    parser.add_argument(
        "--output_dir", default="data/raw",
        help="Direktori output (default: data/raw)"
    )
    args = parser.parse_args()

    scenarios = {
        "historical": ("historical", HISTORICAL_YEARS,
                       os.path.join(args.output_dir, "historical")),
        "rcp45":      ("rcp_4_5", RCP45_YEARS,
                       os.path.join(args.output_dir, "rcp45")),
        "rcp85":      ("rcp_8_5", RCP85_YEARS,
                       os.path.join(args.output_dir, "rcp85")),
    }

    if args.scenario == "all":
        targets = list(scenarios.values())
    else:
        targets = [scenarios[args.scenario]]

    print("\n=== CDS Downloader — CORDEX South-East Asia ===")
    for cds_scenario, years, out_dir in targets:
        print(f"\n--- Skenario: {cds_scenario} ({len(years)} tahun) ---")
        download_cordex(cds_scenario, years, out_dir)

    print("\n[SELESAI] Download selesai.")


if __name__ == "__main__":
    main()
