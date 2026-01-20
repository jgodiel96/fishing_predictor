#!/usr/bin/env python3
"""
Download SST data from Copernicus Marine Service - Year by Year.

This script downloads SST data in yearly chunks to avoid timeouts.
Data is saved per year and then combined into a single database.

Usage:
    python scripts/download_sst_by_year.py --year 2024
    python scripts/download_sst_by_year.py --all  # Download all available years
    python scripts/download_sst_by_year.py --combine  # Combine all years into one DB
"""

import sys
import os
from pathlib import Path
import json
import sqlite3
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

# Region: Peru Tacna-Ilo
REGION = {
    'lat_min': -18.3,
    'lat_max': -17.3,
    'lon_min': -71.5,
    'lon_max': -70.8
}

OUTPUT_DIR = Path("data/copernicus")


def download_year(year: int) -> list:
    """Download SST data for a specific year."""

    import copernicusmarine as cm

    print(f"\n{'='*60}")
    print(f"DESCARGANDO SST - AÑO {year}")
    print(f"{'='*60}")

    # The NRT dataset has data from 2024-01-17 onwards
    # For historical data, we need to check availability

    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    # Adjust for dataset availability (NRT starts 2024-01-17)
    if year == 2024:
        start_date = "2024-01-17"

    print(f"Período: {start_date} a {end_date}")
    print(f"Región: Tacna-Ilo ({REGION['lat_min']} a {REGION['lat_max']})")
    print()

    try:
        print("[INFO] Conectando a Copernicus Marine...")

        data = cm.open_dataset(
            dataset_id='METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2',
            minimum_longitude=REGION['lon_min'],
            maximum_longitude=REGION['lon_max'],
            minimum_latitude=REGION['lat_min'],
            maximum_latitude=REGION['lat_max'],
            start_datetime=start_date,
            end_datetime=end_date,
            variables=['analysed_sst']
        )

        print(f"[OK] Dataset cargado")
        print(f"    Tiempo: {len(data.time)} días")
        print(f"    Latitud: {len(data.latitude)} puntos")
        print(f"    Longitud: {len(data.longitude)} puntos")

        # Convert to records
        records = []
        sst_var = data['analysed_sst']

        total_days = len(data.time)

        for t_idx in range(total_days):
            time_val = str(data.time.values[t_idx])[:10]

            for lat_idx, lat in enumerate(data.latitude.values):
                for lon_idx, lon in enumerate(data.longitude.values):
                    sst = float(sst_var[t_idx, lat_idx, lon_idx].values)

                    # Convert Kelvin to Celsius if needed
                    if sst > 100:
                        sst = sst - 273.15

                    # Skip NaN values
                    if sst != sst:
                        continue

                    records.append({
                        'date': time_val,
                        'lat': round(float(lat), 2),
                        'lon': round(float(lon), 2),
                        'sst': round(sst, 2),
                        'source': 'copernicus_ostia'
                    })

            # Progress indicator
            if t_idx % 30 == 0:
                pct = (t_idx + 1) / total_days * 100
                print(f"    [{pct:.0f}%] Procesando: {time_val}")

        print(f"\n[OK] Total registros: {len(records)}")

        # Save to JSON
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        json_file = OUTPUT_DIR / f"sst_{year}.json"

        with open(json_file, 'w') as f:
            json.dump(records, f)

        print(f"[OK] Guardado: {json_file}")

        return records

    except Exception as e:
        print(f"[ERROR] {e}")
        return []


def combine_all_years():
    """Combine all yearly SST files into a single database."""

    print("\n" + "=" * 60)
    print("COMBINANDO DATOS SST DE TODOS LOS AÑOS")
    print("=" * 60)

    all_records = []

    # Find all yearly JSON files
    json_files = sorted(OUTPUT_DIR.glob("sst_*.json"))

    if not json_files:
        print("[ERROR] No se encontraron archivos SST")
        return None

    for json_file in json_files:
        print(f"[INFO] Cargando: {json_file.name}")
        with open(json_file, 'r') as f:
            records = json.load(f)
            all_records.extend(records)
            print(f"       {len(records)} registros")

    print(f"\n[OK] Total combinado: {len(all_records)} registros")

    # Save to database
    db_path = OUTPUT_DIR / "sst_combined.db"

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('DROP TABLE IF EXISTS sst')
    cursor.execute('DROP TABLE IF EXISTS metadata')

    cursor.execute('''
        CREATE TABLE sst (
            id INTEGER PRIMARY KEY,
            date TEXT,
            lat REAL,
            lon REAL,
            sst REAL,
            source TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    ''')

    # Insert records
    for record in all_records:
        cursor.execute(
            'INSERT INTO sst (date, lat, lon, sst, source) VALUES (?, ?, ?, ?, ?)',
            (record['date'], record['lat'], record['lon'], record['sst'], record['source'])
        )

    # Get date range
    dates = sorted(set(r['date'] for r in all_records))

    cursor.execute("INSERT INTO metadata VALUES ('source', 'copernicus_marine_ostia')")
    cursor.execute("INSERT INTO metadata VALUES ('total_records', ?)", (str(len(all_records)),))
    cursor.execute("INSERT INTO metadata VALUES ('start_date', ?)", (dates[0],))
    cursor.execute("INSERT INTO metadata VALUES ('end_date', ?)", (dates[-1],))
    cursor.execute("INSERT INTO metadata VALUES ('created_at', ?)", (datetime.now().isoformat(),))

    cursor.execute('CREATE INDEX idx_sst_date ON sst(date)')
    cursor.execute('CREATE INDEX idx_sst_location ON sst(lat, lon)')

    conn.commit()
    conn.close()

    print(f"\n[OK] Base de datos: {db_path}")
    print(f"    Período: {dates[0]} a {dates[-1]}")
    print(f"    Registros: {len(all_records)}")

    return db_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download SST from Copernicus by year")
    parser.add_argument('--year', type=int, help='Specific year to download (e.g., 2024)')
    parser.add_argument('--all', action='store_true', help='Download all available years (2024-2025)')
    parser.add_argument('--combine', action='store_true', help='Combine all years into one database')

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.combine:
        combine_all_years()
    elif args.all:
        # Download available years (NRT dataset: 2024-present)
        for year in [2024, 2025]:
            download_year(year)
        combine_all_years()
    elif args.year:
        download_year(args.year)
    else:
        print("Uso:")
        print("  python scripts/download_sst_by_year.py --year 2024")
        print("  python scripts/download_sst_by_year.py --all")
        print("  python scripts/download_sst_by_year.py --combine")
        print()
        print("Nota: El dataset NRT solo tiene datos desde 2024-01-17")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
