#!/usr/bin/env python3
"""
Download 3 years of SST data from Copernicus Marine Service.

Uses the OSTIA dataset (same as IMARPE uses).
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import sqlite3

# Copernicus credentials
COPERNICUS_USER = os.environ.get('COPERNICUS_USER', 'jgodiel96@gmail.com')
COPERNICUS_PASS = os.environ.get('COPERNICUS_PASS', 'P@ulstrauss96')

# Region: Peru Tacna-Ilo
REGION = {
    'lat_min': -18.3,
    'lat_max': -17.3,
    'lon_min': -71.5,
    'lon_max': -70.8
}


def download_sst_copernicus(years: int = 3):
    """Download SST data from Copernicus Marine Service."""

    print("=" * 60)
    print("DESCARGA SST - COPERNICUS MARINE SERVICE")
    print("=" * 60)
    print(f"Usuario: {COPERNICUS_USER}")
    print(f"Años a descargar: {years}")
    print(f"Región: Tacna-Ilo, Perú")
    print()

    try:
        import copernicusmarine as cm
    except ImportError:
        print("ERROR: Instala copernicusmarine: pip install copernicusmarine")
        return None

    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years)

    print(f"Período: {start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')}")
    print()

    # Output directory
    output_dir = Path("data/copernicus")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset ID for SST
    # Using the reprocessed OSTIA dataset
    dataset_id = "cmems_SST_GLO_SST_L4_REP_OBSERVATIONS_010_024"

    print("[INFO] Conectando a Copernicus Marine...")

    try:
        # Download SST data
        data = cm.open_dataset(
            dataset_id=dataset_id,
            username=COPERNICUS_USER,
            password=COPERNICUS_PASS,
            minimum_longitude=REGION['lon_min'],
            maximum_longitude=REGION['lon_max'],
            minimum_latitude=REGION['lat_min'],
            maximum_latitude=REGION['lat_max'],
            start_datetime=start_date.strftime('%Y-%m-%d'),
            end_datetime=end_date.strftime('%Y-%m-%d'),
            variables=['analysed_sst']
        )

        print(f"[OK] Datos descargados: {data}")

        # Convert to records
        sst_records = []

        # Process the xarray dataset
        if 'analysed_sst' in data.variables:
            sst_data = data['analysed_sst']

            for t_idx, time_val in enumerate(data.time.values):
                date_str = str(time_val)[:10]

                for lat_idx, lat in enumerate(data.latitude.values):
                    for lon_idx, lon in enumerate(data.longitude.values):
                        sst_val = float(sst_data[t_idx, lat_idx, lon_idx].values)

                        # Convert from Kelvin to Celsius if needed
                        if sst_val > 100:
                            sst_val = sst_val - 273.15

                        if not (sst_val != sst_val):  # Check for NaN
                            sst_records.append({
                                'date': date_str,
                                'lat': float(lat),
                                'lon': float(lon),
                                'sst': round(sst_val, 2),
                                'source': 'copernicus_ostia'
                            })

                if t_idx % 30 == 0:
                    print(f"[INFO] Procesando: {date_str}")

        print(f"\n[OK] Total registros SST: {len(sst_records)}")

        # Save to JSON
        json_file = output_dir / "sst_3years.json"
        with open(json_file, 'w') as f:
            json.dump(sst_records, f)
        print(f"[OK] Guardado: {json_file}")

        # Save to database
        db_path = save_sst_to_db(sst_records, output_dir)

        return sst_records

    except Exception as e:
        print(f"[ERROR] {e}")
        print()
        print("Intentando método alternativo...")
        return download_sst_alternative()


def download_sst_alternative():
    """Alternative method using direct API."""

    import requests

    print("\n[INFO] Usando API directa de Copernicus...")

    # Try the subset API
    base_url = "https://nrt.cmems-du.eu/motu-web/Motu"

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 3)

    params = {
        'action': 'describeproduct',
        'service': 'SST_GLO_SST_L4_NRT_OBSERVATIONS_010_001-TDS',
        'product': 'cmems_obs-sst_glo_phy-sst_nrt_diurnal-oi-0.01-hr_P1D-m'
    }

    try:
        response = requests.get(base_url, params=params, auth=(COPERNICUS_USER, COPERNICUS_PASS), timeout=30)
        print(f"[INFO] Response: {response.status_code}")

        if response.status_code == 200:
            print("[OK] API accesible")
        else:
            print(f"[WARN] {response.text[:200]}")

    except Exception as e:
        print(f"[ERROR] {e}")

    return None


def save_sst_to_db(records: list, output_dir: Path) -> Path:
    """Save SST records to SQLite database."""

    db_path = output_dir / "copernicus_sst.db"

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
    for record in records:
        cursor.execute(
            'INSERT INTO sst (date, lat, lon, sst, source) VALUES (?, ?, ?, ?, ?)',
            (record['date'], record['lat'], record['lon'], record['sst'], record['source'])
        )

    # Metadata
    cursor.execute("INSERT INTO metadata VALUES ('source', 'copernicus_marine_ostia')")
    cursor.execute("INSERT INTO metadata VALUES ('total_records', ?)", (str(len(records)),))
    cursor.execute("INSERT INTO metadata VALUES ('created_at', ?)", (datetime.now().isoformat(),))

    # Index
    cursor.execute('CREATE INDEX idx_sst_date ON sst(date)')
    cursor.execute('CREATE INDEX idx_sst_location ON sst(lat, lon)')

    conn.commit()
    conn.close()

    print(f"[OK] Base de datos: {db_path}")

    return db_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download SST from Copernicus Marine")
    parser.add_argument('--years', type=int, default=3, help='Years of data to download')

    args = parser.parse_args()

    result = download_sst_copernicus(args.years)

    if result:
        print("\n" + "=" * 60)
        print("DESCARGA COMPLETADA")
        print("=" * 60)
        print(f"Total registros: {len(result)}")
    else:
        print("\n[ERROR] No se pudo descargar los datos")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
