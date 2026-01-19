#!/usr/bin/env python3
"""
Download 100% REAL Data - NO SYNTHETIC DATA.

This script downloads ONLY real data from verified sources:
- SST: NOAA NCEI satellite data (OISST)
- Waves/Wind: Open-Meteo ERA5 reanalysis
- Fishing: Global Fishing Watch AIS data (requires API key)

Usage:
    # First, get your free GFW API key from:
    # https://globalfishingwatch.org/our-apis/

    export GFW_API_KEY='your_api_key_here'
    python scripts/download_100_real.py --months 6
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import sqlite3
from datetime import datetime, timedelta
from data.fetchers.real_data_only import RealDataFetcher, RealDataError


def check_gfw_api_key():
    """Check if GFW API key is set."""
    api_key = os.environ.get('GFW_API_KEY', '')

    if not api_key:
        print("=" * 60)
        print("⚠️  GLOBAL FISHING WATCH API KEY NOT SET")
        print("=" * 60)
        print()
        print("Para obtener datos de pesca REALES, necesitas una API key gratuita:")
        print()
        print("1. Ve a: https://globalfishingwatch.org/our-apis/")
        print("2. Haz click en 'Request API Access'")
        print("3. Completa el formulario (es gratis para investigación)")
        print("4. Recibirás la API key por email")
        print()
        print("Una vez tengas la key, configúrala así:")
        print()
        print("    export GFW_API_KEY='tu_api_key_aqui'")
        print()
        print("Luego ejecuta este script nuevamente.")
        print("=" * 60)
        return False

    return True


def download_real_data(months: int = 6):
    """Download real data for specified number of months."""

    end_date = datetime.now()
    start_date = end_date - timedelta(days=30 * months)

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    print("=" * 60)
    print("DESCARGA DE DATOS 100% REALES")
    print("=" * 60)
    print(f"Período: {start_str} a {end_str}")
    print(f"Región: Tacna-Ilo, Perú")
    print()
    print("Fuentes de datos REALES:")
    print("  - SST: NOAA NCEI (satélite OISST)")
    print("  - Olas/Viento: Open-Meteo (ERA5 reanálisis)")
    print("  - Pesca: Global Fishing Watch (AIS)")
    print("=" * 60)

    fetcher = RealDataFetcher(cache_dir="data/real_only")

    all_data = {
        'sst': [],
        'marine': [],
        'fishing': [],
        'metadata': {
            'start_date': start_str,
            'end_date': end_str,
            'data_type': 'REAL_ONLY',
            'sources': []
        }
    }

    # 1. Download SST
    print("\n" + "=" * 40)
    print("PASO 1/3: SST SATELITAL")
    print("=" * 40)

    try:
        sst_data = fetcher.fetch_real_sst(start_str, end_str)
        all_data['sst'] = sst_data
        all_data['metadata']['sources'].append('noaa_ncei_sst')
        print(f"✅ SST: {len(sst_data)} registros reales")
    except RealDataError as e:
        print(f"❌ SST Error: {e}")
        return None

    # 2. Download Marine conditions
    print("\n" + "=" * 40)
    print("PASO 2/3: CONDICIONES MARINAS (ERA5)")
    print("=" * 40)

    try:
        marine_data = fetcher.fetch_real_marine_conditions(start_str, end_str)
        all_data['marine'] = marine_data
        all_data['metadata']['sources'].append('open_meteo_era5')
        print(f"✅ Marine: {len(marine_data)} registros reales")
    except RealDataError as e:
        print(f"❌ Marine Error: {e}")
        return None

    # 3. Download Fishing activity
    print("\n" + "=" * 40)
    print("PASO 3/3: ACTIVIDAD PESQUERA (GFW)")
    print("=" * 40)

    try:
        fishing_events = fetcher.fetch_real_fishing_activity(start_str, end_str)
        all_data['fishing'] = [
            {
                'date': e.date,
                'lat': e.lat,
                'lon': e.lon,
                'fishing_hours': e.fishing_hours,
                'vessel_id': e.vessel_id,
                'flag_state': e.flag_state,
                'gear_type': e.gear_type,
                'source': e.source
            }
            for e in fishing_events
        ]
        all_data['metadata']['sources'].append('gfw_ais')
        print(f"✅ Fishing: {len(fishing_events)} eventos reales")
    except RealDataError as e:
        print(f"❌ Fishing Error: {e}")
        print()
        print("Sin datos de pesca reales, no podemos entrenar el modelo supervisado.")
        print("Por favor configura GFW_API_KEY y vuelve a ejecutar.")
        return None

    # Save to database
    print("\n" + "=" * 40)
    print("GUARDANDO DATOS")
    print("=" * 40)

    db_path = save_real_data_to_db(all_data, fetcher.cache_dir)

    # Summary
    print("\n" + "=" * 60)
    print("✅ DESCARGA COMPLETADA - 100% DATOS REALES")
    print("=" * 60)
    print()
    print(f"Base de datos: {db_path}")
    print()
    print("Estadísticas:")
    print(f"  - SST satelital: {len(all_data['sst'])} registros")
    print(f"  - Condiciones marinas: {len(all_data['marine'])} registros")
    print(f"  - Eventos de pesca: {len(all_data['fishing'])} eventos")
    print()
    print("Fuentes verificadas:")
    for source in all_data['metadata']['sources']:
        print(f"  ✓ {source}")
    print()
    print("Para ejecutar el análisis con estos datos:")
    print("  python main.py --supervised")

    return db_path


def save_real_data_to_db(data: dict, cache_dir: Path) -> Path:
    """Save real data to SQLite database."""
    db_path = cache_dir / "real_data_100.db"

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Drop old tables
    cursor.execute('DROP TABLE IF EXISTS sst')
    cursor.execute('DROP TABLE IF EXISTS marine')
    cursor.execute('DROP TABLE IF EXISTS fishing')
    cursor.execute('DROP TABLE IF EXISTS training')
    cursor.execute('DROP TABLE IF EXISTS metadata')

    # Create tables
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
        CREATE TABLE marine (
            id INTEGER PRIMARY KEY,
            date TEXT,
            lat REAL,
            lon REAL,
            wave_height REAL,
            wave_period REAL,
            wave_direction REAL,
            wind_speed REAL,
            wind_direction REAL,
            source TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE fishing (
            id INTEGER PRIMARY KEY,
            date TEXT,
            lat REAL,
            lon REAL,
            fishing_hours REAL,
            vessel_id TEXT,
            flag_state TEXT,
            gear_type TEXT,
            source TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE training (
            id INTEGER PRIMARY KEY,
            date TEXT,
            lat REAL,
            lon REAL,
            sst REAL,
            wave_height REAL,
            wave_period REAL,
            wind_speed REAL,
            wind_direction REAL,
            fishing_hours REAL,
            is_fishing INTEGER,
            month INTEGER,
            all_real INTEGER DEFAULT 1
        )
    ''')

    cursor.execute('''
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    ''')

    # Insert SST
    for record in data['sst']:
        cursor.execute(
            'INSERT INTO sst (date, lat, lon, sst, source) VALUES (?, ?, ?, ?, ?)',
            (record['date'], record['lat'], record['lon'], record['sst'], record['sst_source'])
        )

    # Insert Marine
    for record in data['marine']:
        cursor.execute(
            '''INSERT INTO marine (date, lat, lon, wave_height, wave_period, wave_direction, wind_speed, wind_direction, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (record['date'], record['lat'], record['lon'],
             record.get('wave_height'), record.get('wave_period'), record.get('wave_direction'),
             record.get('wind_speed'), record.get('wind_direction'), record['data_source'])
        )

    # Insert Fishing
    for event in data['fishing']:
        cursor.execute(
            '''INSERT INTO fishing (date, lat, lon, fishing_hours, vessel_id, flag_state, gear_type, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
            (event['date'], event['lat'], event['lon'], event['fishing_hours'],
             event.get('vessel_id'), event.get('flag_state'), event.get('gear_type'), event['source'])
        )

    # Build training table by merging
    # Create lookups
    sst_lookup = {}
    for r in data['sst']:
        key = (r['date'], round(r['lat'], 1), round(r['lon'], 1))
        sst_lookup[key] = r['sst']

    fishing_lookup = {}
    for e in data['fishing']:
        key = (e['date'], round(e['lat'], 1), round(e['lon'], 1))
        if key not in fishing_lookup:
            fishing_lookup[key] = 0
        fishing_lookup[key] += e['fishing_hours']

    training_count = 0
    positive_count = 0

    for record in data['marine']:
        key = (record['date'], round(record['lat'], 1), round(record['lon'], 1))
        sst = sst_lookup.get(key)
        if sst is None:
            continue

        fishing_hours = fishing_lookup.get(key, 0)
        is_fishing = 1 if fishing_hours > 0 else 0
        month = int(record['date'][5:7])

        cursor.execute(
            '''INSERT INTO training (date, lat, lon, sst, wave_height, wave_period, wind_speed, wind_direction, fishing_hours, is_fishing, month)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (record['date'], record['lat'], record['lon'], sst,
             record.get('wave_height'), record.get('wave_period'),
             record.get('wind_speed'), record.get('wind_direction'),
             fishing_hours, is_fishing, month)
        )

        training_count += 1
        positive_count += is_fishing

    # Metadata
    cursor.execute("INSERT INTO metadata VALUES ('data_type', 'REAL_ONLY_100_PERCENT')")
    cursor.execute("INSERT INTO metadata VALUES ('training_samples', ?)", (str(training_count),))
    cursor.execute("INSERT INTO metadata VALUES ('positive_samples', ?)", (str(positive_count),))
    cursor.execute("INSERT INTO metadata VALUES ('sources', ?)", (','.join(data['metadata']['sources']),))
    cursor.execute("INSERT INTO metadata VALUES ('created_at', ?)", (datetime.now().isoformat(),))

    # Indexes
    cursor.execute('CREATE INDEX idx_sst_date ON sst(date)')
    cursor.execute('CREATE INDEX idx_marine_date ON marine(date)')
    cursor.execute('CREATE INDEX idx_fishing_date ON fishing(date)')
    cursor.execute('CREATE INDEX idx_training_date ON training(date)')

    conn.commit()
    conn.close()

    print(f"✅ Training samples: {training_count}")
    print(f"✅ Positive (fishing): {positive_count} ({100*positive_count/training_count:.1f}%)")

    return db_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download 100% REAL data")
    parser.add_argument('--months', type=int, default=6, help='Months of data to download')

    args = parser.parse_args()

    # Check API key first
    if not check_gfw_api_key():
        return 1

    result = download_real_data(args.months)

    if result is None:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
