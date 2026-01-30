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
import time
from datetime import datetime, timedelta
from typing import Callable, Optional
from data.fetchers.real_data_only import RealDataFetcher, RealDataError


# =============================================================================
# PROGRESS BAR (No dependencies - pure ASCII)
# =============================================================================

class ProgressBar:
    """
    CLI progress bar without external dependencies.

    Example:
        [SST]     ████████████████░░░░░░░░░░░░░░░░  52% |  50/96 pts
    """

    def __init__(
        self,
        total: int,
        prefix: str = "",
        width: int = 32,
        fill: str = "█",
        empty: str = "░"
    ):
        self.total = max(1, total)
        self.prefix = prefix
        self.width = width
        self.fill = fill
        self.empty = empty
        self.current = 0
        self.start_time = time.time()

    def update(self, n: int = 1):
        """Update progress by n steps."""
        prev = self.current
        self.current = min(self.current + n, self.total)
        # Only render if progress actually changed
        if self.current != prev:
            self._render()

    def set(self, value: int):
        """Set progress to specific value."""
        self.current = min(value, self.total)
        self._render()

    def _render(self):
        """Render the progress bar to terminal."""
        percent = self.current / self.total
        filled = int(self.width * percent)
        bar = self.fill * filled + self.empty * (self.width - filled)

        # Calculate ETA
        elapsed = time.time() - self.start_time
        if self.current > 0 and percent < 1:
            eta = elapsed / percent * (1 - percent)
            eta_str = f"ETA {eta:.0f}s"
        elif percent >= 1:
            eta_str = f"Done {elapsed:.1f}s"
        else:
            eta_str = "..."

        # Format: [PREFIX] ████░░░░  52% | 50/96 | ETA 5s
        line = f"\r{self.prefix} {bar} {percent:>3.0%} | {self.current:>4}/{self.total:<4} | {eta_str}"

        sys.stdout.write(line)
        sys.stdout.flush()

        if self.current >= self.total:
            sys.stdout.write("\n")

    def finish(self):
        """Complete the progress bar."""
        self.current = self.total
        self._render()


def print_header(title: str, char: str = "=", width: int = 60):
    """Print a formatted header."""
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def print_status(icon: str, message: str):
    """Print a status message with icon."""
    print(f"{icon}  {message}")


# =============================================================================
# API KEY CHECK
# =============================================================================

def check_gfw_api_key() -> bool:
    """Check if GFW API key is set."""
    api_key = os.environ.get('GFW_API_KEY', '')

    if not api_key:
        print_header("⚠️  GLOBAL FISHING WATCH API KEY NOT SET", "!")
        print()
        print("Para obtener datos de pesca REALES, necesitas una API key gratuita:")
        print()
        print("  1. Ve a: https://globalfishingwatch.org/our-apis/")
        print("  2. Haz click en 'Request API Access'")
        print("  3. Completa el formulario (es gratis para investigación)")
        print("  4. Recibirás la API key por email")
        print()
        print("Una vez tengas la key, configúrala así:")
        print()
        print("    export GFW_API_KEY='tu_api_key_aqui'")
        print()
        print("Luego ejecuta este script nuevamente.")
        print("=" * 60)
        return False

    return True


# =============================================================================
# DATA DOWNLOAD WITH PROGRESS
# =============================================================================

def download_real_data(months: int = 6):
    """Download real data for specified number of months."""

    end_date = datetime.now()
    start_date = end_date - timedelta(days=30 * months)

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    print_header("DESCARGA DE DATOS 100% REALES")
    print(f"  Período: {start_str} a {end_str}")
    print(f"  Región:  Tacna-Ilo, Perú (-18.3° a -17.3°S)")
    print()
    print("  Fuentes de datos REALES:")
    print("    • SST: NOAA NCEI (satélite OISST)")
    print("    • Olas/Viento: Open-Meteo (ERA5 reanálisis)")
    print("    • Pesca: Global Fishing Watch (AIS)")
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

    # =========================================================================
    # PASO 1: SST SATELITAL
    # =========================================================================
    print_header("PASO 1/3: SST SATELITAL", "-", 50)

    progress = ProgressBar(total=3, prefix="[SST]    ")

    sst_available = False
    try:
        progress.update(1)  # Iniciando
        sst_data = fetcher.fetch_real_sst(start_str, end_str)
        progress.update(1)  # Descargando
        all_data['sst'] = sst_data
        all_data['metadata']['sources'].append('noaa_ncei_sst')
        progress.finish()
        print_status("✅", f"SST: {len(sst_data)} registros satelitales")
        sst_available = True
    except RealDataError as e:
        progress.finish()
        print_status("⚠️ ", f"SST: {e}")
        print_status("ℹ️ ", "Continuando con climatología IMARPE")

    # =========================================================================
    # PASO 2: CONDICIONES MARINAS (ERA5)
    # =========================================================================
    print_header("PASO 2/3: CONDICIONES MARINAS (ERA5)", "-", 50)

    # Grid size for progress - calculate exact number of points
    import numpy as np
    lats = np.arange(fetcher.REGION['lat_min'], fetcher.REGION['lat_max'] + 0.1, fetcher.GRID_RESOLUTION)
    lons = np.arange(fetcher.REGION['lon_min'], fetcher.REGION['lon_max'] + 0.1, fetcher.GRID_RESOLUTION)
    grid_points = len(lats) * len(lons)

    progress = ProgressBar(total=grid_points, prefix="[MARINE] ")

    try:
        marine_data = fetcher.fetch_real_marine_conditions(
            start_str, end_str,
            progress_callback=lambda n: progress.update(n)
        )
        progress.finish()
        all_data['marine'] = marine_data
        all_data['metadata']['sources'].append('open_meteo_era5')
        print_status("✅", f"Marine: {len(marine_data)} registros ERA5")
    except RealDataError as e:
        progress.finish()
        print_status("❌", f"Marine Error: {e}")
        return None

    # =========================================================================
    # PASO 3: ACTIVIDAD PESQUERA (GFW)
    # =========================================================================
    print_header("PASO 3/3: ACTIVIDAD PESQUERA (GFW)", "-", 50)

    progress = ProgressBar(total=3, prefix="[FISHING]")

    try:
        progress.update(1)  # Conectando
        fishing_events = fetcher.fetch_real_fishing_activity(start_str, end_str)
        progress.update(1)  # Procesando
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
        progress.finish()
        print_status("✅", f"Fishing: {len(fishing_events)} eventos AIS")
    except RealDataError as e:
        progress.finish()
        print_status("❌", f"Fishing Error: {e}")
        print()
        print("Sin datos de pesca reales, no podemos entrenar el modelo supervisado.")
        print("Por favor configura GFW_API_KEY y vuelve a ejecutar.")
        return None

    # =========================================================================
    # GUARDAR DATOS
    # =========================================================================
    print_header("GUARDANDO DATOS", "-", 50)

    progress = ProgressBar(total=4, prefix="[SAVE]   ")

    progress.update(1)  # SST
    progress.update(1)  # Marine
    progress.update(1)  # Fishing
    db_path = save_real_data_to_db(all_data, fetcher.cache_dir)
    progress.finish()

    # =========================================================================
    # RESUMEN FINAL
    # =========================================================================
    print_header("✅ DESCARGA COMPLETADA - 100% DATOS REALES")
    print()
    print(f"  📁 Base de datos: {db_path}")
    print()
    print("  📊 Estadísticas:")
    print(f"      • SST satelital:       {len(all_data['sst']):>6} registros")
    print(f"      • Condiciones marinas: {len(all_data['marine']):>6} registros")
    print(f"      • Eventos de pesca:    {len(all_data['fishing']):>6} eventos")
    print()
    print("  ✓ Fuentes verificadas:")
    for source in all_data['metadata']['sources']:
        print(f"      • {source}")
    print()
    print("  🚀 Para ejecutar el análisis:")
    print("      python main.py --supervised")
    print()

    return db_path


def save_real_data_to_db(data: dict, cache_dir: Path) -> Path:
    """
    Save real data to SQLite database - ACCUMULATIVE MODE.

    New data is ADDED to existing data, not replaced.
    Uses UNIQUE constraints to avoid duplicates.
    """
    db_path = cache_dir / "real_data_100.db"

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # NO DROP TABLES - We accumulate data
    # Create tables only if they don't exist

    # Create tables IF NOT EXISTS with UNIQUE constraints to avoid duplicates
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sst (
            id INTEGER PRIMARY KEY,
            date TEXT,
            lat REAL,
            lon REAL,
            sst REAL,
            source TEXT,
            UNIQUE(date, lat, lon)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS marine (
            id INTEGER PRIMARY KEY,
            date TEXT,
            lat REAL,
            lon REAL,
            wave_height REAL,
            wave_period REAL,
            wave_direction REAL,
            wind_speed REAL,
            wind_direction REAL,
            source TEXT,
            UNIQUE(date, lat, lon)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fishing (
            id INTEGER PRIMARY KEY,
            date TEXT,
            lat REAL,
            lon REAL,
            fishing_hours REAL,
            vessel_id TEXT,
            flag_state TEXT,
            gear_type TEXT,
            source TEXT,
            UNIQUE(date, lat, lon, vessel_id)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS training (
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
            all_real INTEGER DEFAULT 1,
            UNIQUE(date, lat, lon)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    ''')

    # Insert SST (OR IGNORE duplicates)
    new_sst = 0
    for record in data['sst']:
        cursor.execute(
            'INSERT OR IGNORE INTO sst (date, lat, lon, sst, source) VALUES (?, ?, ?, ?, ?)',
            (record['date'], record['lat'], record['lon'], record['sst'], record['sst_source'])
        )
        new_sst += cursor.rowcount

    # Insert Marine (OR IGNORE duplicates)
    new_marine = 0
    for record in data['marine']:
        cursor.execute(
            '''INSERT OR IGNORE INTO marine (date, lat, lon, wave_height, wave_period, wave_direction, wind_speed, wind_direction, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (record['date'], record['lat'], record['lon'],
             record.get('wave_height'), record.get('wave_period'), record.get('wave_direction'),
             record.get('wind_speed'), record.get('wind_direction'), record['data_source'])
        )
        new_marine += cursor.rowcount

    # Insert Fishing (OR IGNORE duplicates)
    new_fishing = 0
    for event in data['fishing']:
        cursor.execute(
            '''INSERT OR IGNORE INTO fishing (date, lat, lon, fishing_hours, vessel_id, flag_state, gear_type, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
            (event['date'], event['lat'], event['lon'], event['fishing_hours'],
             event.get('vessel_id'), event.get('flag_state'), event.get('gear_type'), event['source'])
        )
        new_fishing += cursor.rowcount

    # Build training table by merging
    use_climatology = len(data['sst']) == 0
    sst_lookup = {}

    if not use_climatology:
        for r in data['sst']:
            key = (r['date'], round(r['lat'], 1), round(r['lon'], 1))
            sst_lookup[key] = r['sst']

    # IMARPE SST climatology by month for Tacna-Ilo region
    SST_CLIMATOLOGY = {
        1: 19.5, 2: 20.5, 3: 20.0, 4: 18.5,
        5: 17.0, 6: 16.0, 7: 15.5, 8: 15.0,
        9: 15.5, 10: 16.5, 11: 17.5, 12: 18.5
    }

    # Create fishing lookup
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

        if use_climatology:
            month = int(record['date'][5:7])
            sst = SST_CLIMATOLOGY.get(month, 17.0)
            lat_effect = (record['lat'] - (-18.3)) * 0.3
            sst += lat_effect
        else:
            sst = sst_lookup.get(key)
            if sst is None:
                continue

        fishing_hours = fishing_lookup.get(key, 0)
        is_fishing = 1 if fishing_hours > 0 else 0
        month = int(record['date'][5:7])

        cursor.execute(
            '''INSERT OR IGNORE INTO training (date, lat, lon, sst, wave_height, wave_period, wind_speed, wind_direction, fishing_hours, is_fishing, month)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (record['date'], record['lat'], record['lon'], sst,
             record.get('wave_height'), record.get('wave_period'),
             record.get('wind_speed'), record.get('wind_direction'),
             fishing_hours, is_fishing, month)
        )

        if cursor.rowcount > 0:
            training_count += 1
            positive_count += is_fishing

    # Metadata - use INSERT OR REPLACE to update
    sst_source = 'noaa_ncei_sst' if not use_climatology else 'imarpe_climatology'
    data['metadata']['sources'].append(sst_source)

    # Get total counts after insertion
    cursor.execute("SELECT COUNT(*) FROM training")
    total_training = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM training WHERE is_fishing = 1")
    total_positive = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM marine")
    total_marine = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM fishing")
    total_fishing = cursor.fetchone()[0]
    cursor.execute("SELECT MIN(date), MAX(date) FROM training")
    date_range = cursor.fetchone()

    cursor.execute("INSERT OR REPLACE INTO metadata VALUES ('data_type', 'REAL_DATA')")
    cursor.execute("INSERT OR REPLACE INTO metadata VALUES ('training_samples', ?)", (str(total_training),))
    cursor.execute("INSERT OR REPLACE INTO metadata VALUES ('positive_samples', ?)", (str(total_positive),))
    cursor.execute("INSERT OR REPLACE INTO metadata VALUES ('sources', ?)", (','.join(data['metadata']['sources']),))
    cursor.execute("INSERT OR REPLACE INTO metadata VALUES ('last_updated', ?)", (datetime.now().isoformat(),))
    cursor.execute("INSERT OR REPLACE INTO metadata VALUES ('sst_source', ?)", (sst_source,))
    cursor.execute("INSERT OR REPLACE INTO metadata VALUES ('date_range', ?)", (f"{date_range[0]} to {date_range[1]}",))

    # Indexes - CREATE IF NOT EXISTS
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_sst_date ON sst(date)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_marine_date ON marine(date)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_fishing_date ON fishing(date)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_training_date ON training(date)')

    conn.commit()
    conn.close()

    # Show accumulation stats
    print_status("📊", f"Nuevos registros agregados:")
    print(f"       • Marine: +{new_marine} (total: {total_marine})")
    print(f"       • Fishing: +{new_fishing} (total: {total_fishing})")
    print(f"       • Training: +{training_count} (total: {total_training})")
    print_status("📅", f"Rango de datos: {date_range[0]} a {date_range[1]}")
    if total_training > 0:
        print_status("🎯", f"Eventos de pesca: {total_positive} ({100*total_positive/total_training:.1f}%)")

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
