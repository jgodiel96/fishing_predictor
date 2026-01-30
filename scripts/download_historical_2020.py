#!/usr/bin/env python3
"""
Download historical data from 2020 to present.
Accumulative mode - adds to existing database.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import sqlite3
import requests
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict

# =============================================================================
# CONFIGURATION
# =============================================================================

REGION = {
    'lat_min': -18.3,
    'lat_max': -17.3,
    'lon_min': -71.5,
    'lon_max': -70.8
}

GRID_RESOLUTION = 0.1
DB_PATH = Path("data/real_only/real_data_100.db")

# IMARPE SST climatology by month
SST_CLIMATOLOGY = {
    1: 19.5, 2: 20.5, 3: 20.0, 4: 18.5,
    5: 17.0, 6: 16.0, 7: 15.5, 8: 15.0,
    9: 15.5, 10: 16.5, 11: 17.5, 12: 18.5
}


# =============================================================================
# PROGRESS BAR
# =============================================================================

class ProgressBar:
    def __init__(self, total: int, prefix: str = "", width: int = 40):
        self.total = max(1, total)
        self.prefix = prefix
        self.width = width
        self.current = 0
        self.start_time = time.time()

    def update(self, n: int = 1):
        prev = self.current
        self.current = min(self.current + n, self.total)
        if self.current != prev:
            self._render()

    def _render(self):
        percent = self.current / self.total
        filled = int(self.width * percent)
        bar = "█" * filled + "░" * (self.width - filled)

        elapsed = time.time() - self.start_time
        if self.current > 0 and percent < 1:
            eta = elapsed / percent * (1 - percent)
            eta_str = f"ETA {eta/60:.0f}m" if eta > 60 else f"ETA {eta:.0f}s"
        else:
            eta_str = f"{elapsed/60:.1f}m"

        sys.stdout.write(f"\r{self.prefix} {bar} {percent:>3.0%} | {self.current}/{self.total} | {eta_str}   ")
        sys.stdout.flush()

        if self.current >= self.total:
            sys.stdout.write("\n")

    def finish(self):
        self.current = self.total
        self._render()


# =============================================================================
# DATABASE
# =============================================================================

def init_database():
    """Initialize database with proper schema."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

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

    cursor.execute('CREATE INDEX IF NOT EXISTS idx_marine_date ON marine(date)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_training_date ON training(date)')

    conn.commit()
    conn.close()


# =============================================================================
# FETCH MARINE DATA (Open-Meteo ERA5)
# =============================================================================

def fetch_marine_month(year: int, month: int, progress: Optional[ProgressBar] = None) -> List[Dict]:
    """Fetch marine data for one month."""

    start_date = f"{year}-{month:02d}-01"
    if month == 12:
        end_date = f"{year}-12-31"
    else:
        end_date = f"{year}-{month+1:02d}-01"
        # Subtract one day
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=1)
        end_date = end_dt.strftime("%Y-%m-%d")

    records = []
    lats = np.arange(REGION['lat_min'], REGION['lat_max'] + 0.1, GRID_RESOLUTION)
    lons = np.arange(REGION['lon_min'], REGION['lon_max'] + 0.1, GRID_RESOLUTION)

    for lat in lats:
        for lon in lons:
            try:
                # Marine API
                marine_url = "https://marine-api.open-meteo.com/v1/marine"
                marine_params = {
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": start_date,
                    "end_date": end_date,
                    "daily": ["wave_height_max", "wave_direction_dominant", "wave_period_max"],
                    "timezone": "America/Lima"
                }

                # Archive API (wind)
                archive_url = "https://archive-api.open-meteo.com/v1/archive"
                archive_params = {
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": start_date,
                    "end_date": end_date,
                    "daily": ["wind_speed_10m_max", "wind_direction_10m_dominant"],
                    "timezone": "America/Lima"
                }

                marine_resp = requests.get(marine_url, params=marine_params, timeout=30)
                archive_resp = requests.get(archive_url, params=archive_params, timeout=30)

                if marine_resp.status_code == 200:
                    marine_data = marine_resp.json()
                    archive_data = archive_resp.json() if archive_resp.status_code == 200 else None

                    if 'daily' in marine_data:
                        daily = marine_data['daily']
                        times = daily.get('time', [])

                        for t_idx, date in enumerate(times):
                            record = {
                                'date': date,
                                'lat': lat,
                                'lon': lon,
                                'wave_height': daily.get('wave_height_max', [None])[t_idx],
                                'wave_direction': daily.get('wave_direction_dominant', [None])[t_idx],
                                'wave_period': daily.get('wave_period_max', [None])[t_idx],
                                'wind_speed': None,
                                'wind_direction': None,
                                'source': 'open_meteo_era5'
                            }

                            if archive_data and 'daily' in archive_data:
                                arch = archive_data['daily']
                                if t_idx < len(arch.get('wind_speed_10m_max', [])):
                                    record['wind_speed'] = arch.get('wind_speed_10m_max', [None])[t_idx]
                                    record['wind_direction'] = arch.get('wind_direction_10m_dominant', [None])[t_idx]

                            records.append(record)

                time.sleep(0.25)  # Rate limiting

            except Exception as e:
                pass  # Continue on error

            if progress:
                progress.update(1)

    return records


# =============================================================================
# FETCH FISHING DATA (GFW)
# =============================================================================

def fetch_fishing_month(year: int, month: int, api_key: str) -> List[Dict]:
    """Fetch fishing activity for one month from GFW."""

    start_date = f"{year}-{month:02d}-01"
    if month == 12:
        end_date = f"{year}-12-31"
    else:
        end_date = f"{year}-{month+1:02d}-01"
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=1)
        end_date = end_dt.strftime("%Y-%m-%d")

    headers = {"Authorization": f"Bearer {api_key}"}

    polygon = {
        "type": "Polygon",
        "coordinates": [[
            [REGION['lon_min'], REGION['lat_min']],
            [REGION['lon_max'], REGION['lat_min']],
            [REGION['lon_max'], REGION['lat_max']],
            [REGION['lon_min'], REGION['lat_max']],
            [REGION['lon_min'], REGION['lat_min']]
        ]]
    }

    try:
        response = requests.post(
            "https://gateway.api.globalfishingwatch.org/v3/4wings/report",
            headers=headers,
            json={
                "datasets": ["public-global-fishing-effort:latest"],
                "date-range": [start_date, end_date],
                "region": polygon,
                "spatial-resolution": "low",
                "temporal-resolution": "daily",
                "group-by": ["flag", "geartype"]
            },
            timeout=120
        )

        if response.status_code == 200:
            data = response.json()
            entries = data.get('entries', [])

            records = []
            for entry in entries:
                records.append({
                    'date': entry.get('date', entry.get('timestamp', start_date))[:10],
                    'lat': entry.get('lat', (REGION['lat_min'] + REGION['lat_max']) / 2),
                    'lon': entry.get('lon', (REGION['lon_min'] + REGION['lon_max']) / 2),
                    'fishing_hours': entry.get('hours', entry.get('fishing_hours', 0)),
                    'vessel_id': entry.get('vessel_id'),
                    'flag_state': entry.get('flag'),
                    'gear_type': entry.get('geartype'),
                    'source': 'gfw_ais'
                })

            return records

    except Exception as e:
        print(f"\n  ⚠️ GFW error {year}-{month:02d}: {e}")

    return []


# =============================================================================
# SAVE TO DATABASE
# =============================================================================

def save_month_data(marine_records: List[Dict], fishing_records: List[Dict], year: int, month: int):
    """Save one month of data to database."""

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Insert marine
    for r in marine_records:
        cursor.execute('''
            INSERT OR IGNORE INTO marine
            (date, lat, lon, wave_height, wave_period, wave_direction, wind_speed, wind_direction, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (r['date'], r['lat'], r['lon'], r.get('wave_height'), r.get('wave_period'),
              r.get('wave_direction'), r.get('wind_speed'), r.get('wind_direction'), r['source']))

    # Insert fishing
    for r in fishing_records:
        cursor.execute('''
            INSERT OR IGNORE INTO fishing
            (date, lat, lon, fishing_hours, vessel_id, flag_state, gear_type, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (r['date'], r['lat'], r['lon'], r['fishing_hours'],
              r.get('vessel_id'), r.get('flag_state'), r.get('gear_type'), r['source']))

    # Build training records
    fishing_lookup = {}
    for r in fishing_records:
        key = (r['date'], round(r['lat'], 1), round(r['lon'], 1))
        fishing_lookup[key] = fishing_lookup.get(key, 0) + r['fishing_hours']

    for r in marine_records:
        key = (r['date'], round(r['lat'], 1), round(r['lon'], 1))
        m = int(r['date'][5:7])

        # SST from climatology
        sst = SST_CLIMATOLOGY.get(m, 17.0)
        lat_effect = (r['lat'] - (-18.3)) * 0.3
        sst += lat_effect

        fishing_hours = fishing_lookup.get(key, 0)
        is_fishing = 1 if fishing_hours > 0 else 0

        cursor.execute('''
            INSERT OR IGNORE INTO training
            (date, lat, lon, sst, wave_height, wave_period, wind_speed, wind_direction, fishing_hours, is_fishing, month)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (r['date'], r['lat'], r['lon'], sst, r.get('wave_height'), r.get('wave_period'),
              r.get('wind_speed'), r.get('wind_direction'), fishing_hours, is_fishing, m))

    conn.commit()
    conn.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  DESCARGA HISTORICA DESDE 2020")
    print("=" * 70)

    # Check API key
    api_key = os.environ.get('GFW_API_KEY', '')
    if not api_key:
        print("\n❌ GFW_API_KEY no configurada")
        print("   export GFW_API_KEY='tu_api_key'")
        return 1

    # Initialize database
    init_database()

    # Date range
    start_year, start_month = 2020, 1
    end_year, end_month = 2026, 1

    # Calculate total months
    total_months = (end_year - start_year) * 12 + (end_month - start_month) + 1

    # Grid points per month
    lats = np.arange(REGION['lat_min'], REGION['lat_max'] + 0.1, GRID_RESOLUTION)
    lons = np.arange(REGION['lon_min'], REGION['lon_max'] + 0.1, GRID_RESOLUTION)
    points_per_month = len(lats) * len(lons)

    print(f"\n📅 Período: {start_year}-01 a {end_year}-{end_month:02d}")
    print(f"📊 Meses a procesar: {total_months}")
    print(f"🌐 Puntos de grid por mes: {points_per_month}")
    print(f"⏱️  Tiempo estimado: {total_months * 3:.0f} minutos (~{total_months * 3 / 60:.1f} horas)")
    print("\n" + "=" * 70)

    # Process each month
    current_month = 0
    year, month = start_year, start_month

    while (year < end_year) or (year == end_year and month <= end_month):
        current_month += 1

        print(f"\n[{current_month}/{total_months}] {year}-{month:02d}")

        # Progress bar for marine data
        progress = ProgressBar(points_per_month, prefix="  Marine")
        marine_records = fetch_marine_month(year, month, progress)
        progress.finish()

        # Fetch fishing data
        print(f"  Fishing...", end=" ", flush=True)
        fishing_records = fetch_fishing_month(year, month, api_key)
        print(f"✓ {len(fishing_records)} eventos")

        # Save to database
        save_month_data(marine_records, fishing_records, year, month)
        print(f"  💾 Guardado: {len(marine_records)} marine, {len(fishing_records)} fishing")

        # Next month
        month += 1
        if month > 12:
            month = 1
            year += 1

    # Final stats
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM training")
    total_training = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM fishing")
    total_fishing = cursor.fetchone()[0]
    cursor.execute("SELECT MIN(date), MAX(date) FROM training")
    date_range = cursor.fetchone()
    cursor.execute("SELECT COUNT(*) FROM training WHERE is_fishing = 1")
    positive = cursor.fetchone()[0]
    conn.close()

    print("\n" + "=" * 70)
    print("  ✅ DESCARGA COMPLETADA")
    print("=" * 70)
    print(f"\n📊 Base de datos: {DB_PATH}")
    print(f"   • Training: {total_training:,} registros")
    print(f"   • Fishing: {total_fishing:,} eventos")
    print(f"   • Con pesca: {positive:,} ({100*positive/total_training:.1f}%)")
    print(f"   • Rango: {date_range[0]} a {date_range[1]}")
    print(f"\n🚀 Para usar: python main.py --supervised")

    return 0


if __name__ == "__main__":
    sys.exit(main())
