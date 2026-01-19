#!/usr/bin/env python3
"""
DEPRECATED: This script estimates SST and simulates fishing activity.

Use instead: python scripts/download_100_real.py --months 6

While this script uses real marine data from Open-Meteo (ERA5),
it ESTIMATES SST and GENERATES synthetic fishing activity.
For 100% real data including real fishing activity from Global
Fishing Watch, use download_100_real.py instead.

To get REAL data:
1. Get your free Global Fishing Watch API key at:
   https://globalfishingwatch.org/our-apis/
2. Set: export GFW_API_KEY='your_key_here'
3. Run: python scripts/download_100_real.py --months 6
"""

import sys
print("=" * 60)
print("DEPRECATED SCRIPT")
print("=" * 60)
print()
print("This script uses ESTIMATED SST and SIMULATED fishing data.")
print()
print("Use instead:")
print("  python scripts/download_100_real.py --months 6")
print()
print("For more info, see the docstring in this file.")
print("=" * 60)
sys.exit(1)

# Original code below (not executed)
"""
Old script for reference - DO NOT USE

Open-Meteo provides:
- Real SST data from ERA5 reanalysis
- Real wave data
- Real current data (derived)
- No authentication required
- Data from 1940 to present
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import requests
import json
import sqlite3
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Region: Peru Tacna-Ilo
REGION = {
    'lat_min': -18.3,
    'lat_max': -17.3,
    'lon_min': -71.5,
    'lon_max': -70.8
}

# Grid points for sampling
GRID_POINTS = [
    # Coastal points (near shore)
    (-17.40, -71.45), (-17.50, -71.40), (-17.60, -71.38),
    (-17.70, -71.35), (-17.80, -71.30), (-17.90, -71.20),
    (-18.00, -71.10), (-18.10, -71.00), (-18.20, -70.90),
    # Offshore points
    (-17.45, -71.30), (-17.55, -71.25), (-17.65, -71.20),
    (-17.75, -71.15), (-17.85, -71.10), (-17.95, -71.00),
    (-18.05, -70.95), (-18.15, -70.88),
    # Far offshore
    (-17.50, -71.10), (-17.70, -71.00), (-17.90, -70.95),
]

# Known fishing hotspots with historical intensity
HOTSPOTS = [
    {"lat": -17.70, "lon": -71.35, "name": "Punta Coles", "intensity": 1.3},
    {"lat": -17.78, "lon": -71.14, "name": "Pozo Redondo", "intensity": 1.2},
    {"lat": -17.82, "lon": -71.10, "name": "Punta Blanca", "intensity": 1.25},
    {"lat": -17.93, "lon": -70.99, "name": "Ite", "intensity": 1.15},
    {"lat": -18.02, "lon": -70.93, "name": "Vila Vila", "intensity": 1.2},
    {"lat": -18.12, "lon": -70.86, "name": "Boca del Rio", "intensity": 1.1},
]


def fetch_open_meteo_historical(lat: float, lon: float, start_date: str, end_date: str) -> dict:
    """
    Fetch real historical marine data from Open-Meteo.

    Returns SST, wave height, wave period, wave direction.
    Data source: ERA5 reanalysis (ECMWF)
    """
    url = "https://marine-api.open-meteo.com/v1/marine"

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "wave_height_max",
            "wave_direction_dominant",
            "wave_period_max",
            "swell_wave_height_max",
            "swell_wave_period_max"
        ],
        "timezone": "America/Lima"
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"[WARN] API error for ({lat}, {lon}): {response.status_code}")
            return None
    except Exception as e:
        print(f"[WARN] Request failed for ({lat}, {lon}): {e}")
        return None


def fetch_open_meteo_archive(lat: float, lon: float, start_date: str, end_date: str) -> dict:
    """
    Fetch from Open-Meteo Archive API (longer historical data).
    Uses ERA5 reanalysis data.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "temperature_2m_mean",  # Air temp (proxy for SST trend)
            "wind_speed_10m_max",
            "wind_direction_10m_dominant",
            "precipitation_sum"
        ],
        "timezone": "America/Lima"
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def download_real_sst_from_noaa_api(start_date: str, end_date: str, output_dir: Path) -> list:
    """
    Try to get real SST from NOAA's alternative APIs.
    """
    print("\n[INFO] Attempting NOAA CoastWatch ERDDAP for real SST...")

    # CoastWatch West Coast Node - more reliable
    base_url = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg_LonPM180.json"

    # Build query for our region
    params = {
        "sst": f"[({start_date}T12:00:00Z):1:({end_date}T12:00:00Z)][(0.0):1:(0.0)][({REGION['lat_min']}):1:({REGION['lat_max']})][({REGION['lon_min']}):1:({REGION['lon_max']})]"
    }

    try:
        # Construct URL
        query = f"sst[({start_date}T12:00:00Z):1:({end_date}T12:00:00Z)][(0.0):1:(0.0)][({REGION['lat_min']}):1:({REGION['lat_max']})][({REGION['lon_min']}):1:({REGION['lon_max']})]"
        url = f"{base_url}?{query}"

        print(f"[INFO] Requesting: {url[:100]}...")
        response = requests.get(url, timeout=120)

        if response.status_code == 200:
            data = response.json()
            print(f"[OK] Retrieved real NOAA SST data")
            return data
        else:
            print(f"[WARN] NOAA API returned: {response.status_code}")
            return None
    except Exception as e:
        print(f"[WARN] NOAA API failed: {e}")
        return None


def download_all_historical_data(years: int, output_dir: Path):
    """Download all available real historical data."""

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years)

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    print("=" * 60)
    print("DOWNLOADING REAL HISTORICAL DATA")
    print("=" * 60)
    print(f"Period: {start_str} to {end_str}")
    print(f"Grid points: {len(GRID_POINTS)}")
    print(f"Source: Open-Meteo (ERA5 Reanalysis)")

    output_dir.mkdir(parents=True, exist_ok=True)

    all_data = []

    # Download data for each grid point
    print(f"\n[INFO] Downloading marine data for {len(GRID_POINTS)} points...")

    for i, (lat, lon) in enumerate(GRID_POINTS):
        print(f"[{i+1}/{len(GRID_POINTS)}] Fetching ({lat:.2f}, {lon:.2f})...", end=" ")

        # Get marine data (waves)
        marine_data = fetch_open_meteo_historical(lat, lon, start_str, end_str)

        # Get archive data (wind for upwelling calculation)
        archive_data = fetch_open_meteo_archive(lat, lon, start_str, end_str)

        if marine_data and 'daily' in marine_data:
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
                    'swell_height': daily.get('swell_wave_height_max', [None])[t_idx],
                    'swell_period': daily.get('swell_wave_period_max', [None])[t_idx],
                }

                # Add archive data if available
                if archive_data and 'daily' in archive_data:
                    arch_daily = archive_data['daily']
                    if t_idx < len(arch_daily.get('temperature_2m_mean', [])):
                        record['air_temp'] = arch_daily['temperature_2m_mean'][t_idx]
                        record['wind_speed'] = arch_daily.get('wind_speed_10m_max', [None])[t_idx]
                        record['wind_direction'] = arch_daily.get('wind_direction_10m_dominant', [None])[t_idx]

                all_data.append(record)

            print(f"OK ({len(times)} days)")
        else:
            print("FAILED")

        # Rate limiting
        time.sleep(0.5)

    print(f"\n[OK] Downloaded {len(all_data)} data points")

    # Save raw data
    raw_file = output_dir / "real_marine_data.json"
    with open(raw_file, 'w') as f:
        json.dump(all_data, f)
    print(f"[OK] Saved raw data: {raw_file}")

    return all_data


def estimate_sst_from_data(records: list) -> list:
    """
    Estimate SST from available data using oceanographic relationships.

    SST correlations:
    - Air temperature (lagged correlation ~0.7)
    - Wave height (inverse - higher waves = mixing = cooler)
    - Wind speed (upwelling indicator)
    - Season (Humboldt Current patterns)
    """
    print("\n[INFO] Estimating SST from oceanographic relationships...")

    # Humboldt Current SST climatology by month
    # Based on IMARPE data for Tacna-Ilo region
    SST_CLIMATOLOGY = {
        1: 19.5, 2: 20.5, 3: 20.0, 4: 18.5,  # Summer/Fall
        5: 17.0, 6: 16.0, 7: 15.5, 8: 15.0,  # Winter (upwelling peak)
        9: 15.5, 10: 16.5, 11: 17.5, 12: 18.5  # Spring
    }

    for record in records:
        if record.get('date'):
            month = int(record['date'][5:7])

            # Base SST from climatology
            base_sst = SST_CLIMATOLOGY.get(month, 17.0)

            # Air temp adjustment (if available)
            if record.get('air_temp'):
                # SST typically 1-3°C cooler than air temp in this region
                air_adjustment = (record['air_temp'] - 20) * 0.3
                base_sst += air_adjustment

            # Wave mixing effect (higher waves = cooler due to mixing)
            if record.get('wave_height'):
                wave_cooling = -0.3 * max(0, record['wave_height'] - 1.5)
                base_sst += wave_cooling

            # Wind upwelling effect (strong south wind = upwelling = cooler)
            if record.get('wind_speed') and record.get('wind_direction'):
                wind_dir = record['wind_direction']
                wind_speed = record['wind_speed']
                # South wind (160-200°) causes upwelling
                if 160 <= wind_dir <= 200 and wind_speed > 5:
                    upwelling_cooling = -0.5 * (wind_speed - 5) / 10
                    base_sst += upwelling_cooling

            # Coastal vs offshore (cooler near coast due to upwelling)
            lon = record['lon']
            coast_distance = abs(lon - (-71.4))
            coast_effect = -1.5 * np.exp(-coast_distance / 0.2)
            base_sst += coast_effect

            # Random daily variability
            daily_var = np.random.normal(0, 0.3)

            record['sst'] = round(np.clip(base_sst + daily_var, 13, 25), 2)
            record['sst_anomaly'] = round(record['sst'] - SST_CLIMATOLOGY[month], 2)

    print(f"[OK] SST estimated for {len(records)} records")
    return records


def generate_fishing_from_real_conditions(records: list) -> list:
    """
    Generate fishing activity based on REAL oceanographic conditions.

    Uses actual wave, wind, and estimated SST data to predict
    where fishing would occur based on known patterns.
    """
    print("\n[INFO] Generating fishing activity from real conditions...")

    fishing_events = []

    # Group by date
    by_date = {}
    for r in records:
        date = r.get('date')
        if date:
            if date not in by_date:
                by_date[date] = []
            by_date[date].append(r)

    for date, day_records in by_date.items():
        month = int(date[5:7])

        # Seasonal fishing intensity
        if 2 <= month <= 5:
            season_factor = 1.5  # Peak season
        elif 6 <= month <= 8:
            season_factor = 0.6  # Low season
        else:
            season_factor = 1.0

        for record in day_records:
            # Skip if conditions are too rough
            wave_height = record.get('wave_height', 1.5)
            if wave_height and wave_height > 3.0:
                continue  # Too rough to fish

            sst = record.get('sst', 17)
            lat, lon = record['lat'], record['lon']

            # Check proximity to hotspots
            for hotspot in HOTSPOTS:
                dist = np.sqrt((lat - hotspot['lat'])**2 + (lon - hotspot['lon'])**2)

                if dist < 0.15:  # Within ~15km of hotspot
                    # Calculate fishing probability based on conditions

                    # SST score (optimal 15-20°C)
                    sst_score = 1.0 if 15 <= sst <= 20 else 0.5

                    # Wave score (calmer = better)
                    wave_score = max(0, 1 - (wave_height - 1) / 2) if wave_height else 0.8

                    # Proximity score
                    proximity_score = 1 - (dist / 0.15)

                    # Combined probability
                    prob = hotspot['intensity'] * season_factor * sst_score * wave_score * proximity_score

                    # Generate fishing events
                    n_events = np.random.poisson(prob * 3)

                    for _ in range(n_events):
                        fishing_events.append({
                            'date': date,
                            'lat': round(lat + np.random.normal(0, 0.02), 4),
                            'lon': round(lon + np.random.normal(0, 0.02), 4),
                            'fishing_hours': round(np.random.exponential(2.5) * hotspot['intensity'], 2),
                            'hotspot': hotspot['name'],
                            'sst': sst,
                            'wave_height': wave_height,
                            'conditions_based': True
                        })

    print(f"[OK] Generated {len(fishing_events)} fishing events from real conditions")
    return fishing_events


def build_training_database(records: list, fishing_events: list, output_dir: Path):
    """Build SQLite database with real data."""
    print("\n[INFO] Building training database with real data...")

    db_path = output_dir / "real_training_data.db"

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('DROP TABLE IF EXISTS training_samples')
    cursor.execute('DROP TABLE IF EXISTS fishing_events')
    cursor.execute('DROP TABLE IF EXISTS metadata')

    cursor.execute('''
        CREATE TABLE training_samples (
            id INTEGER PRIMARY KEY,
            date TEXT,
            lat REAL,
            lon REAL,
            sst REAL,
            sst_anomaly REAL,
            wave_height REAL,
            wave_period REAL,
            wave_direction REAL,
            wind_speed REAL,
            wind_direction REAL,
            air_temp REAL,
            fishing_hours REAL,
            is_fishing INTEGER,
            month INTEGER,
            data_source TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE fishing_events (
            id INTEGER PRIMARY KEY,
            date TEXT,
            lat REAL,
            lon REAL,
            fishing_hours REAL,
            hotspot TEXT,
            sst REAL,
            wave_height REAL
        )
    ''')

    cursor.execute('''
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    ''')

    # Create fishing lookup
    fishing_lookup = {}
    for event in fishing_events:
        key = (event['date'], round(event['lat'], 1), round(event['lon'], 1))
        if key not in fishing_lookup:
            fishing_lookup[key] = 0
        fishing_lookup[key] += event['fishing_hours']

    # Insert training samples
    samples = []
    for record in records:
        key = (record['date'], round(record['lat'], 1), round(record['lon'], 1))
        fishing_hours = fishing_lookup.get(key, 0)

        month = int(record['date'][5:7]) if record.get('date') else 1

        samples.append((
            record.get('date'),
            record.get('lat'),
            record.get('lon'),
            record.get('sst'),
            record.get('sst_anomaly', 0),
            record.get('wave_height'),
            record.get('wave_period'),
            record.get('wave_direction'),
            record.get('wind_speed'),
            record.get('wind_direction'),
            record.get('air_temp'),
            fishing_hours,
            1 if fishing_hours > 0 else 0,
            month,
            'open_meteo_era5'
        ))

    cursor.executemany('''
        INSERT INTO training_samples
        (date, lat, lon, sst, sst_anomaly, wave_height, wave_period, wave_direction,
         wind_speed, wind_direction, air_temp, fishing_hours, is_fishing, month, data_source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', samples)

    # Insert fishing events
    for event in fishing_events:
        cursor.execute('''
            INSERT INTO fishing_events (date, lat, lon, fishing_hours, hotspot, sst, wave_height)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (event['date'], event['lat'], event['lon'], event['fishing_hours'],
              event['hotspot'], event.get('sst'), event.get('wave_height')))

    # Metadata
    positive = sum(1 for s in samples if s[12] == 1)
    cursor.execute("INSERT INTO metadata VALUES ('total_samples', ?)", (str(len(samples)),))
    cursor.execute("INSERT INTO metadata VALUES ('positive_samples', ?)", (str(positive),))
    cursor.execute("INSERT INTO metadata VALUES ('fishing_events', ?)", (str(len(fishing_events)),))
    cursor.execute("INSERT INTO metadata VALUES ('data_source', 'Open-Meteo ERA5 + estimated SST')")
    cursor.execute("INSERT INTO metadata VALUES ('created_at', ?)", (datetime.now().isoformat(),))

    # Indexes
    cursor.execute('CREATE INDEX idx_date ON training_samples(date)')
    cursor.execute('CREATE INDEX idx_location ON training_samples(lat, lon)')
    cursor.execute('CREATE INDEX idx_fishing ON training_samples(is_fishing)')

    conn.commit()
    conn.close()

    print(f"\n[OK] Database created: {db_path}")
    print(f"[OK] Total samples: {len(samples):,}")
    print(f"[OK] Positive samples: {positive:,} ({100*positive/len(samples):.1f}%)")
    print(f"[OK] Fishing events: {len(fishing_events):,}")

    return db_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download REAL historical marine data")
    parser.add_argument('--years', type=int, default=2, help='Years of data (max 4 recommended)')
    parser.add_argument('--output', type=str, default='data/historical', help='Output directory')

    args = parser.parse_args()

    output_dir = Path(args.output)

    # Download real data
    records = download_all_historical_data(args.years, output_dir)

    if not records:
        print("[ERROR] No data downloaded")
        return 1

    # Estimate SST from real conditions
    records = estimate_sst_from_data(records)

    # Generate fishing based on real conditions
    fishing_events = generate_fishing_from_real_conditions(records)

    # Build database
    db_path = build_training_database(records, fishing_events, output_dir)

    # Save fishing events
    fishing_file = output_dir / "real_fishing_events.json"
    with open(fishing_file, 'w') as f:
        json.dump(fishing_events, f)

    print(f"\n{'='*60}")
    print("REAL DATA DOWNLOAD COMPLETE")
    print("="*60)
    print(f"\nData source: Open-Meteo (ERA5 Reanalysis)")
    print(f"SST: Estimated from air temp, waves, wind, climatology")
    print(f"Fishing: Generated from real oceanographic conditions")
    print(f"\nFiles:")
    print(f"  - Database: {db_path}")
    print(f"  - Fishing events: {fishing_file}")
    print(f"\nTo train with this data:")
    print(f"  python main.py --supervised")

    return 0


if __name__ == "__main__":
    sys.exit(main())
