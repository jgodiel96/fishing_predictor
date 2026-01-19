#!/usr/bin/env python3
"""
DEPRECATED: This script uses synthetic/simulated data.

Use instead: python scripts/download_100_real.py --months 6

This script is kept for reference only. It generates synthetic fishing
activity which is not suitable for production use.

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
print("This script uses SYNTHETIC data and is deprecated.")
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

Downloads:
- NOAA OISST: 4 years of daily SST data (no auth required)
- Simulated fishing activity based on known hotspots

Usage:
    python scripts/download_historical.py --years 4
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from erddapy import ERDDAP
import sqlite3
import json

# Region: Peru Tacna-Ilo
REGION = {
    'lat_min': -18.3,
    'lat_max': -17.3,
    'lon_min': -71.5,
    'lon_max': -70.8
}

# Known fishing hotspots
HOTSPOTS = [
    (-17.70, -71.35, 1.3, "Punta Coles"),
    (-17.78, -71.14, 1.2, "Pozo Redondo"),
    (-17.82, -71.10, 1.25, "Punta Blanca"),
    (-17.93, -70.99, 1.15, "Ite"),
    (-18.02, -70.93, 1.2, "Vila Vila"),
    (-18.12, -70.86, 1.1, "Boca del Rio"),
]


def download_noaa_sst(start_date: str, end_date: str, output_dir: Path) -> xr.Dataset:
    """Download SST from NOAA ERDDAP using direct URL."""
    print(f"\n{'='*60}")
    print("DOWNLOADING NOAA OISST DATA")
    print(f"{'='*60}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Region: Lat {REGION['lat_min']} to {REGION['lat_max']}")
    print(f"        Lon {REGION['lon_min']} to {REGION['lon_max']}")

    cache_file = output_dir / f"noaa_sst_{start_date}_{end_date}.nc"

    if cache_file.exists():
        print(f"\n[INFO] Loading cached data: {cache_file}")
        return xr.open_dataset(cache_file)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Use direct OPeNDAP URL for more reliable access
    print("\n[INFO] Connecting to NOAA THREDDS server via OPeNDAP...")

    # NOAA OISST via THREDDS OPeNDAP
    base_url = "https://www.ncei.noaa.gov/thredds/dodsC/OisatOisst21Agg/oisst-avhrr-v02r01.latest.nc"

    try:
        print("[INFO] Opening remote dataset...")
        ds = xr.open_dataset(base_url)

        print("[INFO] Subsetting data (this may take several minutes)...")

        # Select region and time range
        ds_subset = ds.sel(
            lat=slice(REGION['lat_max'], REGION['lat_min']),  # lat is descending
            lon=slice(REGION['lon_min'] + 360, REGION['lon_max'] + 360),  # Convert to 0-360
            time=slice(start_date, end_date),
            zlev=0
        )

        # Convert longitude back to -180 to 180
        ds_subset = ds_subset.assign_coords(lon=(ds_subset.lon - 360))

        # Load into memory
        print("[INFO] Loading data into memory...")
        ds_subset = ds_subset.load()

        n_times = len(ds_subset.time) if 'time' in ds_subset.dims else 0
        print(f"[OK] Downloaded {n_times} time steps")

        # Save to cache
        ds_subset.to_netcdf(cache_file)
        print(f"[OK] Saved to: {cache_file}")
        print(f"[OK] File size: {cache_file.stat().st_size / 1024 / 1024:.1f} MB")

        return ds_subset

    except Exception as e1:
        print(f"[WARN] OPeNDAP failed: {e1}")
        print("[INFO] Trying alternative: ERDDAP with fresh connection...")

        try:
            # Alternative: Use erddapy with fresh initialization
            from erddapy import ERDDAP

            e = ERDDAP(
                server="https://coastwatch.pfeg.noaa.gov/erddap",
                protocol="griddap"
            )

            e.dataset_id = "ncdcOisst21Agg_LonPM180"
            e.griddap_initialize()

            e.constraints = {
                "time>=": start_date,
                "time<=": end_date,
                "latitude>=": REGION['lat_min'],
                "latitude<=": REGION['lat_max'],
                "longitude>=": REGION['lon_min'],
                "longitude<=": REGION['lon_max'],
                "zlev=": 0
            }

            e.variables = ["sst", "anom"]

            print("[INFO] Downloading via CoastWatch ERDDAP...")
            ds = e.to_xarray()

            ds.to_netcdf(cache_file)

            n_times = len(ds.time) if 'time' in ds.dims else 0
            print(f"[OK] Downloaded {n_times} time steps")
            print(f"[OK] Saved to: {cache_file}")

            return ds

        except Exception as e2:
            print(f"[WARN] ERDDAP also failed: {e2}")
            print("[INFO] Generating synthetic SST data based on climatology...")

            return generate_synthetic_sst(start_date, end_date, output_dir)


def generate_synthetic_sst(start_date: str, end_date: str, output_dir: Path) -> xr.Dataset:
    """Generate synthetic SST data based on Humboldt Current climatology."""
    print("\n[INFO] Generating synthetic SST data...")
    print("       Based on Humboldt Current climatological patterns")

    # Create grid
    lats = np.arange(REGION['lat_min'], REGION['lat_max'] + 0.25, 0.25)
    lons = np.arange(REGION['lon_min'], REGION['lon_max'] + 0.25, 0.25)

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    dates = []
    current = start
    while current <= end:
        dates.append(current)
        current += timedelta(days=1)

    times = np.array(dates, dtype='datetime64[ns]')

    print(f"[INFO] Grid: {len(lats)} x {len(lons)}")
    print(f"[INFO] Days: {len(times)}")

    # Generate SST based on:
    # - Seasonal cycle (warmer in summer Dec-Mar, cooler in winter Jun-Aug)
    # - Latitude gradient (cooler in south due to upwelling)
    # - Distance from coast (cooler near coast due to upwelling)
    # - Random interannual variability

    sst_data = np.zeros((len(times), len(lats), len(lons)))
    anom_data = np.zeros((len(times), len(lats), len(lons)))

    # Climatological mean for the region
    base_sst = 17.5  # °C

    for t_idx, date in enumerate(dates):
        # Seasonal component (amplitude ~3°C)
        day_of_year = date.timetuple().tm_yday
        seasonal = 3.0 * np.cos(2 * np.pi * (day_of_year - 45) / 365)  # Peak in Feb

        # Interannual variability
        year_noise = np.random.normal(0, 0.5)

        for lat_idx, lat in enumerate(lats):
            for lon_idx, lon in enumerate(lons):
                # Latitude effect (cooler south)
                lat_effect = (lat - REGION['lat_min']) * 0.3  # ~0.3°C per degree

                # Distance from coast effect (upwelling - cooler near coast)
                dist_coast = abs(lon - (-71.4))  # Coast at ~-71.4
                coast_effect = -2.0 * np.exp(-dist_coast / 0.3)  # Cooler near coast

                # Random daily/spatial variability
                noise = np.random.normal(0, 0.3)

                sst = base_sst + seasonal + lat_effect + coast_effect + year_noise + noise
                sst = np.clip(sst, 13, 25)  # Realistic range

                sst_data[t_idx, lat_idx, lon_idx] = sst
                anom_data[t_idx, lat_idx, lon_idx] = seasonal + year_noise + noise

    # Create xarray Dataset
    ds = xr.Dataset(
        {
            'sst': (['time', 'latitude', 'longitude'], sst_data),
            'anom': (['time', 'latitude', 'longitude'], anom_data),
        },
        coords={
            'time': times,
            'latitude': lats,
            'longitude': lons,
        },
        attrs={
            'title': 'Synthetic SST data for Peru coast',
            'source': 'Generated from Humboldt Current climatology',
            'institution': 'Fishing Predictor Project',
        }
    )

    # Save
    cache_file = output_dir / f"synthetic_sst_{start_date}_{end_date}.nc"
    ds.to_netcdf(cache_file)

    print(f"[OK] Generated {len(times)} days of synthetic SST")
    print(f"[OK] Saved to: {cache_file}")
    print(f"[OK] File size: {cache_file.stat().st_size / 1024 / 1024:.1f} MB")

    return ds


def generate_fishing_activity(start_date: str, end_date: str) -> list:
    """Generate simulated fishing activity based on known patterns."""
    print(f"\n{'='*60}")
    print("GENERATING FISHING ACTIVITY DATA")
    print(f"{'='*60}")
    print("[INFO] Using known hotspots from IMARPE and local knowledge")

    events = []
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    current = start
    total_days = (end - start).days

    print(f"[INFO] Generating events for {total_days} days...")

    while current <= end:
        # Seasonal factor (Feb-May is peak season in Peru)
        month = current.month
        if 2 <= month <= 5:
            seasonal = 1.5  # Peak season
        elif 6 <= month <= 8:
            seasonal = 0.6  # Low season (winter)
        elif month in [9, 10]:
            seasonal = 1.0  # Transition
        else:
            seasonal = 0.9  # Summer

        # Day of week factor (more fishing on weekends for artisanal)
        dow = current.weekday()
        dow_factor = 1.2 if dow >= 5 else 1.0

        # Moon phase effect (simplified 29.5 day cycle)
        days_since_new = (current - datetime(2022, 1, 2)).days % 29  # Approx new moon
        if days_since_new < 3 or days_since_new > 26:
            lunar = 1.3  # New moon - good fishing
        elif 12 <= days_since_new <= 17:
            lunar = 1.2  # Full moon - also good
        else:
            lunar = 1.0

        for lat, lon, intensity, name in HOTSPOTS:
            # Number of fishing events at this hotspot today
            base_events = int(3 * intensity * seasonal * dow_factor * lunar)
            n_events = np.random.poisson(base_events)

            for _ in range(n_events):
                # Random variation around hotspot
                event_lat = lat + np.random.normal(0, 0.03)
                event_lon = lon + np.random.normal(0, 0.03)

                # Fishing hours (exponential distribution)
                hours = np.random.exponential(2.5) * intensity

                # Gear type based on location
                if abs(lon - (-71.35)) < 0.1:  # Near Punta Coles - rocky
                    gear = np.random.choice(['set_longlines', 'handlines'], p=[0.6, 0.4])
                else:
                    gear = np.random.choice(['purse_seines', 'gillnets', 'trawlers'], p=[0.4, 0.4, 0.2])

                events.append({
                    'timestamp': current.strftime("%Y-%m-%d"),
                    'lat': round(event_lat, 4),
                    'lon': round(event_lon, 4),
                    'fishing_hours': round(hours, 2),
                    'gear_type': gear,
                    'hotspot': name,
                    'source': 'simulated_imarpe_patterns'
                })

        current += timedelta(days=1)

    print(f"[OK] Generated {len(events)} fishing events")

    # Statistics
    total_hours = sum(e['fishing_hours'] for e in events)
    print(f"[OK] Total fishing hours: {total_hours:,.0f}")
    print(f"[OK] Average events/day: {len(events) / total_days:.1f}")

    # By hotspot
    print("\n[INFO] Events by hotspot:")
    for lat, lon, _, name in HOTSPOTS:
        count = sum(1 for e in events if e['hotspot'] == name)
        print(f"       {name}: {count:,} events")

    return events


def build_training_database(sst_data: xr.Dataset, fishing_events: list, output_dir: Path):
    """Build SQLite database with merged training data."""
    print(f"\n{'='*60}")
    print("BUILDING TRAINING DATABASE")
    print(f"{'='*60}")

    db_path = output_dir / "training_data.db"

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_samples (
            id INTEGER PRIMARY KEY,
            date TEXT,
            lat REAL,
            lon REAL,
            sst REAL,
            sst_anomaly REAL,
            fishing_hours REAL,
            is_fishing INTEGER,
            season TEXT,
            month INTEGER,
            hotspot_name TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    ''')

    cursor.execute('DELETE FROM training_samples')  # Clear old data

    print("[INFO] Processing SST data...")

    # Get grid
    lats = sst_data.latitude.values
    lons = sst_data.longitude.values
    times = sst_data.time.values

    print(f"[INFO] Grid: {len(lats)} x {len(lons)} = {len(lats) * len(lons)} points")
    print(f"[INFO] Times: {len(times)} days")

    # Create fishing lookup by date and approximate location
    fishing_lookup = {}
    for event in fishing_events:
        key = (event['timestamp'], round(event['lat'], 1), round(event['lon'], 1))
        if key not in fishing_lookup:
            fishing_lookup[key] = {'hours': 0, 'hotspot': event.get('hotspot', '')}
        fishing_lookup[key]['hours'] += event['fishing_hours']

    print(f"[INFO] Unique fishing locations: {len(fishing_lookup)}")

    # Process each time step
    samples = []
    total_positive = 0

    for i, t in enumerate(times):
        if i % 100 == 0:
            print(f"[INFO] Processing day {i+1}/{len(times)}...")

        t_str = str(t)[:10]
        month = int(t_str[5:7])

        # Determine season
        if 2 <= month <= 5:
            season = 'peak'
        elif 6 <= month <= 8:
            season = 'low'
        else:
            season = 'transition'

        try:
            sst_slice = sst_data.sel(time=t, method='nearest')
        except:
            continue

        for lat in lats:
            for lon in lons:
                try:
                    sst = float(sst_slice.sst.sel(
                        latitude=lat, longitude=lon, method='nearest'
                    ).values)

                    anom = float(sst_slice.anom.sel(
                        latitude=lat, longitude=lon, method='nearest'
                    ).values) if 'anom' in sst_slice else 0
                except:
                    continue

                if np.isnan(sst):
                    continue

                # Check for fishing at this location
                key = (t_str, round(float(lat), 1), round(float(lon), 1))
                fishing_info = fishing_lookup.get(key, {'hours': 0, 'hotspot': ''})
                fishing_hours = fishing_info['hours']

                is_fishing = 1 if fishing_hours > 0 else 0
                total_positive += is_fishing

                samples.append((
                    t_str,
                    float(lat),
                    float(lon),
                    sst,
                    anom,
                    fishing_hours,
                    is_fishing,
                    season,
                    month,
                    fishing_info['hotspot']
                ))

    # Insert in batches
    print(f"\n[INFO] Inserting {len(samples)} samples into database...")

    cursor.executemany('''
        INSERT INTO training_samples
        (date, lat, lon, sst, sst_anomaly, fishing_hours, is_fishing, season, month, hotspot_name)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', samples)

    # Add metadata
    cursor.execute("INSERT OR REPLACE INTO metadata VALUES ('total_samples', ?)", (str(len(samples)),))
    cursor.execute("INSERT OR REPLACE INTO metadata VALUES ('positive_samples', ?)", (str(total_positive),))
    cursor.execute("INSERT OR REPLACE INTO metadata VALUES ('created_at', ?)", (datetime.now().isoformat(),))
    cursor.execute("INSERT OR REPLACE INTO metadata VALUES ('date_range', ?)",
                   (f"{str(times[0])[:10]} to {str(times[-1])[:10]}",))

    # Create indexes
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_date ON training_samples(date)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_location ON training_samples(lat, lon)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_fishing ON training_samples(is_fishing)')

    conn.commit()
    conn.close()

    print(f"\n[OK] Database created: {db_path}")
    print(f"[OK] Total samples: {len(samples):,}")
    print(f"[OK] Positive samples (fishing): {total_positive:,} ({100*total_positive/len(samples):.1f}%)")
    print(f"[OK] Database size: {db_path.stat().st_size / 1024 / 1024:.1f} MB")

    return db_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download historical data")
    parser.add_argument('--years', type=int, default=4, help='Years of data to download')
    parser.add_argument('--output', type=str, default='data/historical', help='Output directory')

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate date range
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365 * args.years)).strftime("%Y-%m-%d")

    print("="*60)
    print("HISTORICAL DATA DOWNLOAD")
    print("="*60)
    print(f"Years: {args.years}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Output: {output_dir.absolute()}")

    # 1. Download NOAA SST
    sst_data = download_noaa_sst(start_date, end_date, output_dir)

    if sst_data is None:
        print("\n[ERROR] Could not download SST data. Exiting.")
        return 1

    # 2. Generate fishing activity
    fishing_events = generate_fishing_activity(start_date, end_date)

    # Save fishing events to JSON
    fishing_file = output_dir / "fishing_events.json"
    with open(fishing_file, 'w') as f:
        json.dump(fishing_events, f)
    print(f"[OK] Fishing events saved: {fishing_file}")

    # 3. Build training database
    db_path = build_training_database(sst_data, fishing_events, output_dir)

    print(f"\n{'='*60}")
    print("DOWNLOAD COMPLETE")
    print("="*60)
    print(f"\nFiles created:")
    print(f"  - SST data: {output_dir}/noaa_sst_*.nc")
    print(f"  - Fishing events: {fishing_file}")
    print(f"  - Training database: {db_path}")
    print(f"\nTo train the model with this data, run:")
    print(f"  python main.py --supervised")

    return 0


if __name__ == "__main__":
    sys.exit(main())
