#!/usr/bin/env python3
"""
REAL DATA ONLY Fetcher - NO SYNTHETIC DATA ALLOWED.

This module fetches ONLY real oceanographic and fishing data.
If real data is unavailable, it raises an error instead of generating synthetic data.

Real Data Sources:
- SST: NOAA OISST via multiple endpoints (ERDDAP, OPeNDAP, direct download)
- Waves: Open-Meteo Marine API (ERA5 reanalysis)
- Wind: Open-Meteo Archive API (ERA5 reanalysis)
- Fishing: Global Fishing Watch API (requires API key)
- Coastline: OpenStreetMap via Overpass API
"""

import os
import json
import numpy as np
import requests
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time

# NO SYNTHETIC DATA IMPORTS - We don't use np.random for data generation


@dataclass
class RealMarineData:
    """Container for REAL marine data only."""
    date: str
    lat: float
    lon: float
    sst: Optional[float]  # From satellite
    sst_source: str  # Must be 'satellite' or 'reanalysis'
    wave_height: Optional[float]
    wave_period: Optional[float]
    wave_direction: Optional[float]
    wind_speed: Optional[float]
    wind_direction: Optional[float]
    data_source: str  # API name


@dataclass
class RealFishingEvent:
    """Container for REAL fishing event from AIS/VMS."""
    date: str
    lat: float
    lon: float
    fishing_hours: float
    vessel_id: Optional[str]
    flag_state: Optional[str]
    gear_type: Optional[str]
    source: str  # Must be 'gfw_ais' or 'vms'


class RealDataError(Exception):
    """Raised when real data cannot be fetched."""
    pass


class RealDataFetcher:
    """
    Fetches ONLY real data. NO synthetic fallbacks.

    If data cannot be fetched, raises RealDataError.
    """

    # Peru Tacna-Ilo region
    REGION = {
        'lat_min': -18.3,
        'lat_max': -17.3,
        'lon_min': -71.5,
        'lon_max': -70.8
    }

    # Grid for sampling
    GRID_RESOLUTION = 0.1  # degrees

    def __init__(self, cache_dir: str = "data/real_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # API credentials from environment
        self.gfw_api_key = os.environ.get('GFW_API_KEY', '')
        self.copernicus_user = os.environ.get('COPERNICUS_USER', '')
        self.copernicus_pass = os.environ.get('COPERNICUS_PASS', '')

        # Track data sources used
        self.data_sources_used = []

    def fetch_real_sst(
        self,
        start_date: str,
        end_date: str,
        max_retries: int = 3
    ) -> List[Dict]:
        """
        Fetch REAL SST data from satellite sources.

        Tries multiple sources in order:
        1. NOAA CoastWatch ERDDAP
        2. NOAA NCEI OPeNDAP
        3. Copernicus Marine (if credentials available)

        Raises RealDataError if all sources fail.
        """
        print("\n[SST] Fetching REAL satellite SST data...")

        # Try NOAA CoastWatch ERDDAP first
        sst_data = self._try_noaa_coastwatch(start_date, end_date)
        if sst_data:
            return sst_data

        # Try NOAA NCEI
        sst_data = self._try_noaa_ncei(start_date, end_date)
        if sst_data:
            return sst_data

        # Try Copernicus if credentials available
        if self.copernicus_user and self.copernicus_pass:
            sst_data = self._try_copernicus_sst(start_date, end_date)
            if sst_data:
                return sst_data

        raise RealDataError(
            "Could not fetch real SST data from any source.\n"
            "Tried: NOAA CoastWatch, NOAA NCEI, Copernicus Marine.\n"
            "Please check your internet connection or try again later.\n"
            "For Copernicus, set COPERNICUS_USER and COPERNICUS_PASS environment variables."
        )

    def _try_noaa_coastwatch(self, start_date: str, end_date: str) -> Optional[List[Dict]]:
        """Try NOAA CoastWatch ERDDAP for SST."""
        print("[SST] Trying NOAA CoastWatch ERDDAP...")

        try:
            # Use the working CoastWatch endpoint
            base_url = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg_LonPM180.json"

            # Build query - get daily data for our region
            # Format: variable[(time_start):(time_end)][(z)][(lat_start):(lat_end)][(lon_start):(lon_end)]
            query = (
                f"sst[({start_date}T12:00:00Z):1:({end_date}T12:00:00Z)]"
                f"[(0.0):1:(0.0)]"
                f"[({self.REGION['lat_min']}):1:({self.REGION['lat_max']})]"
                f"[({self.REGION['lon_min']}):1:({self.REGION['lon_max']})]"
            )

            url = f"{base_url}?{query}"
            print(f"[SST] Requesting: {url[:80]}...")

            response = requests.get(url, timeout=120)

            if response.status_code == 200:
                data = response.json()
                sst_records = self._parse_erddap_json(data, 'noaa_coastwatch')
                print(f"[SST] SUCCESS: Got {len(sst_records)} real SST records from NOAA CoastWatch")
                self.data_sources_used.append('noaa_coastwatch_sst')
                return sst_records
            else:
                print(f"[SST] CoastWatch returned status {response.status_code}")
                return None

        except requests.exceptions.Timeout:
            print("[SST] CoastWatch timed out")
            return None
        except Exception as e:
            print(f"[SST] CoastWatch error: {e}")
            return None

    def _try_noaa_ncei(self, start_date: str, end_date: str) -> Optional[List[Dict]]:
        """Try NOAA NCEI for SST."""
        print("[SST] Trying NOAA NCEI...")

        try:
            # NCEI ERDDAP endpoint
            base_url = "https://www.ncei.noaa.gov/erddap/griddap/ncdc_oisst_v2_avhrr_by_time_zlev_lat_lon.json"

            query = (
                f"sst[({start_date}T12:00:00Z):1:({end_date}T12:00:00Z)]"
                f"[(0.0):1:(0.0)]"
                f"[({self.REGION['lat_min']}):1:({self.REGION['lat_max']})]"
                f"[({self.REGION['lon_min'] + 360}):1:({self.REGION['lon_max'] + 360})]"  # NCEI uses 0-360
            )

            url = f"{base_url}?{query}"
            print(f"[SST] Requesting NCEI...")

            response = requests.get(url, timeout=120)

            if response.status_code == 200:
                data = response.json()
                sst_records = self._parse_erddap_json(data, 'noaa_ncei')
                print(f"[SST] SUCCESS: Got {len(sst_records)} real SST records from NOAA NCEI")
                self.data_sources_used.append('noaa_ncei_sst')
                return sst_records
            else:
                print(f"[SST] NCEI returned status {response.status_code}")
                return None

        except Exception as e:
            print(f"[SST] NCEI error: {e}")
            return None

    def _try_copernicus_sst(self, start_date: str, end_date: str) -> Optional[List[Dict]]:
        """Try Copernicus Marine for SST."""
        print("[SST] Trying Copernicus Marine...")

        try:
            import copernicusmarine

            output_file = self.cache_dir / f"copernicus_sst_{start_date}_{end_date}.nc"

            copernicusmarine.subset(
                dataset_id="SST_GLO_SST_L4_REP_OBSERVATIONS_010_011",
                variables=["analysed_sst"],
                minimum_longitude=self.REGION['lon_min'],
                maximum_longitude=self.REGION['lon_max'],
                minimum_latitude=self.REGION['lat_min'],
                maximum_latitude=self.REGION['lat_max'],
                start_datetime=f"{start_date}T00:00:00",
                end_datetime=f"{end_date}T23:59:59",
                output_filename=str(output_file),
                username=self.copernicus_user,
                password=self.copernicus_pass
            )

            import xarray as xr
            ds = xr.open_dataset(output_file)

            records = []
            for t in ds.time.values:
                t_str = str(t)[:10]
                for lat in ds.lat.values:
                    for lon in ds.lon.values:
                        sst = float(ds.analysed_sst.sel(time=t, lat=lat, lon=lon).values)
                        if not np.isnan(sst):
                            records.append({
                                'date': t_str,
                                'lat': float(lat),
                                'lon': float(lon),
                                'sst': sst - 273.15,  # Convert K to C
                                'sst_source': 'copernicus_satellite'
                            })

            print(f"[SST] SUCCESS: Got {len(records)} real SST records from Copernicus")
            self.data_sources_used.append('copernicus_sst')
            return records

        except ImportError:
            print("[SST] copernicusmarine not installed")
            return None
        except Exception as e:
            print(f"[SST] Copernicus error: {e}")
            return None

    def _parse_erddap_json(self, data: dict, source: str) -> List[Dict]:
        """Parse ERDDAP JSON response to records."""
        records = []

        if 'table' not in data:
            return records

        table = data['table']
        col_names = table.get('columnNames', [])
        rows = table.get('rows', [])

        # Find column indices
        time_idx = col_names.index('time') if 'time' in col_names else -1
        lat_idx = col_names.index('latitude') if 'latitude' in col_names else -1
        lon_idx = col_names.index('longitude') if 'longitude' in col_names else -1
        sst_idx = col_names.index('sst') if 'sst' in col_names else -1

        for row in rows:
            if sst_idx >= 0 and row[sst_idx] is not None:
                lon = row[lon_idx]
                if lon > 180:
                    lon -= 360  # Convert 0-360 to -180-180

                records.append({
                    'date': row[time_idx][:10] if time_idx >= 0 else '',
                    'lat': row[lat_idx] if lat_idx >= 0 else 0,
                    'lon': lon,
                    'sst': row[sst_idx],
                    'sst_source': f'{source}_satellite'
                })

        return records

    def fetch_real_marine_conditions(
        self,
        start_date: str,
        end_date: str
    ) -> List[Dict]:
        """
        Fetch REAL wave and wind data from Open-Meteo ERA5.

        ERA5 is reanalysis data (real observations + model), not synthetic.
        """
        print("\n[MARINE] Fetching REAL wave/wind data from Open-Meteo ERA5...")

        records = []

        # Create grid points
        lats = np.arange(self.REGION['lat_min'], self.REGION['lat_max'] + 0.1, self.GRID_RESOLUTION)
        lons = np.arange(self.REGION['lon_min'], self.REGION['lon_max'] + 0.1, self.GRID_RESOLUTION)

        total_points = len(lats) * len(lons)
        print(f"[MARINE] Fetching {total_points} grid points...")

        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                # Marine data (waves)
                marine_url = "https://marine-api.open-meteo.com/v1/marine"
                marine_params = {
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": start_date,
                    "end_date": end_date,
                    "daily": ["wave_height_max", "wave_direction_dominant", "wave_period_max"],
                    "timezone": "America/Lima"
                }

                # Archive data (wind)
                archive_url = "https://archive-api.open-meteo.com/v1/archive"
                archive_params = {
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": start_date,
                    "end_date": end_date,
                    "daily": ["wind_speed_10m_max", "wind_direction_10m_dominant"],
                    "timezone": "America/Lima"
                }

                try:
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
                                    'data_source': 'open_meteo_era5'
                                }

                                if archive_data and 'daily' in archive_data:
                                    arch = archive_data['daily']
                                    record['wind_speed'] = arch.get('wind_speed_10m_max', [None])[t_idx]
                                    record['wind_direction'] = arch.get('wind_direction_10m_dominant', [None])[t_idx]

                                records.append(record)

                except Exception as e:
                    print(f"[MARINE] Error at ({lat}, {lon}): {e}")

                time.sleep(0.3)  # Rate limiting

        if not records:
            raise RealDataError("Could not fetch marine data from Open-Meteo")

        print(f"[MARINE] SUCCESS: Got {len(records)} real marine records from ERA5")
        self.data_sources_used.append('open_meteo_era5')
        return records

    def fetch_real_fishing_activity(
        self,
        start_date: str,
        end_date: str
    ) -> List[RealFishingEvent]:
        """
        Fetch REAL fishing activity from Global Fishing Watch.

        Requires GFW_API_KEY environment variable.
        """
        print("\n[FISHING] Fetching REAL fishing activity from Global Fishing Watch...")

        if not self.gfw_api_key:
            raise RealDataError(
                "GFW_API_KEY not set. Cannot fetch real fishing data.\n"
                "Get a free API key at: https://globalfishingwatch.org/our-apis/\n"
                "Then set: export GFW_API_KEY='your_key_here'"
            )

        headers = {"Authorization": f"Bearer {self.gfw_api_key}"}

        # Define region polygon
        polygon = {
            "type": "Polygon",
            "coordinates": [[
                [self.REGION['lon_min'], self.REGION['lat_min']],
                [self.REGION['lon_max'], self.REGION['lat_min']],
                [self.REGION['lon_max'], self.REGION['lat_max']],
                [self.REGION['lon_min'], self.REGION['lat_max']],
                [self.REGION['lon_min'], self.REGION['lat_min']]
            ]]
        }

        try:
            # GFW 4Wings API for fishing effort
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

                events = []
                for entry in entries:
                    events.append(RealFishingEvent(
                        date=entry.get('date', ''),
                        lat=entry.get('lat', 0),
                        lon=entry.get('lon', 0),
                        fishing_hours=entry.get('hours', 0),
                        vessel_id=entry.get('vesselId'),
                        flag_state=entry.get('flag'),
                        gear_type=entry.get('geartype'),
                        source='gfw_ais'
                    ))

                print(f"[FISHING] SUCCESS: Got {len(events)} real fishing events from GFW")
                self.data_sources_used.append('gfw_ais')
                return events

            elif response.status_code == 401:
                raise RealDataError("GFW API key is invalid. Please check your API key.")
            elif response.status_code == 403:
                raise RealDataError("GFW API access denied. Your API key may not have access to this data.")
            else:
                raise RealDataError(f"GFW API error: {response.status_code} - {response.text[:200]}")

        except requests.exceptions.Timeout:
            raise RealDataError("GFW API timed out. Please try again later.")
        except RealDataError:
            raise
        except Exception as e:
            raise RealDataError(f"GFW API error: {e}")

    def build_real_training_dataset(
        self,
        start_date: str,
        end_date: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build training dataset using ONLY real data.

        NO synthetic data generation. Raises error if real data unavailable.
        """
        print("=" * 60)
        print("BUILDING TRAINING DATASET - REAL DATA ONLY")
        print("=" * 60)
        print(f"Period: {start_date} to {end_date}")
        print("NO synthetic data will be used.\n")

        # 1. Fetch real SST
        sst_data = self.fetch_real_sst(start_date, end_date)

        # 2. Fetch real marine conditions
        marine_data = self.fetch_real_marine_conditions(start_date, end_date)

        # 3. Fetch real fishing activity
        fishing_events = self.fetch_real_fishing_activity(start_date, end_date)

        # 4. Merge data
        print("\n[MERGE] Merging real data sources...")

        # Create SST lookup
        sst_lookup = {}
        for record in sst_data:
            key = (record['date'], round(record['lat'], 1), round(record['lon'], 1))
            sst_lookup[key] = record['sst']

        # Create fishing lookup
        fishing_lookup = {}
        for event in fishing_events:
            key = (event.date, round(event.lat, 1), round(event.lon, 1))
            if key not in fishing_lookup:
                fishing_lookup[key] = 0
            fishing_lookup[key] += event.fishing_hours

        # Build feature matrix
        X_list = []
        y_list = []

        for record in marine_data:
            key = (record['date'], round(record['lat'], 1), round(record['lon'], 1))

            # Get SST for this location/time
            sst = sst_lookup.get(key)
            if sst is None:
                continue  # Skip if no SST data

            # Get fishing hours
            fishing_hours = fishing_lookup.get(key, 0)

            # Feature vector (all REAL data)
            features = [
                sst,
                record.get('wave_height') or 0,
                record.get('wave_period') or 0,
                record.get('wave_direction') or 0,
                record.get('wind_speed') or 0,
                record.get('wind_direction') or 0,
                record['lat'],
                record['lon'],
                int(record['date'][5:7])  # Month
            ]

            X_list.append(features)
            y_list.append(fishing_hours)

        X = np.array(X_list)
        y = np.array(y_list)

        print(f"\n[OK] Training dataset built with REAL data:")
        print(f"     Samples: {len(X)}")
        print(f"     Features: {X.shape[1] if len(X) > 0 else 0}")
        print(f"     Positive (fishing): {(y > 0).sum()}")
        print(f"     Data sources: {', '.join(self.data_sources_used)}")

        return X, y

    def save_to_database(
        self,
        sst_data: List[Dict],
        marine_data: List[Dict],
        fishing_events: List[RealFishingEvent]
    ) -> Path:
        """Save real data to SQLite database."""
        db_path = self.cache_dir / "real_data_only.db"

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sst_data (
                id INTEGER PRIMARY KEY,
                date TEXT,
                lat REAL,
                lon REAL,
                sst REAL,
                source TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS marine_data (
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
            CREATE TABLE IF NOT EXISTS fishing_events (
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
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')

        # Insert data
        for record in sst_data:
            cursor.execute(
                'INSERT INTO sst_data (date, lat, lon, sst, source) VALUES (?, ?, ?, ?, ?)',
                (record['date'], record['lat'], record['lon'], record['sst'], record['sst_source'])
            )

        for record in marine_data:
            cursor.execute(
                '''INSERT INTO marine_data
                   (date, lat, lon, wave_height, wave_period, wave_direction, wind_speed, wind_direction, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (record['date'], record['lat'], record['lon'],
                 record.get('wave_height'), record.get('wave_period'), record.get('wave_direction'),
                 record.get('wind_speed'), record.get('wind_direction'), record['data_source'])
            )

        for event in fishing_events:
            cursor.execute(
                '''INSERT INTO fishing_events
                   (date, lat, lon, fishing_hours, vessel_id, flag_state, gear_type, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                (event.date, event.lat, event.lon, event.fishing_hours,
                 event.vessel_id, event.flag_state, event.gear_type, event.source)
            )

        # Metadata
        cursor.execute("INSERT OR REPLACE INTO metadata VALUES ('data_type', 'REAL_ONLY')")
        cursor.execute("INSERT OR REPLACE INTO metadata VALUES ('created_at', ?)", (datetime.now().isoformat(),))
        cursor.execute("INSERT OR REPLACE INTO metadata VALUES ('sources', ?)", (','.join(self.data_sources_used),))

        conn.commit()
        conn.close()

        print(f"[OK] Real data saved to: {db_path}")
        return db_path


def main():
    """Download real data only - no synthetic fallbacks."""
    import argparse

    parser = argparse.ArgumentParser(description="Download REAL data only (no synthetic)")
    parser.add_argument('--start', default='2024-01-01', help='Start date')
    parser.add_argument('--end', default='2024-12-31', help='End date')

    args = parser.parse_args()

    fetcher = RealDataFetcher()

    try:
        X, y = fetcher.build_real_training_dataset(args.start, args.end)
        print(f"\nSUCCESS: Built real training dataset with {len(X)} samples")
    except RealDataError as e:
        print(f"\nERROR: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
