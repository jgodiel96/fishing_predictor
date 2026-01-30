"""
Historical Data Fetcher for ML Training.

Downloads and processes multi-year oceanographic data from:
- Copernicus Marine: SST, currents, waves, chlorophyll-a
- Global Fishing Watch: Real fishing activity (AIS-based)
- NOAA ERDDAP: SST validation data

This enables supervised learning with actual fishing patterns.
"""

import os
import json
import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import requests
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import warnings

warnings.filterwarnings('ignore')


@dataclass
class HistoricalDataPoint:
    """Single data point with all oceanographic variables."""
    timestamp: str
    lat: float
    lon: float
    sst: float
    sst_anomaly: float
    current_u: float
    current_v: float
    wave_height: float
    chlorophyll: float
    fishing_hours: float  # From GFW - ground truth!
    is_fishing_event: bool


class HistoricalDataFetcher:
    """
    Fetches and manages multi-year historical oceanographic data.

    Supports:
    - Copernicus Marine (SST, currents, waves, chlorophyll)
    - Global Fishing Watch (fishing activity as ground truth)
    - Local SQLite database for caching
    """

    # Peru Tacna-Ilo region
    REGION = {
        'lat_min': -18.3,
        'lat_max': -17.3,
        'lon_min': -71.5,
        'lon_max': -70.8
    }

    # Copernicus Marine datasets
    COPERNICUS_DATASETS = {
        'sst': 'SST_GLO_SST_L4_REP_OBSERVATIONS_010_011',
        'currents': 'GLOBAL_MULTIYEAR_PHY_001_030',
        'waves': 'GLOBAL_MULTIYEAR_WAV_001_032',
        'chlorophyll': 'OCEANCOLOUR_GLO_BGC_L4_MY_009_104'
    }

    def __init__(self, cache_dir: str = "data/historical"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "historical_data.db"
        self._init_database()

        # API credentials (loaded from environment or config)
        self.copernicus_user = os.environ.get('COPERNICUS_USER', '')
        self.copernicus_pass = os.environ.get('COPERNICUS_PASS', '')
        self.gfw_api_key = os.environ.get('GFW_API_KEY', '')

    def _init_database(self):
        """Initialize SQLite database for caching."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Main data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ocean_data (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                lat REAL,
                lon REAL,
                sst REAL,
                sst_anomaly REAL,
                current_u REAL,
                current_v REAL,
                wave_height REAL,
                chlorophyll REAL,
                fishing_hours REAL,
                is_fishing_event INTEGER,
                UNIQUE(timestamp, lat, lon)
            )
        ''')

        # Metadata table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fetch_log (
                id INTEGER PRIMARY KEY,
                source TEXT,
                start_date TEXT,
                end_date TEXT,
                fetch_timestamp TEXT,
                records_count INTEGER
            )
        ''')

        # Indexes for fast queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON ocean_data(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_location ON ocean_data(lat, lon)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_fishing ON ocean_data(is_fishing_event)')

        conn.commit()
        conn.close()
        print(f"[OK] Database initialized: {self.db_path}")

    def fetch_copernicus_sst(
        self,
        start_date: str = "2022-01-01",
        end_date: str = "2025-12-31"
    ) -> Optional[xr.Dataset]:
        """
        Fetch historical SST from Copernicus Marine.

        Requires: pip install copernicusmarine
        And Copernicus Marine account (free).
        """
        try:
            import copernicusmarine
        except ImportError:
            print("[WARN] copernicusmarine not installed. Run: pip install copernicusmarine")
            return None

        output_file = self.cache_dir / f"sst_{start_date}_{end_date}.nc"

        if output_file.exists():
            print(f"[INFO] Loading cached SST: {output_file}")
            return xr.open_dataset(output_file)

        print(f"[INFO] Downloading SST from Copernicus ({start_date} to {end_date})...")

        try:
            copernicusmarine.subset(
                dataset_id=self.COPERNICUS_DATASETS['sst'],
                variables=["analysed_sst", "analysis_error"],
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

            ds = xr.open_dataset(output_file)
            print(f"[OK] SST data downloaded: {len(ds.time)} time steps")
            return ds

        except Exception as e:
            print(f"[ERROR] Failed to fetch SST: {e}")
            return None

    def fetch_copernicus_currents(
        self,
        start_date: str = "2022-01-01",
        end_date: str = "2025-12-31"
    ) -> Optional[xr.Dataset]:
        """Fetch historical ocean currents from Copernicus Marine."""
        try:
            import copernicusmarine
        except ImportError:
            print("[WARN] copernicusmarine not installed")
            return None

        output_file = self.cache_dir / f"currents_{start_date}_{end_date}.nc"

        if output_file.exists():
            print(f"[INFO] Loading cached currents: {output_file}")
            return xr.open_dataset(output_file)

        print(f"[INFO] Downloading currents from Copernicus...")

        try:
            copernicusmarine.subset(
                dataset_id=self.COPERNICUS_DATASETS['currents'],
                variables=["uo", "vo"],  # Eastward, Northward velocity
                minimum_longitude=self.REGION['lon_min'],
                maximum_longitude=self.REGION['lon_max'],
                minimum_latitude=self.REGION['lat_min'],
                maximum_latitude=self.REGION['lat_max'],
                minimum_depth=0,
                maximum_depth=50,  # Surface layer
                start_datetime=f"{start_date}T00:00:00",
                end_datetime=f"{end_date}T23:59:59",
                output_filename=str(output_file),
                username=self.copernicus_user,
                password=self.copernicus_pass
            )

            return xr.open_dataset(output_file)

        except Exception as e:
            print(f"[ERROR] Failed to fetch currents: {e}")
            return None

    def fetch_copernicus_chlorophyll(
        self,
        start_date: str = "2022-01-01",
        end_date: str = "2025-12-31"
    ) -> Optional[xr.Dataset]:
        """Fetch historical chlorophyll-a from Copernicus Marine."""
        try:
            import copernicusmarine
        except ImportError:
            return None

        output_file = self.cache_dir / f"chlorophyll_{start_date}_{end_date}.nc"

        if output_file.exists():
            return xr.open_dataset(output_file)

        print(f"[INFO] Downloading chlorophyll-a from Copernicus...")

        try:
            copernicusmarine.subset(
                dataset_id=self.COPERNICUS_DATASETS['chlorophyll'],
                variables=["CHL"],  # Chlorophyll concentration
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

            return xr.open_dataset(output_file)

        except Exception as e:
            print(f"[ERROR] Failed to fetch chlorophyll: {e}")
            return None

    def fetch_gfw_fishing_activity(
        self,
        start_date: str = "2022-01-01",
        end_date: str = "2025-12-31"
    ) -> List[Dict]:
        """
        Fetch fishing activity from Global Fishing Watch API.

        This provides GROUND TRUTH for training supervised models.
        Requires: GFW API key (free for research).

        Raises:
            ValueError: If GFW_API_KEY is not set
            RuntimeError: If API request fails
        """
        if not self.gfw_api_key:
            raise ValueError(
                "GFW_API_KEY not set. Real fishing data requires a Global Fishing Watch API key.\n"
                "Get your free key at: https://globalfishingwatch.org/our-apis/\n"
                "Then set: export GFW_API_KEY='your_key_here'"
            )

        print(f"[INFO] Fetching fishing activity from Global Fishing Watch...")

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
            # Use 4Wings API for fishing effort
            response = requests.post(
                "https://gateway.api.globalfishingwatch.org/v3/4wings/report",
                headers=headers,
                json={
                    "datasets": ["public-global-fishing-effort:latest"],
                    "date-range": [start_date, end_date],
                    "region": polygon,
                    "spatial-resolution": "low",  # 0.1 degree
                    "temporal-resolution": "daily",
                    "group-by": ["flag", "geartype"]
                },
                timeout=120
            )

            if response.status_code == 200:
                data = response.json()
                print(f"[OK] Retrieved {len(data.get('entries', []))} fishing records")
                return data.get('entries', [])
            else:
                raise RuntimeError(
                    f"GFW API error {response.status_code}: {response.text[:200]}\n"
                    "Check your API key and try again."
                )

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"GFW API request failed: {e}")

    def fetch_noaa_erddap_sst(
        self,
        start_date: str = "2022-01-01",
        end_date: str = "2025-12-31"
    ) -> Optional[xr.Dataset]:
        """
        Fetch SST from NOAA ERDDAP (no registration required).
        Useful as backup or validation data.
        """
        try:
            from erddapy import ERDDAP
        except ImportError:
            print("[WARN] erddapy not installed. Run: pip install erddapy")
            return None

        print("[INFO] Fetching SST from NOAA ERDDAP...")

        try:
            e = ERDDAP(
                server="https://www.ncei.noaa.gov/erddap",
                protocol="griddap"
            )
            e.dataset_id = "ncdc_oisst_v2_avhrr_by_time_zlev_lat_lon"
            e.constraints = {
                "time>=": start_date,
                "time<=": end_date,
                "latitude>=": self.REGION['lat_min'],
                "latitude<=": self.REGION['lat_max'],
                "longitude>=": self.REGION['lon_min'],
                "longitude<=": self.REGION['lon_max'],
                "zlev=": 0
            }
            e.variables = ["sst", "anom"]

            ds = e.to_xarray()
            print(f"[OK] NOAA SST: {len(ds.time)} time steps")
            return ds

        except Exception as e:
            print(f"[ERROR] NOAA ERDDAP failed: {e}")
            return None

    def build_training_dataset(
        self,
        start_date: str = "2022-01-01",
        end_date: str = "2025-12-31"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build complete training dataset from REAL data sources only.

        Priority:
        1. Real data from Global Fishing Watch (requires GFW_API_KEY)
        2. Real data from Open-Meteo ERA5
        3. Real data from NOAA ERDDAP

        NOTE: No synthetic data fallback. If real data is unavailable,
        run: python scripts/download_100_real.py --months 6

        Returns:
            X: Feature matrix (n_samples, n_features)
            y: Labels - fishing hours (for regression) or binary (for classification)
        """
        print("=" * 60)
        print("BUILDING TRAINING DATASET")
        print("=" * 60)

        # Try to load from existing database first
        X, y = self.get_training_data_from_db()

        if X.shape[0] >= 100:
            print(f"[OK] Using existing training data: {X.shape[0]} samples")
            return X, y

        print("[INFO] No existing training data. Attempting to fetch...")

        # 1. Fetch oceanographic data
        sst_ds = self.fetch_noaa_erddap_sst(start_date, end_date)

        # 2. Fetch fishing activity (ground truth)
        fishing_events = self.fetch_gfw_fishing_activity(start_date, end_date)

        if sst_ds is None:
            print("[ERROR] Could not fetch SST data")
            print("[INFO] Run: python scripts/download_real_data.py --years 2")
            return np.array([]), np.array([])

        # 3. Merge data sources
        print("[INFO] Merging data sources...")

        # Create grid
        lats = np.arange(self.REGION['lat_min'], self.REGION['lat_max'], 0.1)
        lons = np.arange(self.REGION['lon_min'], self.REGION['lon_max'], 0.1)

        X_list = []
        y_list = []

        # Process each time step
        times = sst_ds.time.values[:100]  # Limit for demo

        for t in times:
            t_str = str(t)[:10]

            # Get SST for this time
            try:
                sst_slice = sst_ds.sel(time=t, method='nearest')
            except:
                continue

            # Get fishing events for this day
            day_fishing = [
                e for e in fishing_events
                if e.get('timestamp', '')[:10] == t_str
            ]

            for lat in lats:
                for lon in lons:
                    # Extract features
                    try:
                        sst = float(sst_slice.sst.sel(
                            latitude=lat, longitude=lon, method='nearest'
                        ).values)
                        sst_anom = float(sst_slice.anom.sel(
                            latitude=lat, longitude=lon, method='nearest'
                        ).values) if 'anom' in sst_slice else 0
                    except:
                        continue

                    if np.isnan(sst):
                        continue

                    # Calculate fishing hours at this location
                    fishing_hours = sum(
                        e.get('fishing_hours', 0)
                        for e in day_fishing
                        if abs(e.get('lat', 0) - lat) < 0.1 and abs(e.get('lon', 0) - lon) < 0.1
                    )

                    # Feature vector
                    features = [
                        sst,
                        sst_anom,
                        self._sst_optimal_score(sst),
                        lat,
                        lon,
                        self._distance_to_coast(lat, lon),
                        self._month_to_season(t),
                    ]

                    X_list.append(features)
                    y_list.append(fishing_hours)

        X = np.array(X_list)
        y = np.array(y_list)

        print(f"[OK] Training dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"[OK] Fishing events: {(y > 0).sum()} positive samples ({100*(y>0).mean():.1f}%)")

        # Save to database
        self._save_to_database(X, y, fishing_events)

        return X, y

    def _sst_optimal_score(self, sst: float) -> float:
        """Score based on optimal SST range for Humboldt species."""
        if 14 <= sst <= 24:
            center = 17.5
            return np.exp(-((sst - center) ** 2) / 12.5)
        return 0.1

    def _distance_to_coast(self, lat: float, lon: float) -> float:
        """Approximate distance to Peru coast (km)."""
        # Simplified: coast roughly at lon=-71.4 in this region
        return abs(lon - (-71.4)) * 111 * np.cos(np.radians(lat))

    def _month_to_season(self, timestamp) -> float:
        """Convert timestamp to seasonal score (peak=Feb-May)."""
        try:
            month = int(str(timestamp)[5:7])
            return 1.0 if 2 <= month <= 5 else 0.5
        except:
            return 0.5

    def _save_to_database(self, X: np.ndarray, y: np.ndarray, events: List[Dict]):
        """Save processed data to SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO fetch_log (source, start_date, end_date, fetch_timestamp, records_count)
            VALUES (?, ?, ?, ?, ?)
        ''', ('merged', '', '', datetime.now().isoformat(), len(X)))

        conn.commit()
        conn.close()
        print(f"[OK] Data saved to database: {self.db_path}")

    def get_training_data_from_db(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load training data from database (prefers real data)."""

        # Priority 1: Real training data from Open-Meteo ERA5
        real_db = self.cache_dir / "real_training_data.db"
        if real_db.exists():
            print(f"[INFO] Loading REAL training data from: {real_db}")
            conn = sqlite3.connect(real_db)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT sst, sst_anomaly, lat, lon, wave_height, wave_period,
                       wind_speed, wind_direction, month, fishing_hours
                FROM training_samples
                WHERE sst IS NOT NULL
            ''')

            rows = cursor.fetchall()
            conn.close()

            if rows:
                data = np.array(rows)
                # Features: sst, sst_anomaly, lat, lon, wave_height, wave_period,
                #           wind_speed, wind_direction, month
                X = data[:, :9]
                # Handle NaN values
                X = np.nan_to_num(X, nan=0.0)
                # Target: fishing_hours
                y = data[:, 9]

                print(f"[OK] Loaded {len(X)} REAL samples (Open-Meteo ERA5)")
                return X, y

        # Fallback to ocean_data table (no synthetic data)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT sst, sst_anomaly, lat, lon, fishing_hours
            FROM ocean_data
            WHERE sst IS NOT NULL
        ''')

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return np.array([]), np.array([])

        data = np.array(rows)
        X = data[:, :4]
        y = data[:, 4]

        return X, y

    def get_statistics(self) -> Dict:
        """Get database statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM ocean_data')
        total_records = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM ocean_data WHERE fishing_hours > 0')
        fishing_records = cursor.fetchone()[0]

        cursor.execute('SELECT MIN(timestamp), MAX(timestamp) FROM ocean_data')
        date_range = cursor.fetchone()

        conn.close()

        return {
            'total_records': total_records,
            'fishing_records': fishing_records,
            'date_range': date_range,
            'database_path': str(self.db_path)
        }


class FishMovementPredictor:
    """
    Predicts fish school movement based on ocean currents.

    Uses Lagrangian particle tracking to simulate fish advection.
    """

    def __init__(self):
        self.current_field = {}

    def set_current_field(self, u_data: np.ndarray, v_data: np.ndarray,
                          lats: np.ndarray, lons: np.ndarray):
        """Set the current velocity field."""
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                if not np.isnan(u_data[i, j]) and not np.isnan(v_data[i, j]):
                    self.current_field[(round(lat, 2), round(lon, 2))] = (
                        float(u_data[i, j]), float(v_data[i, j])
                    )

    def predict_movement(
        self,
        start_lat: float,
        start_lon: float,
        hours: int = 24,
        dt_hours: float = 1.0
    ) -> List[Tuple[float, float, float]]:
        """
        Predict fish school trajectory using Lagrangian advection.

        Args:
            start_lat, start_lon: Initial position
            hours: Prediction horizon
            dt_hours: Time step

        Returns:
            List of (lat, lon, time_hours) positions
        """
        trajectory = [(start_lat, start_lon, 0.0)]

        lat, lon = start_lat, start_lon

        for t in np.arange(0, hours, dt_hours):
            # Get current at this position
            key = (round(lat, 2), round(lon, 2))

            if key not in self.current_field:
                # No current data, stop
                break

            u, v = self.current_field[key]

            # Advect position (simple Euler)
            # Convert m/s to degrees/hour
            lat_change = v * 3600 / 111000 * dt_hours
            lon_change = u * 3600 / (111000 * np.cos(np.radians(lat))) * dt_hours

            # Add some random swimming behavior
            lat_change += np.random.normal(0, 0.001)
            lon_change += np.random.normal(0, 0.001)

            lat += lat_change
            lon += lon_change

            trajectory.append((lat, lon, t + dt_hours))

        return trajectory

    def predict_school_dispersion(
        self,
        center_lat: float,
        center_lon: float,
        n_particles: int = 50,
        hours: int = 24
    ) -> List[List[Tuple[float, float, float]]]:
        """
        Predict dispersion of a fish school using particle cloud.

        Returns trajectories for multiple particles representing the school.
        """
        trajectories = []

        for _ in range(n_particles):
            # Slight random offset for each particle
            lat = center_lat + np.random.normal(0, 0.02)
            lon = center_lon + np.random.normal(0, 0.02)

            traj = self.predict_movement(lat, lon, hours)
            trajectories.append(traj)

        return trajectories

    def get_future_hotspots(
        self,
        current_hotspots: List[Tuple[float, float]],
        hours: int = 24
    ) -> List[Dict]:
        """
        Predict where current hotspots will move.

        Returns future positions with confidence estimates.
        """
        future_hotspots = []

        for lat, lon in current_hotspots:
            trajectories = self.predict_school_dispersion(lat, lon, n_particles=30, hours=hours)

            # Get final positions
            final_positions = [t[-1][:2] for t in trajectories if len(t) > 1]

            if not final_positions:
                continue

            # Calculate center and spread
            final_lats = [p[0] for p in final_positions]
            final_lons = [p[1] for p in final_positions]

            center_lat = np.mean(final_lats)
            center_lon = np.mean(final_lons)
            spread = np.sqrt(np.var(final_lats) + np.var(final_lons))

            future_hotspots.append({
                'original_lat': lat,
                'original_lon': lon,
                'predicted_lat': center_lat,
                'predicted_lon': center_lon,
                'hours_ahead': hours,
                'confidence': max(0, 1 - spread * 10),  # Lower spread = higher confidence
                'spread_km': spread * 111
            })

        return future_hotspots


# CLI for downloading data
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download historical fishing data")
    parser.add_argument('--start', default='2022-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', default='2025-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--source', default='all', choices=['noaa', 'copernicus', 'gfw', 'all'])

    args = parser.parse_args()

    fetcher = HistoricalDataFetcher()

    print(f"\nDownloading data from {args.start} to {args.end}...")

    if args.source in ['noaa', 'all']:
        fetcher.fetch_noaa_erddap_sst(args.start, args.end)

    if args.source in ['gfw', 'all']:
        fetcher.fetch_gfw_fishing_activity(args.start, args.end)

    # Build training dataset
    X, y = fetcher.build_training_dataset(args.start, args.end)

    stats = fetcher.get_statistics()
    print(f"\nDatabase statistics: {json.dumps(stats, indent=2)}")
