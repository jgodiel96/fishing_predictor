#!/usr/bin/env python3
"""
Incremental Data Downloader for Bronze Layer.

Downloads data by month and saves to the Bronze (raw) layer.
Only downloads months that don't already exist.

Features:
- Respects existing data (never overwrites)
- Updates manifests automatically
- Supports all data sources (GFW, Open-Meteo, SST)
- Progress tracking with resume capability

Usage:
    # Download all sources for date range
    python scripts/download_incremental.py --start 2020-01 --end 2026-01

    # Download specific source only
    python scripts/download_incremental.py --source gfw --start 2024-01 --end 2026-01

    # Check what would be downloaded
    python scripts/download_incremental.py --dry-run

Environment Variables:
    GFW_API_KEY       - Global Fishing Watch API key (required for fishing data)
    EARTHDATA_USER    - NASA Earthdata username (for MUR SST)
    EARTHDATA_PASS    - NASA Earthdata password
    COPERNICUS_USER   - Copernicus Marine username
    COPERNICUS_PASS   - Copernicus Marine password
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from calendar import monthrange
from typing import List, Dict, Optional, Tuple

import requests
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

from data.data_config import DataConfig
from data.manifest import ManifestManager


class ProgressBar:
    """Simple ASCII progress bar."""

    def __init__(self, total: int, prefix: str = "", width: int = 30):
        self.total = max(1, total)
        self.prefix = prefix
        self.width = width
        self.current = 0
        self.start_time = time.time()

    def update(self, n: int = 1):
        self.current = min(self.current + n, self.total)
        self._render()

    def _render(self):
        percent = self.current / self.total
        filled = int(self.width * percent)
        bar = "█" * filled + "░" * (self.width - filled)
        elapsed = time.time() - self.start_time
        eta = f"ETA {(elapsed / max(percent, 0.01) * (1 - percent)):.0f}s" if percent < 1 else f"Done {elapsed:.1f}s"
        sys.stdout.write(f"\r{self.prefix} {bar} {percent:>3.0%} | {self.current}/{self.total} | {eta}")
        sys.stdout.flush()
        if self.current >= self.total:
            print()


def get_month_range(year: int, month: int) -> Tuple[str, str]:
    """Get first and last day of a month as ISO strings."""
    first_day = f"{year}-{month:02d}-01"
    last_day = f"{year}-{month:02d}-{monthrange(year, month)[1]:02d}"
    return first_day, last_day


def parse_month(month_str: str) -> Tuple[int, int]:
    """Parse YYYY-MM string to (year, month) tuple."""
    parts = month_str.split('-')
    return int(parts[0]), int(parts[1])


class IncrementalDownloader:
    """Downloads data incrementally by month to Bronze layer."""

    def __init__(self, verbose: bool = False, dry_run: bool = False):
        self.verbose = verbose
        self.dry_run = dry_run
        self.region = DataConfig.REGION

        # API credentials
        self.gfw_api_key = os.environ.get('GFW_API_KEY', '')
        self.earthdata_user = os.environ.get('EARTHDATA_USER', '')
        self.earthdata_pass = os.environ.get('EARTHDATA_PASS', '')
        self.copernicus_user = os.environ.get('COPERNICUS_USER', '')
        self.copernicus_pass = os.environ.get('COPERNICUS_PASS', '')

    def log(self, msg: str):
        if self.verbose:
            print(f"  {msg}")

    # =========================================================================
    # OPEN-METEO (Marine conditions)
    # =========================================================================

    def download_open_meteo(
        self,
        year: int,
        month: int,
        manager: ManifestManager
    ) -> bool:
        """
        Download Open-Meteo ERA5 marine data for a month.

        Returns True if successful.
        """
        filename = f"{year}-{month:02d}.parquet"
        output_path = DataConfig.RAW_OPEN_METEO / filename

        if output_path.exists():
            self.log(f"Skipping {filename} (already exists)")
            return True

        start_date, end_date = get_month_range(year, month)
        self.log(f"Downloading Open-Meteo: {start_date} to {end_date}")

        if self.dry_run:
            return True

        try:
            # Build grid of points
            lats = np.arange(
                self.region['lat_min'],
                self.region['lat_max'] + 0.05,
                DataConfig.GRID_RESOLUTION
            )
            lons = np.arange(
                self.region['lon_min'],
                self.region['lon_max'] + 0.05,
                DataConfig.GRID_RESOLUTION
            )

            all_data = []

            # Query API for each grid point (batch by latitude)
            for lat in lats:
                url = "https://marine-api.open-meteo.com/v1/marine"
                params = {
                    'latitude': [lat] * len(lons),
                    'longitude': lons.tolist(),
                    'start_date': start_date,
                    'end_date': end_date,
                    'daily': 'wave_height_max,wave_period_max,wave_direction_dominant,wind_wave_height_max',
                    'timezone': 'UTC'
                }

                response = requests.get(url, params=params, timeout=60)
                if response.status_code != 200:
                    self.log(f"API error: {response.status_code}")
                    continue

                data = response.json()

                # Handle single vs multiple locations
                if isinstance(data, list):
                    for i, loc_data in enumerate(data):
                        self._parse_open_meteo_response(loc_data, all_data)
                else:
                    self._parse_open_meteo_response(data, all_data)

                time.sleep(0.1)  # Rate limiting

            if not all_data:
                self.log(f"No data received for {filename}")
                return False

            # Save as parquet
            df = pd.DataFrame(all_data)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(output_path, index=False)

            # Update manifest
            manager.add_download(
                filename=filename,
                period_start=start_date,
                period_end=end_date,
                records=len(df),
                source_url='https://marine-api.open-meteo.com/v1/marine',
                api_response_code=200
            )

            self.log(f"Saved {filename}: {len(df)} records")
            return True

        except Exception as e:
            self.log(f"Error downloading {filename}: {e}")
            return False

    def _parse_open_meteo_response(self, data: Dict, all_data: List):
        """Parse Open-Meteo response and append to all_data."""
        if 'daily' not in data:
            return

        lat = data.get('latitude')
        lon = data.get('longitude')
        daily = data['daily']

        dates = daily.get('time', [])
        wave_heights = daily.get('wave_height_max', [])
        wave_periods = daily.get('wave_period_max', [])
        wave_dirs = daily.get('wave_direction_dominant', [])
        wind_wave_heights = daily.get('wind_wave_height_max', [])

        for i, date in enumerate(dates):
            all_data.append({
                'date': date,
                'lat': lat,
                'lon': lon,
                'wave_height': wave_heights[i] if i < len(wave_heights) else None,
                'wave_period': wave_periods[i] if i < len(wave_periods) else None,
                'wave_direction': wave_dirs[i] if i < len(wave_dirs) else None,
                'wind_speed': None,  # Not in marine API
                'wind_direction': None,
                'source': 'open_meteo_era5'
            })

    # =========================================================================
    # GLOBAL FISHING WATCH
    # =========================================================================

    def download_gfw(
        self,
        year: int,
        month: int,
        manager: ManifestManager
    ) -> bool:
        """
        Download Global Fishing Watch data for a month.

        Requires GFW_API_KEY environment variable.
        Uses the correct API format with query parameters.
        """
        if not self.gfw_api_key:
            self.log("GFW_API_KEY not set, skipping fishing data")
            return False

        filename = f"{year}-{month:02d}.parquet"
        output_path = DataConfig.RAW_GFW / filename

        if output_path.exists():
            self.log(f"Skipping {filename} (already exists)")
            return True

        start_date, end_date = get_month_range(year, month)
        self.log(f"Downloading GFW: {start_date} to {end_date}")

        if self.dry_run:
            return True

        try:
            url = "https://gateway.api.globalfishingwatch.org/v3/4wings/report"

            headers = {
                'Authorization': f'Bearer {self.gfw_api_key}',
                'Content-Type': 'application/json'
            }

            # Query parameters (required by GFW API)
            params = {
                'spatial-resolution': 'LOW',
                'temporal-resolution': 'MONTHLY',
                'group-by': 'FLAGANDGEARTYPE',
                'datasets[0]': 'public-global-fishing-effort:latest',
                'date-range': f'{start_date},{end_date}',
                'format': 'JSON'
            }

            # Region goes in the body as geojson
            geojson_region = {
                'type': 'Polygon',
                'coordinates': [[
                    [self.region['lon_min'], self.region['lat_min']],
                    [self.region['lon_max'], self.region['lat_min']],
                    [self.region['lon_max'], self.region['lat_max']],
                    [self.region['lon_min'], self.region['lat_max']],
                    [self.region['lon_min'], self.region['lat_min']]
                ]]
            }
            body = {'geojson': geojson_region}

            response = requests.post(url, params=params, json=body, headers=headers, timeout=120)

            if response.status_code != 200:
                self.log(f"GFW API error: {response.status_code} - {response.text[:200]}")
                return False

            data = response.json()

            # Parse response - GFW format has nested structure
            all_data = []
            entries = data.get('entries', [])

            for entry in entries:
                # Each entry is a dict with dataset name as key
                for dataset_key, records in entry.items():
                    if isinstance(records, list):
                        for record in records:
                            date_val = record.get('date', '')
                            hours_val = record.get('hours', 0)

                            if hours_val and hours_val > 0:
                                # Convert YYYY-MM to YYYY-MM-01
                                if len(str(date_val)) == 7:
                                    date_val = f"{date_val}-01"

                                all_data.append({
                                    'date': str(date_val)[:10],
                                    'lat': record.get('lat'),
                                    'lon': record.get('lon'),
                                    'fishing_hours': float(hours_val),
                                    'vessel_id': record.get('vesselId', ''),
                                    'flag_state': record.get('flag', ''),
                                    'gear_type': record.get('geartype', ''),
                                    'source': 'gfw_ais'
                                })

            if not all_data:
                # Create empty file to mark as checked
                df = pd.DataFrame(columns=[
                    'date', 'lat', 'lon', 'fishing_hours',
                    'vessel_id', 'flag_state', 'gear_type', 'source'
                ])
            else:
                df = pd.DataFrame(all_data)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(output_path, index=False)

            manager.add_download(
                filename=filename,
                period_start=start_date,
                period_end=end_date,
                records=len(df),
                source_url=url,
                api_response_code=response.status_code
            )

            self.log(f"Saved {filename}: {len(df)} records")
            return True

        except Exception as e:
            self.log(f"Error downloading {filename}: {e}")
            return False

    # =========================================================================
    # NOAA SST
    # =========================================================================

    def download_noaa_sst(
        self,
        year: int,
        month: int,
        manager: ManifestManager
    ) -> bool:
        """
        Download NOAA OISST data for a month.

        Uses ERDDAP griddap API.
        """
        filename = f"{year}-{month:02d}.parquet"
        output_path = DataConfig.RAW_SST_NOAA / filename

        if output_path.exists():
            self.log(f"Skipping {filename} (already exists)")
            return True

        start_date, end_date = get_month_range(year, month)
        self.log(f"Downloading NOAA SST: {start_date} to {end_date}")

        if self.dry_run:
            return True

        try:
            # NOAA CoastWatch ERDDAP
            base_url = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg_LonPM180.json"

            query = (
                f"sst[({start_date}T12:00:00Z):1:({end_date}T12:00:00Z)]"
                f"[(0.0):1:(0.0)]"
                f"[({self.region['lat_min']}):1:({self.region['lat_max']})]"
                f"[({self.region['lon_min']}):1:({self.region['lon_max']})]"
            )

            url = f"{base_url}?{query}"
            response = requests.get(url, timeout=180)

            if response.status_code != 200:
                self.log(f"NOAA API error: {response.status_code}")
                return False

            data = response.json()

            # Parse ERDDAP JSON response
            table = data.get('table', {})
            column_names = table.get('columnNames', [])
            rows = table.get('rows', [])

            all_data = []
            for row in rows:
                record = dict(zip(column_names, row))
                # Convert lon if needed (ERDDAP may use different conventions)
                lon = record.get('longitude', record.get('lon', 0))
                lat = record.get('latitude', record.get('lat', 0))
                sst = record.get('sst')

                if sst is not None and not (isinstance(sst, float) and np.isnan(sst)):
                    all_data.append({
                        'date': record.get('time', '')[:10],
                        'lat': lat,
                        'lon': lon,
                        'sst': sst,
                        'source': 'noaa_oisst'
                    })

            if not all_data:
                self.log(f"No valid SST data for {filename}")
                return False

            df = pd.DataFrame(all_data)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(output_path, index=False)

            manager.add_download(
                filename=filename,
                period_start=start_date,
                period_end=end_date,
                records=len(df),
                source_url=base_url,
                api_response_code=200
            )

            self.log(f"Saved {filename}: {len(df)} records")
            return True

        except Exception as e:
            self.log(f"Error downloading {filename}: {e}")
            return False

    # =========================================================================
    # COPERNICUS SST
    # =========================================================================

    def download_copernicus_sst(
        self,
        year: int,
        month: int,
        manager: ManifestManager
    ) -> bool:
        """
        Download Copernicus Marine SST data for a month.

        Uses two datasets:
        - METOFFICE-GLO-SST-L4-REP-OBS-SST: Historical data (1981-2023)
        - METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2: Near real-time (2024+)

        Requires COPERNICUS_USER and COPERNICUS_PASS environment variables.
        """
        if not self.copernicus_user or not self.copernicus_pass:
            self.log("Copernicus credentials not set, skipping")
            return False

        filename = f"{year}-{month:02d}.parquet"
        output_path = DataConfig.RAW_SST_COPERNICUS / filename

        if output_path.exists():
            self.log(f"Skipping {filename} (already exists)")
            return True

        start_date, end_date = get_month_range(year, month)
        self.log(f"Downloading Copernicus SST: {start_date} to {end_date}")

        if self.dry_run:
            return True

        try:
            import copernicusmarine
            import tempfile
            import xarray as xr

            # Create temporary file for NetCDF download
            with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp:
                tmp_path = tmp.name

            # Choose dataset based on year
            # REP dataset: historical data up to 2023
            # NRT dataset: near real-time data from 2024+
            if year <= 2023:
                dataset_id = "METOFFICE-GLO-SST-L4-REP-OBS-SST"
            else:
                dataset_id = "METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2"

            self.log(f"  Using dataset: {dataset_id}")

            # Download SST data from Copernicus
            copernicusmarine.subset(
                dataset_id=dataset_id,
                variables=["analysed_sst"],
                minimum_longitude=self.region['lon_min'],
                maximum_longitude=self.region['lon_max'],
                minimum_latitude=self.region['lat_min'],
                maximum_latitude=self.region['lat_max'],
                start_datetime=f"{start_date}T00:00:00",
                end_datetime=f"{end_date}T23:59:59",
                output_filename=tmp_path,
                username=self.copernicus_user,
                password=self.copernicus_pass,
                overwrite=True
            )

            # Read NetCDF and convert to DataFrame
            # Try different engines to handle various NetCDF formats
            ds = None
            for engine in ['netcdf4', 'h5netcdf', 'scipy']:
                try:
                    ds = xr.open_dataset(tmp_path, engine=engine)
                    break
                except Exception:
                    continue

            if ds is None:
                # Try without engine specification as last resort
                ds = xr.open_dataset(tmp_path)

            all_data = []

            # Get coordinate names (may be lat/lon or latitude/longitude)
            lat_coord = 'latitude' if 'latitude' in ds.dims else 'lat'
            lon_coord = 'longitude' if 'longitude' in ds.dims else 'lon'

            for t in ds.time.values:
                t_str = str(t)[:10]
                for lat_val in ds[lat_coord].values:
                    for lon_val in ds[lon_coord].values:
                        try:
                            sst_val = float(ds.analysed_sst.sel(
                                time=t,
                                **{lat_coord: lat_val, lon_coord: lon_val}
                            ).values)
                            if not np.isnan(sst_val):
                                # Convert from Kelvin to Celsius if needed
                                if sst_val > 100:
                                    sst_val = sst_val - 273.15
                                all_data.append({
                                    'date': t_str,
                                    'lat': float(lat_val),
                                    'lon': float(lon_val),
                                    'sst': sst_val,
                                    'source': 'copernicus_sst'
                                })
                        except:
                            continue

            ds.close()

            # Clean up temp file
            import os as os_module
            os_module.unlink(tmp_path)

            if not all_data:
                self.log(f"No valid SST data for {filename}")
                return False

            df = pd.DataFrame(all_data)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(output_path, index=False)

            manager.add_download(
                filename=filename,
                period_start=start_date,
                period_end=end_date,
                records=len(df),
                source_url='copernicusmarine',
                api_response_code=200
            )

            self.log(f"Saved {filename}: {len(df)} records")
            return True

        except Exception as e:
            self.log(f"Error downloading {filename}: {e}")
            return False

    # =========================================================================
    # MAIN DOWNLOAD LOGIC
    # =========================================================================

    def download_source(
        self,
        source: str,
        start_year: int,
        start_month: int,
        end_year: int,
        end_month: int
    ) -> Dict:
        """
        Download all missing months for a data source.

        Returns stats dictionary.
        """
        print(f"\nDownloading {source}...")

        manager = ManifestManager(source)
        missing = manager.get_missing_months(start_year, start_month, end_year, end_month)

        if not missing:
            print(f"  All months already downloaded for {source}")
            return {'downloaded': 0, 'failed': 0, 'skipped': 0}

        print(f"  {len(missing)} months to download")

        if self.dry_run:
            for year, month in missing:
                print(f"    Would download: {year}-{month:02d}")
            return {'downloaded': 0, 'failed': 0, 'skipped': len(missing)}

        progress = ProgressBar(len(missing), prefix=f"[{source}]")
        stats = {'downloaded': 0, 'failed': 0, 'skipped': 0}

        download_func = {
            'gfw': self.download_gfw,
            'open_meteo': self.download_open_meteo,
            'noaa_sst': self.download_noaa_sst,
            'copernicus_sst': self.download_copernicus_sst,
        }.get(source)

        if not download_func:
            print(f"  Unknown source: {source}")
            return stats

        for year, month in missing:
            success = download_func(year, month, manager)
            if success:
                stats['downloaded'] += 1
            else:
                stats['failed'] += 1
            progress.update()
            time.sleep(0.5)  # Rate limiting between months

        # Save manifest
        manager.save()

        return stats

    def download_all(
        self,
        start_year: int,
        start_month: int,
        end_year: int,
        end_month: int,
        sources: Optional[List[str]] = None
    ) -> Dict:
        """
        Download all sources for date range.

        Args:
            start_year, start_month: Start of range
            end_year, end_month: End of range
            sources: List of sources to download (default: all)

        Returns:
            Dictionary of stats per source
        """
        if sources is None:
            sources = ['open_meteo', 'gfw', 'noaa_sst', 'copernicus_sst']

        print("=" * 60)
        print("INCREMENTAL DATA DOWNLOAD")
        print("=" * 60)
        print(f"Date range: {start_year}-{start_month:02d} to {end_year}-{end_month:02d}")
        print(f"Sources: {', '.join(sources)}")
        print(f"Region: {self.region['name']}")

        if self.dry_run:
            print("\n*** DRY RUN - No files will be created ***")

        # Ensure directories exist
        DataConfig.ensure_directories()

        all_stats = {}
        for source in sources:
            stats = self.download_source(
                source,
                start_year, start_month,
                end_year, end_month
            )
            all_stats[source] = stats

        # Print summary
        print("\n" + "=" * 60)
        print("DOWNLOAD SUMMARY")
        print("=" * 60)

        total_downloaded = 0
        total_failed = 0
        for source, stats in all_stats.items():
            print(f"{source}:")
            print(f"  Downloaded: {stats['downloaded']}")
            print(f"  Failed: {stats['failed']}")
            total_downloaded += stats['downloaded']
            total_failed += stats['failed']

        print(f"\nTotal: {total_downloaded} months downloaded, {total_failed} failed")

        return all_stats


def main():
    parser = argparse.ArgumentParser(
        description='Download data incrementally to Bronze layer'
    )
    parser.add_argument('--start', type=str, default='2020-01',
                       help='Start month (YYYY-MM)')
    parser.add_argument('--end', type=str, default='2026-01',
                       help='End month (YYYY-MM)')
    parser.add_argument('--source', type=str,
                       choices=['gfw', 'open_meteo', 'noaa_sst', 'copernicus_sst', 'all'],
                       default='all', help='Data source to download')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be downloaded without doing it')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print detailed progress')

    args = parser.parse_args()

    start_year, start_month = parse_month(args.start)
    end_year, end_month = parse_month(args.end)

    sources = None if args.source == 'all' else [args.source]

    downloader = IncrementalDownloader(
        verbose=args.verbose,
        dry_run=args.dry_run
    )

    downloader.download_all(
        start_year, start_month,
        end_year, end_month,
        sources=sources
    )


if __name__ == '__main__':
    main()
