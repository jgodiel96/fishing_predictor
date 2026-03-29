#!/usr/bin/env python3
"""
Download Sentinel-2 L2A scenes from Copernicus Data Space Ecosystem (CDSE).

Uses CDSE STAC API (pystac-client + stackstac) to download multiband
GeoTIFF scenes for coastal morphodynamics analysis.

Bands: B02, B03, B04, B08, B8A, B11, B12, SCL (8 bands)
Resolution: 20m (B8A, B11, B12, SCL resampled from native; B02-B04 downsampled)
            10m scenes saved separately for waterline extraction (B03, B08)

Requires:
    pip install pystac-client stackstac rasterio rioxarray
    CDSE account: https://dataspace.copernicus.eu
    S3 keys in .env: CDSE_ACCESS_KEY, CDSE_SECRET_KEY

Usage:
    python scripts/download_sentinel2.py --year 2023
    python scripts/download_sentinel2.py --start 2015 --end 2025
    python scripts/download_sentinel2.py --start 2023 --end 2023 --dry-run
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from calendar import monthrange

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
load_dotenv(ROOT_DIR / '.env')

from data.data_config import DataConfig
from domain import STUDY_AREA


# CDSE STAC configuration
CDSE_STAC_URL = "https://stac.dataspace.copernicus.eu/v1/"
CDSE_S3_ENDPOINT = "eodata.dataspace.copernicus.eu"
COLLECTION = "sentinel-2-l2a"
MAX_CLOUD_COVER = 30

# Bands for 20m classification
BANDS_20M = ["B02", "B03", "B04", "B8A", "B11", "B12", "SCL"]
# Bands for 10m waterline extraction
BANDS_10M = ["B03", "B08"]

# Coastal strip bounding box (narrower than full STUDY_AREA)
# 2km inland + 500m offshore buffer
COASTAL_BBOX = [
    STUDY_AREA.west + 0.01,   # Slightly narrower west (more offshore)
    STUDY_AREA.south,
    STUDY_AREA.east - 0.05,   # Crop inland
    STUDY_AREA.north
]


class Sentinel2Downloader:
    """Downloads Sentinel-2 scenes from CDSE STAC API."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.access_key = os.environ.get('CDSE_ACCESS_KEY', '')
        self.secret_key = os.environ.get('CDSE_SECRET_KEY', '')
        self.output_dir = DataConfig.RAW_SENTINEL2_SCENES

    def log(self, msg: str):
        if self.verbose:
            print(msg)

    def has_credentials(self) -> bool:
        return bool(self.access_key and self.secret_key)

    def setup_environment(self):
        """Configure GDAL/rasterio for CDSE S3 access."""
        os.environ["AWS_S3_ENDPOINT"] = CDSE_S3_ENDPOINT
        os.environ["AWS_ACCESS_KEY_ID"] = self.access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = self.secret_key
        os.environ["AWS_HTTPS"] = "YES"
        os.environ["AWS_VIRTUAL_HOSTING"] = "FALSE"
        os.environ["GDAL_HTTP_TCP_KEEPALIVE"] = "YES"

    def search_scenes(self, year: int, month: int) -> list:
        """Search for Sentinel-2 scenes in CDSE STAC catalog."""
        try:
            from pystac_client import Client
        except ImportError:
            self.log("ERROR: pip install pystac-client")
            return []

        client = Client.open(CDSE_STAC_URL)

        last_day = monthrange(year, month)[1]
        date_range = f"{year}-{month:02d}-01/{year}-{month:02d}-{last_day}"

        search = client.search(
            collections=[COLLECTION],
            bbox=COASTAL_BBOX,
            datetime=date_range,
            query={"eo:cloud_cover": {"lt": MAX_CLOUD_COVER}},
        )

        items = list(search.items())

        # Deduplicate by acquisition date
        unique = {}
        for item in items:
            acq_date = item.datetime.strftime('%Y-%m-%d')
            if acq_date not in unique:
                unique[acq_date] = item

        return sorted(unique.values(), key=lambda x: x.datetime)

    def download_month(self, year: int, month: int, dry_run: bool = False) -> int:
        """Download all scenes for a month.

        Returns number of scenes downloaded.
        """
        month_dir = self.output_dir / f"{year}-{month:02d}"

        scenes = self.search_scenes(year, month)
        self.log(f"  {year}-{month:02d}: {len(scenes)} escenas encontradas (cloud<{MAX_CLOUD_COVER}%)")

        if dry_run:
            for s in scenes:
                cloud = s.properties.get('eo:cloud_cover', '?')
                self.log(f"    {s.datetime.strftime('%Y-%m-%d')} cloud={cloud}%")
            return len(scenes)

        if not scenes:
            return 0

        month_dir.mkdir(parents=True, exist_ok=True)
        downloaded = 0

        for item in scenes:
            date_str = item.datetime.strftime('%Y%m%d')
            output_20m = month_dir / f"S2_{date_str}_20m.tif"
            output_10m = month_dir / f"S2_{date_str}_10m.tif"

            if output_20m.exists():
                self.log(f"    {date_str} ya existe, saltando")
                downloaded += 1
                continue

            try:
                self._download_scene(item, output_20m, output_10m)
                downloaded += 1
                self.log(f"    {date_str} descargado OK")
            except Exception as e:
                self.log(f"    {date_str} ERROR: {e}")

        return downloaded

    def _download_scene(self, item, output_20m: Path, output_10m: Path):
        """Download a single scene as multiband GeoTIFF."""
        import stackstac
        import rasterio
        from rasterio.enums import Resampling

        self.setup_environment()

        # 20m classification bands
        cube_20m = stackstac.stack(
            [item],
            assets=BANDS_20M,
            resolution=20,
            resampling=Resampling.bilinear,
            bounds=COASTAL_BBOX
        )

        data_20m = cube_20m.compute()
        arr_20m = data_20m.values[0]  # (bands, H, W)

        # Handle NaN → 0
        arr_20m = np.nan_to_num(arr_20m, nan=0).astype(np.float32)

        # Save 20m GeoTIFF
        h, w = arr_20m.shape[1], arr_20m.shape[2]
        transform = rasterio.transform.from_bounds(
            *COASTAL_BBOX, w, h
        )

        with rasterio.open(
            str(output_20m), 'w', driver='GTiff',
            height=h, width=w, count=len(BANDS_20M),
            dtype='float32', crs='EPSG:4326',
            transform=transform, compress='lzw'
        ) as dst:
            for i in range(len(BANDS_20M)):
                dst.write(arr_20m[i], i + 1)
            dst.update_tags(
                bands=','.join(BANDS_20M),
                date=item.datetime.strftime('%Y-%m-%d'),
                cloud_cover=str(item.properties.get('eo:cloud_cover', '')),
                source='CDSE_STAC'
            )

        # 10m waterline bands (B03, B08)
        cube_10m = stackstac.stack(
            [item],
            assets=BANDS_10M,
            resolution=10,
            bounds=COASTAL_BBOX
        )

        data_10m = cube_10m.compute()
        arr_10m = data_10m.values[0]
        arr_10m = np.nan_to_num(arr_10m, nan=0).astype(np.float32)

        h10, w10 = arr_10m.shape[1], arr_10m.shape[2]
        transform_10m = rasterio.transform.from_bounds(
            *COASTAL_BBOX, w10, h10
        )

        with rasterio.open(
            str(output_10m), 'w', driver='GTiff',
            height=h10, width=w10, count=2,
            dtype='float32', crs='EPSG:4326',
            transform=transform_10m, compress='lzw'
        ) as dst:
            for i in range(2):
                dst.write(arr_10m[i], i + 1)

    def download_range(self, start_year: int, end_year: int,
                       dry_run: bool = False) -> dict:
        """Download all scenes for a year range."""
        results = {}
        total_scenes = 0

        for year in range(start_year, end_year + 1):
            # Sentinel-2A launched June 2015
            start_month = 6 if year == 2015 else 1
            end_month = 12

            for month in range(start_month, end_month + 1):
                n = self.download_month(year, month, dry_run)
                results[f"{year}-{month:02d}"] = n
                total_scenes += n

        return results


# Need numpy for _download_scene
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description='Download Sentinel-2 L2A from CDSE STAC'
    )
    parser.add_argument('--year', type=int, help='Single year to download')
    parser.add_argument('--start', type=int, default=2023, help='Start year')
    parser.add_argument('--end', type=int, default=2023, help='End year')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only list available scenes, do not download')
    parser.add_argument('--verbose', action='store_true', default=True)

    args = parser.parse_args()

    downloader = Sentinel2Downloader(verbose=args.verbose)

    if not downloader.has_credentials():
        print("=" * 60)
        print("ERROR: Credenciales CDSE no configuradas")
        print("1. Registrate en: https://dataspace.copernicus.eu")
        print("2. Genera S3 keys en User Settings → S3 Access")
        print("3. Agrega a .env:")
        print("   CDSE_ACCESS_KEY=tu_access_key")
        print("   CDSE_SECRET_KEY=tu_secret_key")
        print("=" * 60)
        return 1

    start = args.year or args.start
    end = args.year or args.end

    print("=" * 60)
    print("DESCARGA SENTINEL-2 L2A (CDSE STAC)")
    print("=" * 60)
    print(f"Período: {start} - {end}")
    print(f"Región: {COASTAL_BBOX}")
    print(f"Cloud cover máximo: {MAX_CLOUD_COVER}%")
    print(f"Bandas 20m: {BANDS_20M}")
    print(f"Bandas 10m: {BANDS_10M}")
    if args.dry_run:
        print("MODO DRY-RUN (solo listar)")
    print("=" * 60)

    results = downloader.download_range(start, end, args.dry_run)

    print(f"\nTotal: {sum(results.values())} escenas")
    return 0


if __name__ == '__main__':
    sys.exit(main())
