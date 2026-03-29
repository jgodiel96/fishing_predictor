#!/usr/bin/env python3
"""
Incremental Sentinel-2 L2A download from CDSE.

1. Searches CDSE STAC for scenes matching date/cloud/region filters
2. Downloads specific bands via S3 (boto3) from eodata bucket
3. Merges bands into a single multiband GeoTIFF per scene
4. Tracks progress — can be stopped and resumed at any time

Usage:
    python scripts/download_sentinel2.py --month 2023-01
    python scripts/download_sentinel2.py --year 2023
    python scripts/download_sentinel2.py --start 2015 --end 2025
    python scripts/download_sentinel2.py --status
    python scripts/download_sentinel2.py --year 2023 --dry-run
"""

import os
import sys
import json
import time
import argparse
import tempfile
from pathlib import Path
from datetime import datetime
from calendar import monthrange

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
load_dotenv(ROOT_DIR / '.env')

from data.data_config import DataConfig
from domain import STUDY_AREA

# CDSE config
CDSE_STAC_URL = "https://stac.dataspace.copernicus.eu/v1/"
S3_ENDPOINT = "https://eodata.dataspace.copernicus.eu"
S3_BUCKET = "eodata"

MAX_CLOUD_COVER = 30

# Bands to download
# 20m: B02(10m→20m), B03(10m→20m), B04(10m→20m), B8A(20m), B11(20m), B12(20m), SCL(20m)
# 10m: B03(10m), B08(10m) — for waterline extraction
BANDS_20M = ["B02", "B03", "B04", "B8A", "B11", "B12", "SCL"]
BANDS_10M = ["B03", "B08"]

PROGRESS_FILE = DataConfig.RAW_SENTINEL2 / "_download_progress.json"

PAUSE_BETWEEN_SCENES = 2
PAUSE_ON_ERROR = 10


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {'downloaded': [], 'failed': [], 'last_update': ''}


def save_progress(progress: dict):
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    progress['last_update'] = datetime.now().isoformat()
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def progress_bar(current, total, prefix='', width=40):
    pct = current / total if total > 0 else 0
    filled = int(width * pct)
    bar = '█' * filled + '░' * (width - filled)
    sys.stdout.write(f'\r  {prefix} [{bar}] {current}/{total} ({pct*100:.0f}%)')
    sys.stdout.flush()
    if current >= total:
        print()


class Sentinel2Downloader:
    """Downloads Sentinel-2 bands via STAC search + S3 download."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.access_key = os.environ.get('CDSE_ACCESS_KEY', '')
        self.secret_key = os.environ.get('CDSE_SECRET_KEY', '')
        self.output_dir = DataConfig.RAW_SENTINEL2_SCENES
        self.progress = load_progress()
        self._s3 = None

    def log(self, msg: str):
        if self.verbose:
            print(msg)

    def has_credentials(self) -> bool:
        return bool(self.access_key and self.secret_key)

    def get_s3(self):
        """Get boto3 S3 resource for CDSE eodata bucket."""
        if self._s3:
            return self._s3
        import boto3
        session = boto3.session.Session()
        self._s3 = session.resource(
            's3',
            endpoint_url=S3_ENDPOINT,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name='default'
        )
        return self._s3

    def search_month(self, year: int, month: int) -> list:
        """Search CDSE STAC for scenes. Returns list of STAC items."""
        from pystac_client import Client

        last_day = monthrange(year, month)[1]
        date_range = f"{year}-{month:02d}-01/{year}-{month:02d}-{last_day}"

        bbox = [STUDY_AREA.west, STUDY_AREA.south,
                STUDY_AREA.east, STUDY_AREA.north]

        for attempt in range(3):
            try:
                client = Client.open(CDSE_STAC_URL)
                search = client.search(
                    collections=["sentinel-2-l2a"],
                    bbox=bbox,
                    datetime=date_range,
                    query={"eo:cloud_cover": {"lt": MAX_CLOUD_COVER}},
                )
                items = list(search.items())

                # Deduplicate by date
                unique = {}
                for item in items:
                    d = item.datetime.strftime('%Y-%m-%d')
                    if d not in unique:
                        unique[d] = item
                return sorted(unique.values(), key=lambda x: x.datetime)

            except Exception as e:
                if attempt < 2:
                    self.log(f"    Retry {attempt+1}/3: {type(e).__name__}")
                    time.sleep(PAUSE_ON_ERROR * (attempt + 1))
                else:
                    self.log(f"    ERROR buscando {year}-{month:02d}: {e}")
                    return []

    def find_band_path(self, item, band_name: str) -> str:
        """Find the S3 path to a specific band in a STAC item."""
        # Try direct asset
        if band_name in item.assets:
            href = item.assets[band_name].href
            # Convert HTTP URL to S3 key
            if '/eodata/' in href:
                return href.split('/eodata/')[1]
            return href

        # Try alternate keys
        alt_keys = {
            'B02': ['B02_10m', 'B02'], 'B03': ['B03_10m', 'B03'],
            'B04': ['B04_10m', 'B04'], 'B08': ['B08_10m', 'B08'],
            'B8A': ['B8A_20m', 'B8A'], 'B11': ['B11_20m', 'B11'],
            'B12': ['B12_20m', 'B12'], 'SCL': ['SCL_20m', 'SCL']
        }
        for key in alt_keys.get(band_name, []):
            if key in item.assets:
                href = item.assets[key].href
                if '/eodata/' in href:
                    return href.split('/eodata/')[1]
                return href

        # Fallback: search in assets by band name pattern
        for asset_key, asset in item.assets.items():
            if band_name in asset_key or band_name.lower() in asset_key.lower():
                href = asset.href
                if '/eodata/' in href:
                    return href.split('/eodata/')[1]
                return href

        return None

    def download_band(self, s3_key: str, local_path: str) -> bool:
        """Download a single band file from S3."""
        try:
            s3 = self.get_s3()
            bucket = s3.Bucket(S3_BUCKET)
            bucket.download_file(s3_key, local_path)
            return True
        except Exception as e:
            self.log(f"      S3 download error: {e}")
            return False

    def download_scene(self, item, month_dir: Path) -> bool:
        """Download all bands for a scene and merge into multiband GeoTIFF."""
        import numpy as np

        date_str = item.datetime.strftime('%Y%m%d')
        scene_id = f"S2_{date_str}"

        if scene_id in self.progress['downloaded']:
            return True

        output_file = month_dir / f"{scene_id}_bands.tif"
        if output_file.exists():
            self.progress['downloaded'].append(scene_id)
            save_progress(self.progress)
            return True

        try:
            import rasterio
            from rasterio.merge import merge
        except ImportError:
            self.log("ERROR: pip install rasterio")
            return False

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                band_files = []
                all_bands = BANDS_20M

                for band in all_bands:
                    s3_key = self.find_band_path(item, band)
                    if not s3_key:
                        self.log(f"      {band}: no encontrado en assets")
                        continue

                    local_path = os.path.join(tmpdir, f"{band}.jp2")
                    if self.download_band(s3_key, local_path):
                        band_files.append((band, local_path))
                    else:
                        self.log(f"      {band}: descarga fallida")

                if len(band_files) < 3:
                    self.log(f"      Solo {len(band_files)} bandas — insuficiente")
                    return False

                # Read all bands and stack into multiband GeoTIFF
                arrays = []
                ref_profile = None

                for band_name, band_path in band_files:
                    with rasterio.open(band_path) as src:
                        data = src.read(1).astype(np.float32)
                        if ref_profile is None:
                            ref_profile = src.profile.copy()
                        else:
                            if data.shape != (ref_profile['height'], ref_profile['width']):
                                from rasterio.enums import Resampling
                                from rasterio.warp import reproject
                                dst_data = np.empty(
                                    (ref_profile['height'], ref_profile['width']),
                                    dtype=np.float32
                                )
                                reproject(
                                    source=data, destination=dst_data,
                                    src_transform=src.transform, src_crs=src.crs,
                                    dst_transform=ref_profile['transform'],
                                    dst_crs=ref_profile['crs'],
                                    resampling=Resampling.bilinear
                                )
                                data = dst_data
                        arrays.append(data)

                if not arrays or ref_profile is None:
                    return False

                stack = np.stack(arrays, axis=0)
                ref_profile.update(
                    driver='GTiff', count=len(arrays),
                    dtype='float32', compress='lzw'
                )

                with rasterio.open(str(output_file), 'w', **ref_profile) as dst:
                    dst.write(stack)
                    dst.update_tags(
                        bands=','.join(b for b, _ in band_files),
                        date=item.datetime.strftime('%Y-%m-%d'),
                        cloud_cover=str(item.properties.get('eo:cloud_cover', '')),
                        source='CDSE_S3'
                    )

                self.progress['downloaded'].append(scene_id)
                save_progress(self.progress)
                return True

        except Exception as e:
            self.log(f"      ERROR: {e}")
            if scene_id not in self.progress['failed']:
                self.progress['failed'].append(scene_id)
                save_progress(self.progress)
            return False

    def download_month(self, year: int, month: int, dry_run: bool = False) -> dict:
        """Download all scenes for a month."""
        month_key = f"{year}-{month:02d}"
        month_dir = self.output_dir / month_key

        self.log(f"\n[{month_key}] Buscando escenas...")
        scenes = self.search_month(year, month)

        n_total = len(scenes)
        n_existing = sum(1 for s in scenes
                        if f"S2_{s.datetime.strftime('%Y%m%d')}" in self.progress['downloaded'])
        n_new = n_total - n_existing

        self.log(f"  {n_total} escenas, {n_existing} descargadas, {n_new} nuevas")

        if dry_run:
            for s in scenes:
                cloud = s.properties.get('eo:cloud_cover', 0)
                sid = f"S2_{s.datetime.strftime('%Y%m%d')}"
                status = "✓" if sid in self.progress['downloaded'] else "pendiente"
                self.log(f"    {s.datetime.strftime('%Y-%m-%d')} "
                         f"cloud={cloud:.1f}% [{status}]")
            return {'total': n_total, 'downloaded': 0, 'failed': 0}

        if n_new == 0:
            return {'total': n_total, 'downloaded': 0, 'failed': 0}

        month_dir.mkdir(parents=True, exist_ok=True)
        downloaded = 0
        failed = 0

        for i, item in enumerate(scenes):
            sid = f"S2_{item.datetime.strftime('%Y%m%d')}"
            if sid in self.progress['downloaded']:
                continue

            cloud = item.properties.get('eo:cloud_cover', 0)
            progress_bar(n_existing + downloaded, n_total,
                        prefix=f"{month_key} {item.datetime.strftime('%m-%d')}")

            if self.download_scene(item, month_dir):
                downloaded += 1
            else:
                failed += 1

            time.sleep(PAUSE_BETWEEN_SCENES)

        progress_bar(n_total, n_total, prefix=f"{month_key} completo")

        return {'total': n_total, 'downloaded': downloaded, 'failed': failed}

    def download_range(self, start_year: int, end_year: int,
                       dry_run: bool = False):
        """Download months incrementally."""
        months = []
        for year in range(start_year, end_year + 1):
            sm = 6 if year == 2015 else 1
            for month in range(sm, 13):
                months.append((year, month))

        total_months = len(months)
        done_months = sum(1 for y, m in months
                         if f"{y}-{m:02d}" in
                         [d[:7] for d in self.progress['downloaded']])

        self.log(f"\n  Meses: {total_months} | Con datos: {done_months}")

        total_dl = 0
        total_fail = 0

        for y, m in months:
            stats = self.download_month(y, m, dry_run)
            total_dl += stats['downloaded']
            total_fail += stats['failed']

        self.log(f"\n{'='*60}")
        self.log(f"  RESUMEN: {total_dl} escenas descargadas, {total_fail} fallidas")
        self.log(f"  Total acumulado: {len(self.progress['downloaded'])} escenas")
        self.log(f"{'='*60}")

    def show_status(self):
        n_done = len(self.progress['downloaded'])
        n_failed = len(self.progress['failed'])
        last = self.progress.get('last_update', 'nunca')
        total_est = 650  # ~650 scenes over 10 years

        print(f"\n{'='*60}")
        print(f"  SENTINEL-2 DOWNLOAD STATUS")
        print(f"{'='*60}")
        progress_bar(n_done, total_est, prefix='Total')
        print(f"  Descargadas: {n_done}")
        print(f"  Fallidas:    {n_failed}")
        print(f"  Última:      {last}")

        if n_done > 0:
            by_year = {}
            for sid in self.progress['downloaded']:
                y = sid[3:7]
                by_year[y] = by_year.get(y, 0) + 1
            print(f"\n  Por año:")
            for y in sorted(by_year):
                bar = '█' * by_year[y]
                print(f"    {y}: {bar} ({by_year[y]})")

        if n_failed > 0:
            print(f"\n  Fallidas:")
            for sid in self.progress['failed'][:10]:
                print(f"    {sid}")

        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='Descarga incremental Sentinel-2 L2A desde CDSE'
    )
    parser.add_argument('--month', type=str, help='Mes (YYYY-MM)')
    parser.add_argument('--year', type=int, help='Año completo')
    parser.add_argument('--start', type=int, default=2023, help='Año inicio')
    parser.add_argument('--end', type=int, default=2023, help='Año fin')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--retry-failed', action='store_true')

    args = parser.parse_args()

    downloader = Sentinel2Downloader()

    if args.status:
        downloader.show_status()
        return 0

    if not downloader.has_credentials():
        print("ERROR: CDSE_ACCESS_KEY y CDSE_SECRET_KEY no configurados en .env")
        print("Genera keys en: https://eodata-s3keysmanager.dataspace.copernicus.eu/")
        return 1

    if args.retry_failed:
        downloader.progress['failed'] = []
        save_progress(downloader.progress)
        print("Lista de fallidos limpiada")

    print(f"{'='*60}")
    print(f"  DESCARGA SENTINEL-2 L2A (CDSE STAC + S3)")
    print(f"{'='*60}")

    if args.month:
        parts = args.month.split('-')
        downloader.download_month(int(parts[0]), int(parts[1]), args.dry_run)
    elif args.year:
        downloader.download_range(args.year, args.year, args.dry_run)
    else:
        downloader.download_range(args.start, args.end, args.dry_run)

    return 0


if __name__ == '__main__':
    sys.exit(main())
