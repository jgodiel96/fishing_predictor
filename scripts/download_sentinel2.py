#!/usr/bin/env python3
"""
Incremental Sentinel-2 L2A download from CDSE STAC.

Downloads scenes one at a time with progress tracking. Can be stopped
and resumed — already downloaded scenes are skipped automatically.

Usage:
    python scripts/download_sentinel2.py --month 2023-01          # Single month
    python scripts/download_sentinel2.py --year 2023              # Full year
    python scripts/download_sentinel2.py --start 2015 --end 2025  # 10 years
    python scripts/download_sentinel2.py --year 2023 --dry-run    # List only
    python scripts/download_sentinel2.py --status                 # Show progress
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from calendar import monthrange

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
load_dotenv(ROOT_DIR / '.env')

from data.data_config import DataConfig
from domain import STUDY_AREA

# CDSE STAC
CDSE_STAC_URL = "https://stac.dataspace.copernicus.eu/v1/"
CDSE_S3_ENDPOINT = "eodata.dataspace.copernicus.eu"
COLLECTION = "sentinel-2-l2a"
MAX_CLOUD_COVER = 30

# Bands
BANDS_20M = ["B02", "B03", "B04", "B8A", "B11", "B12", "SCL"]
BANDS_10M = ["B03", "B08"]

# Full study area — no need to limit with incremental downloads
COASTAL_BBOX = [
    STUDY_AREA.west,
    STUDY_AREA.south,
    STUDY_AREA.east,
    STUDY_AREA.north
]

# Progress file
PROGRESS_FILE = DataConfig.RAW_SENTINEL2 / "_download_progress.json"

# Pause between scenes (seconds) to avoid server overload
PAUSE_BETWEEN_SCENES = 3
PAUSE_BETWEEN_MONTHS = 5
PAUSE_ON_ERROR = 15


def load_progress() -> dict:
    """Load download progress from disk."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {'downloaded': [], 'failed': [], 'last_update': ''}


def save_progress(progress: dict):
    """Save download progress to disk."""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    progress['last_update'] = datetime.now().isoformat()
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


class Sentinel2Downloader:
    """Incremental Sentinel-2 downloader with progress tracking."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.access_key = os.environ.get('CDSE_ACCESS_KEY', '')
        self.secret_key = os.environ.get('CDSE_SECRET_KEY', '')
        self.output_dir = DataConfig.RAW_SENTINEL2_SCENES
        self.progress = load_progress()

    def log(self, msg: str):
        if self.verbose:
            print(msg)

    def has_credentials(self) -> bool:
        return bool(self.access_key and self.secret_key)

    def setup_s3(self):
        """Configure GDAL/rasterio for CDSE S3 access."""
        os.environ["AWS_S3_ENDPOINT"] = CDSE_S3_ENDPOINT
        os.environ["AWS_ACCESS_KEY_ID"] = self.access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = self.secret_key
        os.environ["AWS_HTTPS"] = "YES"
        os.environ["AWS_VIRTUAL_HOSTING"] = "FALSE"
        os.environ["GDAL_HTTP_TCP_KEEPALIVE"] = "YES"

    def search_month(self, year: int, month: int) -> list:
        """Search CDSE STAC for scenes in a month. Retries on failure."""
        from pystac_client import Client

        last_day = monthrange(year, month)[1]
        date_range = f"{year}-{month:02d}-01/{year}-{month:02d}-{last_day}"

        for attempt in range(3):
            try:
                client = Client.open(CDSE_STAC_URL)
                search = client.search(
                    collections=[COLLECTION],
                    bbox=COASTAL_BBOX,
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
                    self.log(f"    Retry {attempt+1}/3: {e}")
                    time.sleep(PAUSE_ON_ERROR * (attempt + 1))
                else:
                    self.log(f"    ERROR buscando {year}-{month:02d}: {e}")
                    return []

    def download_scene(self, item, month_dir: Path) -> bool:
        """Download a single scene as GeoTIFF. Returns True on success."""
        import stackstac
        import rasterio
        from rasterio.enums import Resampling

        date_str = item.datetime.strftime('%Y%m%d')
        scene_id = f"S2_{date_str}"

        # Skip if already downloaded
        if scene_id in self.progress['downloaded']:
            return True

        output_20m = month_dir / f"{scene_id}_20m.tif"
        output_10m = month_dir / f"{scene_id}_10m.tif"

        if output_20m.exists():
            self.progress['downloaded'].append(scene_id)
            save_progress(self.progress)
            return True

        self.setup_s3()

        try:
            # 20m classification bands
            cube_20m = stackstac.stack(
                [item], assets=BANDS_20M,
                resolution=20, resampling=Resampling.bilinear,
                bounds=COASTAL_BBOX
            )
            data_20m = cube_20m.compute()
            arr_20m = np.nan_to_num(data_20m.values[0], nan=0).astype(np.float32)

            h, w = arr_20m.shape[1], arr_20m.shape[2]
            transform = rasterio.transform.from_bounds(*COASTAL_BBOX, w, h)

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

            # 10m waterline bands
            cube_10m = stackstac.stack(
                [item], assets=BANDS_10M,
                resolution=10, bounds=COASTAL_BBOX
            )
            data_10m = cube_10m.compute()
            arr_10m = np.nan_to_num(data_10m.values[0], nan=0).astype(np.float32)

            h10, w10 = arr_10m.shape[1], arr_10m.shape[2]
            transform_10m = rasterio.transform.from_bounds(*COASTAL_BBOX, w10, h10)

            with rasterio.open(
                str(output_10m), 'w', driver='GTiff',
                height=h10, width=w10, count=2,
                dtype='float32', crs='EPSG:4326',
                transform=transform_10m, compress='lzw'
            ) as dst:
                for i in range(2):
                    dst.write(arr_10m[i], i + 1)

            # Track progress
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
        """Download all scenes for a single month. Returns stats."""
        month_key = f"{year}-{month:02d}"
        month_dir = self.output_dir / month_key

        self.log(f"\n[{month_key}] Buscando escenas...")
        scenes = self.search_month(year, month)

        n_total = len(scenes)
        n_existing = sum(1 for s in scenes
                        if f"S2_{s.datetime.strftime('%Y%m%d')}" in self.progress['downloaded'])
        n_new = n_total - n_existing

        self.log(f"  {n_total} escenas (cloud<{MAX_CLOUD_COVER}%), "
                 f"{n_existing} ya descargadas, {n_new} nuevas")

        if dry_run:
            for s in scenes:
                cloud = s.properties.get('eo:cloud_cover', 0)
                sid = f"S2_{s.datetime.strftime('%Y%m%d')}"
                status = "OK" if sid in self.progress['downloaded'] else "pendiente"
                self.log(f"    {s.datetime.strftime('%Y-%m-%d')} "
                         f"cloud={cloud:.1f}% [{status}]")
            return {'total': n_total, 'new': n_new, 'downloaded': 0, 'failed': 0}

        if n_new == 0:
            self.log(f"  Todo descargado, saltando")
            return {'total': n_total, 'new': 0, 'downloaded': 0, 'failed': 0}

        month_dir.mkdir(parents=True, exist_ok=True)
        downloaded = 0
        failed = 0

        for i, item in enumerate(scenes):
            sid = f"S2_{item.datetime.strftime('%Y%m%d')}"
            if sid in self.progress['downloaded']:
                continue

            cloud = item.properties.get('eo:cloud_cover', 0)
            self.log(f"    [{i+1}/{n_total}] {item.datetime.strftime('%Y-%m-%d')} "
                     f"cloud={cloud:.1f}%...")

            if self.download_scene(item, month_dir):
                downloaded += 1
                self.log(f"      OK")
            else:
                failed += 1

            # Pause between scenes
            if i < len(scenes) - 1:
                time.sleep(PAUSE_BETWEEN_SCENES)

        return {'total': n_total, 'new': n_new, 'downloaded': downloaded, 'failed': failed}

    def download_range(self, start_year: int, end_year: int,
                       dry_run: bool = False) -> dict:
        """Download scenes incrementally across a year range."""
        all_stats = {}
        total_downloaded = 0
        total_failed = 0

        for year in range(start_year, end_year + 1):
            start_month = 6 if year == 2015 else 1

            for month in range(start_month, 13):
                stats = self.download_month(year, month, dry_run)
                all_stats[f"{year}-{month:02d}"] = stats
                total_downloaded += stats['downloaded']
                total_failed += stats['failed']

                # Pause between months
                if not dry_run and stats['downloaded'] > 0:
                    time.sleep(PAUSE_BETWEEN_MONTHS)

        self.log(f"\n{'='*60}")
        self.log(f"RESUMEN: {total_downloaded} descargadas, {total_failed} fallidas")
        self.log(f"Total en progreso: {len(self.progress['downloaded'])} escenas")
        self.log(f"{'='*60}")

        return all_stats

    def show_status(self):
        """Show current download progress."""
        n_downloaded = len(self.progress['downloaded'])
        n_failed = len(self.progress['failed'])
        last = self.progress.get('last_update', 'nunca')

        print(f"{'='*60}")
        print(f"ESTADO DE DESCARGA SENTINEL-2")
        print(f"{'='*60}")
        print(f"Descargadas: {n_downloaded}")
        print(f"Fallidas:    {n_failed}")
        print(f"Última:      {last}")

        if n_downloaded > 0:
            # Count by year
            by_year = {}
            for sid in self.progress['downloaded']:
                year = sid[3:7]
                by_year[year] = by_year.get(year, 0) + 1
            print(f"\nPor año:")
            for year in sorted(by_year):
                print(f"  {year}: {by_year[year]} escenas")

        if n_failed > 0:
            print(f"\nFallidas (se reintentarán):")
            for sid in self.progress['failed'][:10]:
                print(f"  {sid}")
            if n_failed > 10:
                print(f"  ... y {n_failed - 10} más")


def main():
    parser = argparse.ArgumentParser(
        description='Descarga incremental de Sentinel-2 L2A desde CDSE'
    )
    parser.add_argument('--month', type=str, help='Mes específico (YYYY-MM)')
    parser.add_argument('--year', type=int, help='Año completo')
    parser.add_argument('--start', type=int, default=2023, help='Año inicio')
    parser.add_argument('--end', type=int, default=2023, help='Año fin')
    parser.add_argument('--dry-run', action='store_true',
                        help='Solo listar escenas disponibles')
    parser.add_argument('--status', action='store_true',
                        help='Mostrar estado de descarga')
    parser.add_argument('--retry-failed', action='store_true',
                        help='Reintentar escenas fallidas')

    args = parser.parse_args()

    downloader = Sentinel2Downloader()

    if args.status:
        downloader.show_status()
        return 0

    if not downloader.has_credentials():
        print("ERROR: Credenciales CDSE no configuradas")
        print("Agrega a .env: CDSE_ACCESS_KEY y CDSE_SECRET_KEY")
        print("Genera keys en: https://eodata-s3keysmanager.dataspace.copernicus.eu/")
        return 1

    # Clear failed list if retrying
    if args.retry_failed:
        downloader.progress['failed'] = []
        save_progress(downloader.progress)
        print("Lista de fallidos limpiada")

    if args.month:
        parts = args.month.split('-')
        year, month = int(parts[0]), int(parts[1])
        print(f"Descargando {args.month}...")
        downloader.download_month(year, month, args.dry_run)
    elif args.year:
        print(f"Descargando año {args.year}...")
        downloader.download_range(args.year, args.year, args.dry_run)
    else:
        print(f"Descargando {args.start} - {args.end}...")
        downloader.download_range(args.start, args.end, args.dry_run)

    return 0


if __name__ == '__main__':
    sys.exit(main())
