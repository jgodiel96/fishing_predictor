#!/usr/bin/env python3
"""
Download OSM Coastline Data for Peru.

Downloads water polygon data from OpenStreetMap for the Peru coastal region.
Data source: https://osmdata.openstreetmap.de/data/coastlines.html

The full global file is ~500MB, but we only need a small subset for Peru.
This script offers two options:
1. Download full global data and extract Peru subset
2. Download pre-extracted Peru subset (if available)
"""

import sys
import os
import zipfile
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Peru coastal bounds (generous to include all fishing areas)
PERU_BOUNDS = {
    'lat_min': -18.5,  # Southern tip
    'lat_max': -3.5,   # Northern tip
    'lon_min': -82.0,  # Western ocean
    'lon_max': -76.0   # Eastern land
}

# OSM Data URLs
WATER_POLYGONS_URL = "https://osmdata.openstreetmap.de/download/water-polygons-split-4326.zip"


def download_global_coastline():
    """Download the full global water polygons file (~500MB)."""
    try:
        import requests
    except ImportError:
        logger.error("requests library required: pip install requests")
        return False

    output_dir = project_root / 'data' / 'coastlines'
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_path = output_dir / 'water-polygons-split-4326.zip'

    if zip_path.exists():
        logger.info(f"Zip file already exists: {zip_path}")
        response = input("Re-download? (y/n): ")
        if response.lower() != 'y':
            return extract_zip(zip_path, output_dir)

    logger.info("Downloading OSM water polygons (~500MB)...")
    logger.info("This may take several minutes...")

    try:
        response = requests.get(WATER_POLYGONS_URL, stream=True, timeout=600)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    mb_down = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    print(f"\rProgress: {percent:.1f}% ({mb_down:.1f}/{mb_total:.1f} MB)", end='')

        print()  # New line after progress
        logger.info("Download complete!")

        return extract_zip(zip_path, output_dir)

    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


def extract_zip(zip_path: Path, output_dir: Path) -> bool:
    """Extract the downloaded zip file."""
    logger.info(f"Extracting {zip_path}...")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)

        logger.info(f"Extracted to {output_dir}")

        # List extracted files
        for f in output_dir.glob('**/*.shp'):
            logger.info(f"  Found shapefile: {f.name}")

        return True

    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return False


def extract_peru_subset():
    """Extract only the Peru coastal region from the global data."""
    try:
        import shapefile
        from shapely.geometry import shape, box
    except ImportError:
        logger.error("Required libraries: pip install pyshp shapely")
        return False

    input_dir = project_root / 'data' / 'coastlines'
    shp_files = list(input_dir.glob('**/water_polygons*.shp'))

    if not shp_files:
        shp_files = list(input_dir.glob('**/water-polygons*.shp'))

    if not shp_files:
        logger.error("No shapefile found. Run download first.")
        return False

    shp_path = shp_files[0]
    logger.info(f"Processing {shp_path}...")

    output_path = input_dir / 'water_polygons_peru.shp'

    # Create bounding box
    bbox = box(
        PERU_BOUNDS['lon_min'],
        PERU_BOUNDS['lat_min'],
        PERU_BOUNDS['lon_max'],
        PERU_BOUNDS['lat_max']
    )

    peru_shapes = []
    peru_records = []

    try:
        with shapefile.Reader(str(shp_path)) as sf:
            total = len(sf)
            logger.info(f"Total shapes in global file: {total:,}")

            for i, (shp_rec, rec) in enumerate(zip(sf.iterShapes(), sf.iterRecords())):
                if i % 10000 == 0:
                    print(f"\rProcessing: {i:,}/{total:,}", end='')

                # Quick bounding box check
                shp_bbox = shp_rec.bbox  # (minx, miny, maxx, maxy)

                if (shp_bbox[2] < PERU_BOUNDS['lon_min'] or
                    shp_bbox[0] > PERU_BOUNDS['lon_max'] or
                    shp_bbox[3] < PERU_BOUNDS['lat_min'] or
                    shp_bbox[1] > PERU_BOUNDS['lat_max']):
                    continue

                # More precise intersection check
                geom = shape(shp_rec.__geo_interface__)
                if geom.intersects(bbox):
                    clipped = geom.intersection(bbox)
                    if not clipped.is_empty:
                        peru_shapes.append(clipped)
                        peru_records.append(rec)

        print()  # New line after progress
        logger.info(f"Found {len(peru_shapes)} shapes in Peru region")

        if peru_shapes:
            # Write to new shapefile
            with shapefile.Writer(str(output_path)) as w:
                # Copy field definitions
                w.fields = sf.fields[1:]  # Skip deletion flag

                for geom, rec in zip(peru_shapes, peru_records):
                    # Handle different geometry types
                    if geom.geom_type == 'Polygon':
                        w.poly([list(geom.exterior.coords)])
                    elif geom.geom_type == 'MultiPolygon':
                        parts = [list(p.exterior.coords) for p in geom.geoms]
                        w.poly(parts)
                    w.record(*rec)

            logger.info(f"Peru subset saved to: {output_path}")
            return True

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return False


def create_sample_coastline():
    """Create a sample coastline for Ilo area without downloading full data."""
    logger.info("Creating sample coastline for Ilo area...")

    output_dir = project_root / 'data' / 'coastlines'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ilo coastline coordinates (approximate, traced from satellite imagery)
    ilo_coastline = [
        (-17.6400, -71.3700),
        (-17.6450, -71.3650),
        (-17.6500, -71.3580),
        (-17.6550, -71.3550),
        (-17.6580, -71.3530),
        (-17.6600, -71.3510),
        (-17.6620, -71.3500),
        (-17.6650, -71.3490),
        (-17.6680, -71.3480),
        (-17.6700, -71.3470),
        (-17.6720, -71.3460),
        (-17.6750, -71.3450),
        (-17.6780, -71.3440),
        (-17.6800, -71.3430),
    ]

    # Create a GeoJSON file instead (simpler, no dependencies)
    import json

    # Create water polygon (west of coastline)
    water_coords = [[-71.40, -17.63]]  # Start northwest
    for lat, lon in ilo_coastline:
        water_coords.append([lon, lat])
    water_coords.append([-71.40, -17.69])  # Southwest corner
    water_coords.append([-71.40, -17.63])  # Close polygon

    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "Ilo Coastal Water", "source": "manual"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [water_coords]
                }
            }
        ]
    }

    output_path = output_dir / 'ilo_coastline.geojson'
    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)

    logger.info(f"Sample coastline saved to: {output_path}")
    return True


def main():
    """Main menu."""
    print("\n" + "="*60)
    print("OSM COASTLINE DATA DOWNLOADER")
    print("="*60)
    print("\nOptions:")
    print("1. Download full global data (~500MB) and extract Peru")
    print("2. Create sample coastline for Ilo (no download)")
    print("3. Extract Peru from existing download")
    print("4. Exit")

    choice = input("\nSelect option (1-4): ").strip()

    if choice == '1':
        if download_global_coastline():
            logger.info("\nGlobal data downloaded.")
            response = input("Extract Peru subset now? (y/n): ")
            if response.lower() == 'y':
                extract_peru_subset()
    elif choice == '2':
        create_sample_coastline()
    elif choice == '3':
        extract_peru_subset()
    elif choice == '4':
        print("Exiting.")
    else:
        print("Invalid option.")


if __name__ == '__main__':
    main()
