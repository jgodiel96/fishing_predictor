#!/usr/bin/env python3
"""
Coastline Filter - Water/Land Mask Based Filtering

Uses satellite imagery to create a water mask and filter out
points that are incorrectly placed on land.

This addresses the fundamental problem: we need ground truth
about what is water and what is land.

Author: Fishing Predictor Project
Date: 2026-02-02
Version: 5.3
"""

import os
import sys
import math
import json
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
import numpy as np

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

try:
    from PIL import Image
    import requests
    from io import BytesIO
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class TileBounds(NamedTuple):
    """Geographic bounds of a tile."""
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


@dataclass
class FilterConfig:
    """Configuration for coastline filtering."""
    # Tile settings
    zoom_level: int = 13  # Lower zoom = faster, less detail
    tile_size: int = 256

    # Water detection (HSV ranges for blue water)
    water_hue_min: int = 85
    water_hue_max: int = 135
    water_sat_min: int = 20
    water_val_min: int = 30

    # Filtering
    water_threshold: float = 0.3  # Min fraction of water around point
    check_radius_pixels: int = 5  # Radius to check around each point

    # Output
    save_debug_images: bool = True
    debug_dir: str = "output/coastline_debug"


class CoastlineFilter:
    """
    Filters coastline points using satellite imagery water detection.

    Uses ESRI World Imagery tiles to determine water vs land,
    then filters out points that are on land.
    """

    ESRI_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

    def __init__(self, config: Optional[FilterConfig] = None):
        self.config = config or FilterConfig()
        self.cache_dir = ROOT_DIR / "data" / "cache" / "tiles" / "satellite"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if self.config.save_debug_images:
            debug_path = ROOT_DIR / self.config.debug_dir
            debug_path.mkdir(parents=True, exist_ok=True)

    def lat_lon_to_tile(self, lat: float, lon: float, zoom: int) -> Tuple[int, int]:
        """Convert lat/lon to tile coordinates."""
        n = 2 ** zoom
        x = int((lon + 180) / 360 * n)
        lat_rad = math.radians(lat)
        y = int((1 - math.asinh(math.tan(lat_rad)) / math.pi) / 2 * n)
        return x, y

    def tile_to_bounds(self, x: int, y: int, zoom: int) -> TileBounds:
        """Get geographic bounds of a tile."""
        n = 2 ** zoom
        lon_min = x / n * 360 - 180
        lon_max = (x + 1) / n * 360 - 180
        lat_max = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
        lat_min = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
        return TileBounds(lat_min, lat_max, lon_min, lon_max)

    def fetch_tile(self, x: int, y: int, zoom: int) -> Optional[np.ndarray]:
        """Fetch a satellite tile, using cache if available."""
        if not PIL_AVAILABLE:
            return None

        cache_file = self.cache_dir / f"{zoom}_{x}_{y}.png"

        if cache_file.exists():
            img = Image.open(cache_file).convert("RGB")
            return np.array(img)

        url = self.ESRI_URL.format(z=zoom, x=x, y=y)

        try:
            headers = {'User-Agent': 'FishingPredictor/1.0'}
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            img = Image.open(BytesIO(response.content)).convert("RGB")
            img.save(cache_file)
            return np.array(img)

        except Exception as e:
            print(f"[WARN] Failed to fetch tile ({x},{y}): {e}")
            return None

    def detect_water_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Create binary water mask from satellite image using HSV color detection.

        Returns mask where 255=water, 0=land
        """
        if not CV2_AVAILABLE:
            # Fallback: use blue channel
            blue = image[:, :, 2]
            return (blue > 100).astype(np.uint8) * 255

        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Water is typically blue/cyan
        lower = np.array([
            self.config.water_hue_min,
            self.config.water_sat_min,
            self.config.water_val_min
        ])
        upper = np.array([
            self.config.water_hue_max,
            255,
            255
        ])

        water_mask = cv2.inRange(hsv, lower, upper)

        # Also detect dark water (deeper ocean)
        dark_lower = np.array([85, 10, 10])
        dark_upper = np.array([140, 255, 150])
        dark_water = cv2.inRange(hsv, dark_lower, dark_upper)

        water_mask = cv2.bitwise_or(water_mask, dark_water)

        # Clean up with morphology
        kernel = np.ones((3, 3), np.uint8)
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel)
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel)

        return water_mask

    def get_region_mosaic(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float
    ) -> Tuple[Optional[np.ndarray], TileBounds]:
        """
        Create a mosaic image covering the specified region.

        Returns the mosaic and its geographic bounds.
        """
        zoom = self.config.zoom_level

        # Get corner tiles
        x_min, y_max = self.lat_lon_to_tile(lat_min, lon_min, zoom)
        x_max, y_min = self.lat_lon_to_tile(lat_max, lon_max, zoom)

        # Expand slightly
        x_min -= 1
        x_max += 1
        y_min -= 1
        y_max += 1

        n_tiles_x = x_max - x_min + 1
        n_tiles_y = y_max - y_min + 1
        tile_size = self.config.tile_size

        print(f"[INFO] Fetching {n_tiles_x}x{n_tiles_y} = {n_tiles_x * n_tiles_y} tiles at zoom {zoom}...")

        # Create mosaic
        mosaic = np.zeros((n_tiles_y * tile_size, n_tiles_x * tile_size, 3), dtype=np.uint8)

        for ty in range(y_min, y_max + 1):
            for tx in range(x_min, x_max + 1):
                tile = self.fetch_tile(tx, ty, zoom)
                if tile is not None:
                    px = (tx - x_min) * tile_size
                    py = (ty - y_min) * tile_size
                    mosaic[py:py+tile_size, px:px+tile_size] = tile

        # Calculate bounds
        bounds = TileBounds(
            lat_min=self.tile_to_bounds(x_min, y_max, zoom).lat_min,
            lat_max=self.tile_to_bounds(x_min, y_min, zoom).lat_max,
            lon_min=self.tile_to_bounds(x_min, y_min, zoom).lon_min,
            lon_max=self.tile_to_bounds(x_max, y_min, zoom).lon_max,
        )

        return mosaic, bounds

    def lat_lon_to_pixel(
        self,
        lat: float,
        lon: float,
        bounds: TileBounds,
        image_shape: Tuple[int, int]
    ) -> Tuple[int, int]:
        """Convert lat/lon to pixel coordinates in the mosaic."""
        height, width = image_shape[:2]

        # Normalize to 0-1
        x_norm = (lon - bounds.lon_min) / (bounds.lon_max - bounds.lon_min)
        y_norm = 1 - (lat - bounds.lat_min) / (bounds.lat_max - bounds.lat_min)

        px = int(x_norm * width)
        py = int(y_norm * height)

        return px, py

    def check_point_is_water(
        self,
        water_mask: np.ndarray,
        px: int,
        py: int
    ) -> Tuple[bool, float]:
        """
        Check if a point is in water by examining surrounding pixels.

        Returns (is_water, water_fraction)
        """
        height, width = water_mask.shape
        radius = self.config.check_radius_pixels

        # Bounds check
        x_min = max(0, px - radius)
        x_max = min(width, px + radius + 1)
        y_min = max(0, py - radius)
        y_max = min(height, py + radius + 1)

        if x_min >= x_max or y_min >= y_max:
            return False, 0.0

        # Get region around point
        region = water_mask[y_min:y_max, x_min:x_max]

        # Calculate water fraction
        water_pixels = np.sum(region > 0)
        total_pixels = region.size
        water_fraction = water_pixels / total_pixels if total_pixels > 0 else 0

        is_water = water_fraction >= self.config.water_threshold

        return is_water, water_fraction

    def filter_points(
        self,
        points: List[Tuple[float, float]],
        water_mask: np.ndarray,
        bounds: TileBounds
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        Filter points, keeping only those in water.

        Returns (water_points, land_points)
        """
        water_points = []
        land_points = []

        for lat, lon in points:
            # Check bounds
            if not (bounds.lat_min <= lat <= bounds.lat_max and
                    bounds.lon_min <= lon <= bounds.lon_max):
                continue

            px, py = self.lat_lon_to_pixel(lat, lon, bounds, water_mask.shape)
            is_water, fraction = self.check_point_is_water(water_mask, px, py)

            if is_water:
                water_points.append((lat, lon))
            else:
                land_points.append((lat, lon))

        return water_points, land_points

    def save_debug_image(
        self,
        mosaic: np.ndarray,
        water_mask: np.ndarray,
        water_points: List[Tuple[float, float]],
        land_points: List[Tuple[float, float]],
        bounds: TileBounds,
        filename: str
    ):
        """Save debug image showing water detection and point classification."""
        if not PIL_AVAILABLE:
            return

        # Create composite image
        # Left: original, Right: water mask with points
        height, width = mosaic.shape[:2]

        # Convert water mask to RGB (blue = water)
        mask_rgb = np.zeros((height, width, 3), dtype=np.uint8)
        mask_rgb[:, :, 2] = water_mask  # Blue channel

        # Overlay on satellite
        alpha = 0.4
        overlay = (mosaic * (1 - alpha) + mask_rgb * alpha).astype(np.uint8)

        # Draw points
        if CV2_AVAILABLE:
            for lat, lon in water_points:
                px, py = self.lat_lon_to_pixel(lat, lon, bounds, mosaic.shape)
                cv2.circle(overlay, (px, py), 3, (0, 255, 0), -1)  # Green = water

            for lat, lon in land_points:
                px, py = self.lat_lon_to_pixel(lat, lon, bounds, mosaic.shape)
                cv2.circle(overlay, (px, py), 3, (255, 0, 0), -1)  # Red = land

        # Create side-by-side image
        combined = np.hstack([mosaic, overlay])

        # Save
        output_path = ROOT_DIR / self.config.debug_dir / filename
        Image.fromarray(combined).save(output_path)
        print(f"[OK] Debug image saved: {output_path}")

        return str(output_path)

    def filter_coastline(
        self,
        points: List[Tuple[float, float]],
        save_debug: bool = True
    ) -> List[Tuple[float, float]]:
        """
        Main entry: filter coastline points to keep only those near water.

        Args:
            points: List of (lat, lon) tuples
            save_debug: Whether to save debug visualization

        Returns:
            Filtered list of points that are on/near water
        """
        if not points:
            return []

        print(f"\n[INFO] Filtering {len(points)} coastline points...")

        # Get region bounds
        lats = [p[0] for p in points]
        lons = [p[1] for p in points]
        lat_min, lat_max = min(lats), max(lats)
        lon_min, lon_max = min(lons), max(lons)

        # Add margin
        margin = 0.05
        lat_min -= margin
        lat_max += margin
        lon_min -= margin
        lon_max += margin

        # Get satellite mosaic
        mosaic, bounds = self.get_region_mosaic(lat_min, lat_max, lon_min, lon_max)

        if mosaic is None:
            print("[WARN] Could not create mosaic, returning all points")
            return points

        print(f"[OK] Mosaic size: {mosaic.shape[1]}x{mosaic.shape[0]} pixels")

        # Create water mask
        print("[INFO] Detecting water...")
        water_mask = self.detect_water_mask(mosaic)

        water_percent = np.sum(water_mask > 0) / water_mask.size * 100
        print(f"[OK] Water detected: {water_percent:.1f}% of image")

        # Filter points
        print("[INFO] Filtering points...")
        water_points, land_points = self.filter_points(points, water_mask, bounds)

        print(f"[OK] Water points: {len(water_points)}")
        print(f"[OK] Land points (removed): {len(land_points)}")

        # Save debug image
        if save_debug and self.config.save_debug_images:
            self.save_debug_image(
                mosaic, water_mask,
                water_points, land_points,
                bounds, "coastline_filter_debug.png"
            )

        return water_points


def filter_geojson_coastline(
    input_path: str,
    output_path: str,
    config: Optional[FilterConfig] = None
) -> dict:
    """
    Filter a GeoJSON coastline file to remove land points.

    Returns statistics about the filtering.
    """
    # Load input
    with open(input_path) as f:
        data = json.load(f)

    # Extract points
    all_points = []
    geom = data['features'][0]['geometry']

    if geom['type'] == 'LineString':
        for lon, lat in geom['coordinates']:
            all_points.append((lat, lon))
    elif geom['type'] == 'MultiLineString':
        for line in geom['coordinates']:
            for lon, lat in line:
                all_points.append((lat, lon))

    print(f"Loaded {len(all_points)} points from {input_path}")

    # Filter
    filter_obj = CoastlineFilter(config)
    filtered_points = filter_obj.filter_coastline(all_points)

    # Save output
    coordinates = [[lon, lat] for lat, lon in filtered_points]

    geojson = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coordinates
            },
            "properties": {
                "name": "Filtered Coastline",
                "source": "satellite_water_filter",
                "original_points": len(all_points),
                "filtered_points": len(filtered_points),
                "removed_points": len(all_points) - len(filtered_points)
            }
        }]
    }

    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)

    print(f"Saved {len(filtered_points)} points to {output_path}")

    return {
        "original": len(all_points),
        "filtered": len(filtered_points),
        "removed": len(all_points) - len(filtered_points)
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Filter coastline using water detection")
    parser.add_argument("input", help="Input GeoJSON file")
    parser.add_argument("-o", "--output", help="Output GeoJSON file")
    parser.add_argument("--zoom", type=int, default=13, help="Tile zoom level")
    parser.add_argument("--threshold", type=float, default=0.3, help="Water threshold")

    args = parser.parse_args()

    config = FilterConfig(
        zoom_level=args.zoom,
        water_threshold=args.threshold
    )

    output = args.output or args.input.replace('.geojson', '_filtered.geojson')
    filter_geojson_coastline(args.input, output, config)
