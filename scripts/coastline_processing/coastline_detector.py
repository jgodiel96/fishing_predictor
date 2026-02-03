#!/usr/bin/env python3
"""
Coastline Detector using Image Recognition

Detects the precise coastline (intersection of land and sea) by:
1. Fetching satellite imagery tiles for the region
2. Using color-based segmentation to identify water vs land
3. Finding the boundary line between water and land
4. Creating a fine-grained polyline along the coast

This provides much more accurate coastline data than OSM for fishing analysis.
"""

import os
import sys
import json
import math
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import requests
from io import BytesIO

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("[WARN] PIL not available. Install with: pip install Pillow")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("[WARN] OpenCV not available. Install with: pip install opencv-python")


@dataclass
class TileCoord:
    """Tile coordinates for map tiles."""
    x: int
    y: int
    zoom: int


class CoastlineDetector:
    """
    Detects coastline from satellite imagery using image processing.

    Uses color segmentation to distinguish water (blue) from land (brown/green),
    then traces the boundary to create a precise coastline polyline.
    """

    # OpenStreetMap tile server (free, no API key needed)
    OSM_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"

    # Satellite imagery from ESRI (free for limited use)
    ESRI_SATELLITE_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

    # Water color ranges in HSV (Hue, Saturation, Value)
    # Water is typically blue with varying saturation
    WATER_HSV_LOW = np.array([90, 30, 30])    # Lower bound (dark blue/cyan)
    WATER_HSV_HIGH = np.array([130, 255, 255])  # Upper bound (bright blue)

    # Alternative: darker water detection
    WATER_DARK_LOW = np.array([85, 20, 10])
    WATER_DARK_HIGH = np.array([135, 255, 180])

    def __init__(self, cache_dir: str = None):
        """Initialize the coastline detector."""
        self.cache_dir = Path(cache_dir) if cache_dir else ROOT_DIR / "data" / "cache" / "tiles"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def lat_lon_to_tile(self, lat: float, lon: float, zoom: int) -> TileCoord:
        """Convert latitude/longitude to tile coordinates."""
        n = 2 ** zoom
        x = int((lon + 180) / 360 * n)
        lat_rad = math.radians(lat)
        y = int((1 - math.asinh(math.tan(lat_rad)) / math.pi) / 2 * n)
        return TileCoord(x=x, y=y, zoom=zoom)

    def tile_to_lat_lon(self, tile: TileCoord) -> Tuple[float, float, float, float]:
        """Convert tile coordinates to bounding box (lat_min, lon_min, lat_max, lon_max)."""
        n = 2 ** tile.zoom
        lon_min = tile.x / n * 360 - 180
        lon_max = (tile.x + 1) / n * 360 - 180
        lat_max = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * tile.y / n))))
        lat_min = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (tile.y + 1) / n))))
        return (lat_min, lon_min, lat_max, lon_max)

    def fetch_tile(self, tile: TileCoord, tile_type: str = "satellite") -> Optional[np.ndarray]:
        """
        Fetch a map tile image.

        Args:
            tile: Tile coordinates
            tile_type: "satellite" or "street"

        Returns:
            Image as numpy array (BGR format for OpenCV)
        """
        if not PIL_AVAILABLE:
            return None

        cache_file = self.cache_dir / f"{tile_type}_{tile.zoom}_{tile.x}_{tile.y}.png"

        # Check cache first
        if cache_file.exists():
            img = Image.open(cache_file)
            return np.array(img)

        # Fetch from server
        if tile_type == "satellite":
            url = self.ESRI_SATELLITE_URL.format(z=tile.zoom, x=tile.x, y=tile.y)
        else:
            url = self.OSM_TILE_URL.format(z=tile.zoom, x=tile.x, y=tile.y)

        try:
            headers = {'User-Agent': 'FishingPredictor/1.0 (coastline detection)'}
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            img = Image.open(BytesIO(response.content))

            # Save to cache
            img.save(cache_file)

            return np.array(img)

        except Exception as e:
            print(f"[WARN] Failed to fetch tile {tile}: {e}")
            return None

    def detect_water_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Create a binary mask where water pixels are white (255) and land is black (0).

        Uses HSV color space for robust water detection.
        """
        if not CV2_AVAILABLE:
            # Fallback: simple blue channel threshold
            if len(image.shape) == 3:
                blue = image[:, :, 2]  # Assuming RGB
                return (blue > 100).astype(np.uint8) * 255
            return np.zeros(image.shape[:2], dtype=np.uint8)

        # Convert to HSV
        if len(image.shape) == 2:
            return np.zeros(image.shape, dtype=np.uint8)

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Create water mask using color range
        mask1 = cv2.inRange(hsv, self.WATER_HSV_LOW, self.WATER_HSV_HIGH)
        mask2 = cv2.inRange(hsv, self.WATER_DARK_LOW, self.WATER_DARK_HIGH)

        # Combine masks
        water_mask = cv2.bitwise_or(mask1, mask2)

        # Clean up with morphological operations
        kernel = np.ones((5, 5), np.uint8)
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel)
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel)

        return water_mask

    def find_coastline_contour(self, water_mask: np.ndarray) -> List[Tuple[int, int]]:
        """
        Find the coastline contour from a water mask.

        Returns list of (x, y) pixel coordinates along the coastline.
        """
        if not CV2_AVAILABLE:
            # Simple fallback: find edges by row scanning
            contour = []
            for y in range(water_mask.shape[0]):
                for x in range(1, water_mask.shape[1]):
                    if water_mask[y, x] != water_mask[y, x-1]:
                        contour.append((x, y))
            return contour

        # Find contours
        contours, _ = cv2.findContours(
            water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        if not contours:
            return []

        # Get the largest contour (main water body)
        largest = max(contours, key=cv2.contourArea)

        # Convert to list of tuples
        points = [(int(p[0][0]), int(p[0][1])) for p in largest]

        return points

    def pixel_to_latlon(
        self,
        pixel_x: int,
        pixel_y: int,
        tile: TileCoord,
        tile_size: int = 256
    ) -> Tuple[float, float]:
        """Convert pixel coordinates within a tile to lat/lon."""
        lat_min, lon_min, lat_max, lon_max = self.tile_to_lat_lon(tile)

        # Interpolate within tile
        lon = lon_min + (pixel_x / tile_size) * (lon_max - lon_min)
        lat = lat_max - (pixel_y / tile_size) * (lat_max - lat_min)  # Y is inverted

        return (lat, lon)

    def detect_coastline_for_region(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        zoom: int = 14,
        spacing_m: float = 100
    ) -> List[Tuple[float, float]]:
        """
        Detect coastline for a geographic region.

        Args:
            lat_min, lat_max, lon_min, lon_max: Bounding box
            zoom: Tile zoom level (14-16 recommended for detail)
            spacing_m: Target spacing between coastline points in meters

        Returns:
            List of (lat, lon) tuples along the coastline
        """
        print(f"[INFO] Detectando linea costera para region:")
        print(f"       Lat: {lat_min:.4f} a {lat_max:.4f}")
        print(f"       Lon: {lon_min:.4f} a {lon_max:.4f}")
        print(f"       Zoom: {zoom}, Espaciado: {spacing_m}m")

        # Get corner tiles
        tile_nw = self.lat_lon_to_tile(lat_max, lon_min, zoom)
        tile_se = self.lat_lon_to_tile(lat_min, lon_max, zoom)

        all_coastline_points = []

        # Iterate over all tiles in region
        total_tiles = (tile_se.x - tile_nw.x + 1) * (tile_se.y - tile_nw.y + 1)
        processed = 0

        for tx in range(tile_nw.x, tile_se.x + 1):
            for ty in range(tile_nw.y, tile_se.y + 1):
                tile = TileCoord(x=tx, y=ty, zoom=zoom)
                processed += 1

                # Fetch satellite image
                img = self.fetch_tile(tile, "satellite")
                if img is None:
                    continue

                # Detect water
                water_mask = self.detect_water_mask(img)

                # Find coastline contour
                contour = self.find_coastline_contour(water_mask)

                # Convert to lat/lon
                for px, py in contour:
                    lat, lon = self.pixel_to_latlon(px, py, tile)

                    # Check if within our region
                    if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                        all_coastline_points.append((lat, lon))

                if processed % 10 == 0:
                    print(f"       Procesados {processed}/{total_tiles} tiles...")

        print(f"[OK] {len(all_coastline_points)} puntos detectados")

        # Remove duplicates and sort
        coastline = self._clean_and_sort_points(all_coastline_points, spacing_m)

        print(f"[OK] {len(coastline)} puntos despues de limpieza")

        return coastline

    def _clean_and_sort_points(
        self,
        points: List[Tuple[float, float]],
        min_spacing_m: float
    ) -> List[Tuple[float, float]]:
        """Remove duplicates and ensure minimum spacing between points."""
        if not points:
            return []

        # Sort by latitude (south to north)
        points = sorted(set(points), key=lambda p: (p[0], p[1]))

        # Remove points that are too close together
        cleaned = [points[0]]
        for lat, lon in points[1:]:
            last_lat, last_lon = cleaned[-1]

            # Calculate approximate distance
            dlat = (lat - last_lat) * 111000  # meters
            dlon = (lon - last_lon) * 111000 * math.cos(math.radians(lat))
            dist = math.sqrt(dlat**2 + dlon**2)

            if dist >= min_spacing_m * 0.5:  # Allow some tolerance
                cleaned.append((lat, lon))

        return cleaned

    def save_as_geojson(
        self,
        coastline: List[Tuple[float, float]],
        output_path: str
    ) -> str:
        """Save coastline as GeoJSON file."""
        # Convert to GeoJSON format (lon, lat order)
        coordinates = [[lon, lat] for lat, lon in coastline]

        geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": coordinates
                },
                "properties": {
                    "name": "Detected Coastline",
                    "source": "satellite_imagery",
                    "points": len(coordinates)
                }
            }]
        }

        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)

        print(f"[OK] Guardado: {output_path}")
        return output_path


def detect_coastline_for_canepa():
    """Detect precise coastline for Playa Canepa area."""
    detector = CoastlineDetector()

    # Playa Canepa region (focused area)
    coastline = detector.detect_coastline_for_region(
        lat_min=-18.10,
        lat_max=-17.95,
        lon_min=-70.40,
        lon_max=-70.20,
        zoom=15,  # High detail
        spacing_m=50  # 50m spacing for fine detail
    )

    if coastline:
        output_path = ROOT_DIR / "data" / "cache" / "coastline_canepa_detected.geojson"
        detector.save_as_geojson(coastline, str(output_path))
        return coastline

    return []


def detect_full_coastline():
    """Detect coastline for the full Tacna-Ilo region."""
    detector = CoastlineDetector()

    from data.data_config import DataConfig
    region = DataConfig.REGION

    coastline = detector.detect_coastline_for_region(
        lat_min=region['lat_min'],
        lat_max=region['lat_max'],
        lon_min=region['lon_min'],
        lon_max=region['lon_max'],
        zoom=14,  # Medium detail for large area
        spacing_m=100  # 100m spacing
    )

    if coastline:
        output_path = ROOT_DIR / "data" / "cache" / "coastline_detected.geojson"
        detector.save_as_geojson(coastline, str(output_path))
        return coastline

    return []


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect coastline from satellite imagery")
    parser.add_argument("--area", choices=["canepa", "full"], default="canepa",
                       help="Area to detect: canepa (focused) or full (entire region)")
    parser.add_argument("--zoom", type=int, default=15,
                       help="Tile zoom level (14-17)")
    parser.add_argument("--spacing", type=float, default=50,
                       help="Target spacing between points in meters")

    args = parser.parse_args()

    if args.area == "canepa":
        detect_coastline_for_canepa()
    else:
        detect_full_coastline()
