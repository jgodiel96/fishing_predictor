#!/usr/bin/env python3
"""
SAM-based Coastline Detector

Uses Meta's Segment Anything Model (SAM) for precise coastline detection.
SAM provides zero-shot segmentation with state-of-the-art accuracy.

Hardware Requirements:
- 16GB+ RAM (18GB available on target MacBook Pro M3 Pro)
- Metal-compatible GPU for MPS acceleration
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

# Import connector for intelligent point connection
try:
    from core.coastline_connector import CoastlineConnector, ConnectorConfig
    CONNECTOR_AVAILABLE = True
except ImportError:
    CONNECTOR_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# SAM imports (optional - graceful degradation)
SAM_AVAILABLE = False
try:
    import torch
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    pass


@dataclass
class TileCoord:
    """Tile coordinates for map tiles."""
    x: int
    y: int
    zoom: int


@dataclass
class SAMConfig:
    """Configuration for SAM model."""
    model_type: str = "vit_h"
    checkpoint_path: str = "models/sam/sam_vit_h_4b8939.pth"
    device: str = "auto"  # "mps", "cuda", "cpu", or "auto"
    image_size: int = 1024
    mask_threshold: float = 0.5
    min_area_pixels: int = 500


class SAMCoastlineDetector:
    """
    Coastline detector using Segment Anything Model (SAM).

    SAM is a zero-shot segmentation model developed by Meta AI Research.
    It can segment any object with high precision without task-specific training.
    """

    # Tile servers
    ESRI_SATELLITE_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    OSM_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"

    def __init__(self, config: SAMConfig = None, cache_dir: str = None):
        """
        Initialize SAM coastline detector.

        Args:
            config: SAM configuration
            cache_dir: Directory for caching tiles
        """
        self.config = config or SAMConfig()
        self.cache_dir = Path(cache_dir) if cache_dir else ROOT_DIR / "data" / "cache" / "tiles"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # SAM model (lazy loading)
        self._sam = None
        self._predictor = None
        self._device = None

    def _init_sam(self):
        """Initialize SAM model (lazy loading to save memory)."""
        if self._sam is not None:
            return True

        if not SAM_AVAILABLE:
            print("[WARN] SAM not available. Install with:")
            print("       pip install segment-anything torch")
            print("       Download checkpoint from: https://github.com/facebookresearch/segment-anything")
            return False

        checkpoint = ROOT_DIR / self.config.checkpoint_path
        if not checkpoint.exists():
            print(f"[WARN] SAM checkpoint not found: {checkpoint}")
            print("       Download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
            return False

        # Determine device
        if self.config.device == "auto":
            if torch.backends.mps.is_available():
                self._device = torch.device("mps")
                print("[INFO] Using MPS (Metal Performance Shaders) for acceleration")
            elif torch.cuda.is_available():
                self._device = torch.device("cuda")
                print("[INFO] Using CUDA for acceleration")
            else:
                self._device = torch.device("cpu")
                print("[INFO] Using CPU (no GPU acceleration)")
        else:
            self._device = torch.device(self.config.device)

        print(f"[INFO] Loading SAM model ({self.config.model_type})...")
        self._sam = sam_model_registry[self.config.model_type](checkpoint=str(checkpoint))
        self._sam.to(self._device)
        self._predictor = SamPredictor(self._sam)
        print("[OK] SAM model loaded")

        return True

    def lat_lon_to_tile(self, lat: float, lon: float, zoom: int) -> TileCoord:
        """Convert latitude/longitude to tile coordinates."""
        n = 2 ** zoom
        x = int((lon + 180) / 360 * n)
        lat_rad = math.radians(lat)
        y = int((1 - math.asinh(math.tan(lat_rad)) / math.pi) / 2 * n)
        return TileCoord(x=x, y=y, zoom=zoom)

    def tile_to_bounds(self, tile: TileCoord) -> Tuple[float, float, float, float]:
        """Convert tile to bounding box (lat_min, lon_min, lat_max, lon_max)."""
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
            Image as numpy array (RGB format)
        """
        if not PIL_AVAILABLE:
            return None

        cache_file = self.cache_dir / f"{tile_type}_{tile.zoom}_{tile.x}_{tile.y}.png"

        # Check cache
        if cache_file.exists():
            img = Image.open(cache_file).convert("RGB")
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

            img = Image.open(BytesIO(response.content)).convert("RGB")
            img.save(cache_file)

            return np.array(img)
        except Exception as e:
            print(f"[WARN] Failed to fetch tile {tile}: {e}")
            return None

    def detect_water_seeds(self, image: np.ndarray) -> np.ndarray:
        """
        Detect seed points in water using color analysis.
        These seeds help SAM understand what to segment.

        Args:
            image: RGB image array

        Returns:
            Array of seed point coordinates [[x, y], ...]
        """
        if not CV2_AVAILABLE:
            # Fallback: use west side of image (ocean side in Peru)
            h, w = image.shape[:2]
            return np.array([[w // 4, h // 2]])

        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Detect blue (water)
        water_mask = cv2.inRange(hsv, (90, 30, 30), (130, 255, 255))

        # Also detect darker water
        dark_water = cv2.inRange(hsv, (85, 20, 10), (135, 255, 180))
        water_mask = cv2.bitwise_or(water_mask, dark_water)

        # Find contours
        contours, _ = cv2.findContours(water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            # Fallback: use west side
            h, w = image.shape[:2]
            return np.array([[w // 4, h // 2]])

        # Get centroid of largest water region
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)

        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return np.array([[cx, cy]])

        return np.array([[image.shape[1] // 4, image.shape[0] // 2]])

    def segment_water_sam(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Segment water in an image using SAM.

        Args:
            image: RGB image array

        Returns:
            Binary mask where 1=water, 0=land (or None if SAM unavailable)
        """
        if not self._init_sam():
            return None

        # Set image for SAM
        self._predictor.set_image(image)

        # Get seed points
        seed_points = self.detect_water_seeds(image)

        # Generate mask using SAM
        masks, scores, _ = self._predictor.predict(
            point_coords=seed_points,
            point_labels=np.ones(len(seed_points)),  # All positive (water)
            multimask_output=False
        )

        # Get the mask
        water_mask = masks[0].astype(np.uint8)

        # Clean up with morphology
        if CV2_AVAILABLE:
            kernel = np.ones((5, 5), np.uint8)
            water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel)
            water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel)

        return water_mask

    def segment_water_hsv(self, image: np.ndarray) -> np.ndarray:
        """
        Fallback: Segment water using HSV color detection.
        Used when SAM is not available.

        Args:
            image: RGB image array

        Returns:
            Binary mask where 255=water, 0=land
        """
        if not CV2_AVAILABLE:
            if len(image.shape) == 3:
                blue = image[:, :, 2]
                return (blue > 100).astype(np.uint8) * 255
            return np.zeros(image.shape[:2], dtype=np.uint8)

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Water color ranges
        mask1 = cv2.inRange(hsv, (90, 30, 30), (130, 255, 255))
        mask2 = cv2.inRange(hsv, (85, 20, 10), (135, 255, 180))
        water_mask = cv2.bitwise_or(mask1, mask2)

        # Clean up
        kernel = np.ones((5, 5), np.uint8)
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel)
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel)

        return water_mask

    def segment_water(self, image: np.ndarray, use_sam: bool = True) -> np.ndarray:
        """
        Segment water in an image.

        Args:
            image: RGB image array
            use_sam: Whether to try SAM first

        Returns:
            Binary mask where 1/255=water, 0=land
        """
        if use_sam:
            mask = self.segment_water_sam(image)
            if mask is not None:
                return mask

        return self.segment_water_hsv(image)

    def extract_coastline_contour(self, water_mask: np.ndarray) -> List[Tuple[int, int]]:
        """
        Extract coastline contour from water mask.

        Args:
            water_mask: Binary mask (water=1/255, land=0)

        Returns:
            List of (x, y) pixel coordinates along coastline
        """
        if not CV2_AVAILABLE:
            # Simple fallback
            contour = []
            for y in range(water_mask.shape[0]):
                for x in range(1, water_mask.shape[1]):
                    if water_mask[y, x] != water_mask[y, x-1]:
                        contour.append((x, y))
            return contour

        # Ensure mask is uint8
        if water_mask.max() <= 1:
            water_mask = (water_mask * 255).astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(
            water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        if not contours:
            return []

        # Get the longest contour (main coastline)
        longest = max(contours, key=len)

        return [(int(p[0][0]), int(p[0][1])) for p in longest]

    def pixel_to_latlon(
        self,
        pixel_x: int,
        pixel_y: int,
        tile: TileCoord,
        tile_size: int = 256
    ) -> Tuple[float, float]:
        """Convert pixel coordinates within a tile to lat/lon."""
        lat_min, lon_min, lat_max, lon_max = self.tile_to_bounds(tile)

        lon = lon_min + (pixel_x / tile_size) * (lon_max - lon_min)
        lat = lat_max - (pixel_y / tile_size) * (lat_max - lat_min)

        return (lat, lon)

    def detect_coastline_for_tile(
        self,
        tile: TileCoord,
        use_sam: bool = True
    ) -> Tuple[List[Tuple[float, float]], np.ndarray]:
        """
        Detect coastline for a single tile.

        Args:
            tile: Tile coordinates
            use_sam: Whether to use SAM

        Returns:
            Tuple of (coastline points as lat/lon, water mask)
        """
        # Fetch satellite image
        image = self.fetch_tile(tile, "satellite")
        if image is None:
            return [], None

        # Segment water
        water_mask = self.segment_water(image, use_sam=use_sam)

        # Extract contour
        contour = self.extract_coastline_contour(water_mask)

        # Convert to lat/lon
        points = []
        for px, py in contour:
            lat, lon = self.pixel_to_latlon(px, py, tile)
            points.append((lat, lon))

        return points, water_mask

    def detect_coastline_for_region(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        zoom: int = 15,
        use_sam: bool = True,
        min_spacing_m: float = 50
    ) -> List[Tuple[float, float]]:
        """
        Detect coastline for a geographic region.

        Args:
            lat_min, lat_max, lon_min, lon_max: Bounding box
            zoom: Tile zoom level (15-16 recommended for precision)
            use_sam: Whether to use SAM (falls back to HSV if unavailable)
            min_spacing_m: Minimum spacing between points in meters

        Returns:
            List of (lat, lon) tuples along the coastline
        """
        print(f"[INFO] Detectando linea costera con {'SAM' if use_sam else 'HSV'}:")
        print(f"       Lat: {lat_min:.4f} a {lat_max:.4f}")
        print(f"       Lon: {lon_min:.4f} a {lon_max:.4f}")
        print(f"       Zoom: {zoom}")

        # Get corner tiles
        tile_nw = self.lat_lon_to_tile(lat_max, lon_min, zoom)
        tile_se = self.lat_lon_to_tile(lat_min, lon_max, zoom)

        all_points = []
        total_tiles = (tile_se.x - tile_nw.x + 1) * (tile_se.y - tile_nw.y + 1)
        processed = 0

        for tx in range(tile_nw.x, tile_se.x + 1):
            for ty in range(tile_nw.y, tile_se.y + 1):
                tile = TileCoord(x=tx, y=ty, zoom=zoom)
                processed += 1

                points, _ = self.detect_coastline_for_tile(tile, use_sam=use_sam)

                # Filter to region
                for lat, lon in points:
                    if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                        all_points.append((lat, lon))

                if processed % 5 == 0:
                    print(f"       Procesados {processed}/{total_tiles} tiles...")

        print(f"[OK] {len(all_points)} puntos detectados")

        # Clean and sort points
        coastline = self._clean_points(all_points, min_spacing_m)
        print(f"[OK] {len(coastline)} puntos despues de limpieza")

        return coastline

    def _clean_points(
        self,
        points: List[Tuple[float, float]],
        min_spacing_m: float
    ) -> List[Tuple[float, float]]:
        """
        Clean and connect points using intelligent algorithm.

        This replaces the old simple latitude-based sorting that caused
        false connections crossing land.
        """
        if not points:
            return []

        # Remove exact duplicates
        points = list(set(points))

        if len(points) < 3:
            return sorted(points, key=lambda p: (p[0], p[1]))

        # Use intelligent connector if available
        if CONNECTOR_AVAILABLE:
            print("[INFO] Using intelligent point connector...")
            config = ConnectorConfig(
                max_gap_m=500,
                max_merge_distance_m=1000,
                min_segment_points=3,
                remove_isolated_points=True,
                min_neighbors_radius_m=min_spacing_m * 4,
                min_neighbors_count=2
            )
            connector = CoastlineConnector(config)
            segments = connector.connect(points)

            # Flatten segments into single list for backward compatibility
            # Each segment is properly ordered
            cleaned = []
            for seg in segments:
                cleaned.extend(seg)

            return cleaned
        else:
            print("[WARN] Connector not available, using legacy method")
            # Fallback to old method (not recommended)
            points = sorted(points, key=lambda p: (p[0], p[1]))

            # Filter by spacing
            cleaned = [points[0]]
            for lat, lon in points[1:]:
                last_lat, last_lon = cleaned[-1]

                dlat = (lat - last_lat) * 111000
                dlon = (lon - last_lon) * 111000 * math.cos(math.radians(lat))
                dist = math.sqrt(dlat**2 + dlon**2)

                if dist >= min_spacing_m * 0.5:
                    cleaned.append((lat, lon))

            return cleaned

    def detect_coastline_segments(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        zoom: int = 15,
        use_sam: bool = True,
        min_spacing_m: float = 50
    ) -> List[List[Tuple[float, float]]]:
        """
        Detect coastline and return as separate segments.

        This is the preferred method as it preserves segment boundaries
        and avoids false connections between distant parts of the coast.

        Returns:
            List of segments, each segment is a list of (lat, lon) points
        """
        print(f"[INFO] Detecting coastline segments with {'SAM' if use_sam else 'HSV'}:")
        print(f"       Lat: {lat_min:.4f} to {lat_max:.4f}")
        print(f"       Lon: {lon_min:.4f} to {lon_max:.4f}")
        print(f"       Zoom: {zoom}")

        # Get corner tiles
        tile_nw = self.lat_lon_to_tile(lat_max, lon_min, zoom)
        tile_se = self.lat_lon_to_tile(lat_min, lon_max, zoom)

        all_points = []
        total_tiles = (tile_se.x - tile_nw.x + 1) * (tile_se.y - tile_nw.y + 1)
        processed = 0

        for tx in range(tile_nw.x, tile_se.x + 1):
            for ty in range(tile_nw.y, tile_se.y + 1):
                tile = TileCoord(x=tx, y=ty, zoom=zoom)
                processed += 1

                points, _ = self.detect_coastline_for_tile(tile, use_sam=use_sam)

                # Filter to region
                for lat, lon in points:
                    if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                        all_points.append((lat, lon))

                if processed % 5 == 0:
                    print(f"       Processed {processed}/{total_tiles} tiles...")

        print(f"[OK] {len(all_points)} points detected")

        if not all_points:
            return []

        # Remove duplicates
        all_points = list(set(all_points))

        # Use connector to build segments
        if CONNECTOR_AVAILABLE:
            config = ConnectorConfig(
                max_gap_m=500,
                max_merge_distance_m=1000,
                min_segment_points=3,
                remove_isolated_points=True,
                min_neighbors_radius_m=min_spacing_m * 4,
            )
            connector = CoastlineConnector(config)
            segments = connector.connect(all_points)
            return segments
        else:
            # Fallback: return as single segment
            sorted_points = sorted(all_points, key=lambda p: (p[0], p[1]))
            return [sorted_points]

    def save_as_geojson(
        self,
        coastline: List[Tuple[float, float]],
        output_path: str,
        metadata: Dict = None
    ) -> str:
        """Save coastline as GeoJSON file (single LineString)."""
        coordinates = [[lon, lat] for lat, lon in coastline]

        properties = {
            "name": "SAM Detected Coastline",
            "source": "segment_anything_model",
            "points": len(coordinates),
            "model": self.config.model_type
        }
        if metadata:
            properties.update(metadata)

        geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": coordinates
                },
                "properties": properties
            }]
        }

        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)

        print(f"[OK] Saved: {output_path}")
        return output_path

    def save_segments_as_geojson(
        self,
        segments: List[List[Tuple[float, float]]],
        output_path: str,
        metadata: Dict = None
    ) -> str:
        """
        Save coastline segments as GeoJSON MultiLineString.

        This preserves segment boundaries to avoid visual artifacts
        from false connections.
        """
        # Convert to GeoJSON coordinates (lon, lat)
        coordinates = []
        for seg in segments:
            seg_coords = [[lon, lat] for lat, lon in seg]
            coordinates.append(seg_coords)

        total_points = sum(len(seg) for seg in segments)

        properties = {
            "name": "SAM Detected Coastline (Segmented)",
            "source": "segment_anything_model",
            "num_segments": len(segments),
            "total_points": total_points,
            "model": self.config.model_type
        }
        if metadata:
            properties.update(metadata)

        geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {
                    "type": "MultiLineString",
                    "coordinates": coordinates
                },
                "properties": properties
            }]
        }

        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)

        print(f"[OK] Saved: {output_path} ({len(segments)} segments, {total_points} points)")
        return output_path


def check_sam_availability() -> Dict[str, bool]:
    """Check SAM dependencies availability."""
    status = {
        "pil": PIL_AVAILABLE,
        "opencv": CV2_AVAILABLE,
        "torch": False,
        "segment_anything": False,
        "mps": False,
        "cuda": False,
        "checkpoint": False
    }

    try:
        import torch
        status["torch"] = True
        status["mps"] = torch.backends.mps.is_available()
        status["cuda"] = torch.cuda.is_available()
    except ImportError:
        pass

    try:
        from segment_anything import sam_model_registry
        status["segment_anything"] = True
    except ImportError:
        pass

    checkpoint = ROOT_DIR / "models/sam/sam_vit_h_4b8939.pth"
    status["checkpoint"] = checkpoint.exists()

    return status


def print_sam_status():
    """Print SAM availability status."""
    status = check_sam_availability()

    print("\n=== SAM Status ===")
    print(f"PIL (Pillow):        {'OK' if status['pil'] else 'MISSING'}")
    print(f"OpenCV:              {'OK' if status['opencv'] else 'MISSING'}")
    print(f"PyTorch:             {'OK' if status['torch'] else 'MISSING'}")
    print(f"segment-anything:    {'OK' if status['segment_anything'] else 'MISSING'}")
    print(f"MPS (Metal):         {'OK' if status['mps'] else 'N/A'}")
    print(f"CUDA:                {'OK' if status['cuda'] else 'N/A'}")
    print(f"SAM Checkpoint:      {'OK' if status['checkpoint'] else 'MISSING'}")

    if not status['checkpoint']:
        print("\nTo download SAM checkpoint:")
        print("  mkdir -p models/sam")
        print("  wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O models/sam/sam_vit_h_4b8939.pth")

    print()
    return status


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SAM-based coastline detection")
    parser.add_argument("--check", action="store_true", help="Check SAM availability")
    parser.add_argument("--area", choices=["canepa", "full"], default="canepa",
                       help="Area to detect")
    parser.add_argument("--zoom", type=int, default=15, help="Tile zoom level")
    parser.add_argument("--no-sam", action="store_true", help="Use HSV instead of SAM")

    args = parser.parse_args()

    if args.check:
        print_sam_status()
        sys.exit(0)

    detector = SAMCoastlineDetector()

    if args.area == "canepa":
        coastline = detector.detect_coastline_for_region(
            lat_min=-18.10,
            lat_max=-17.95,
            lon_min=-70.40,
            lon_max=-70.20,
            zoom=args.zoom,
            use_sam=not args.no_sam
        )
        output_path = ROOT_DIR / "data" / "cache" / "coastline_sam_canepa.geojson"
    else:
        from data.data_config import DataConfig
        region = DataConfig.REGION

        coastline = detector.detect_coastline_for_region(
            lat_min=region['lat_min'],
            lat_max=region['lat_max'],
            lon_min=region['lon_min'],
            lon_max=region['lon_max'],
            zoom=args.zoom,
            use_sam=not args.no_sam
        )
        output_path = ROOT_DIR / "data" / "cache" / "coastline_sam_full.geojson"

    if coastline:
        detector.save_as_geojson(coastline, str(output_path))
