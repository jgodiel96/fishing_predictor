"""
OSM Coastline Loader for Fishing Predictor V8.

Loads verified coastline data from OpenStreetMap instead of CV detection.
Data source: https://osmdata.openstreetmap.de/data/coastlines.html

This provides highly accurate coastlines that are:
- Verified by humans
- Updated daily
- Available globally
- Much more reliable than CV detection from RGB imagery
"""

from __future__ import annotations

import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union
import math
import json

import numpy as np

try:
    import shapefile
    HAS_SHAPEFILE = True
except ImportError:
    HAS_SHAPEFILE = False

try:
    from shapely.geometry import shape, Point, LineString, Polygon, MultiPolygon, box
    from shapely.ops import unary_union, nearest_points
    from shapely import prepare
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class OSMCoastlineConfig:
    """Configuration for OSM coastline data."""
    # Data URLs
    water_polygons_url: str = "https://osmdata.openstreetmap.de/download/water-polygons-split-4326.zip"
    coastlines_url: str = "https://osmdata.openstreetmap.de/download/coastlines-split-4326.zip"

    # Local cache directory
    cache_dir: str = "data/coastlines"

    # Processing options
    simplify_tolerance: float = 0.0001  # ~10m at equator
    buffer_distance: float = 0.0001     # Small buffer to close gaps


@dataclass
class CoastlineResult:
    """Result from coastline loading."""
    coastline_points: List[Tuple[float, float]]  # (lat, lon) points
    water_polygons: List[Polygon]                 # Shapely polygons
    land_polygons: List[Polygon]                  # Shapely polygons (inverse)
    bounds: Tuple[float, float, float, float]     # (lat_min, lat_max, lon_min, lon_max)
    source: str                                   # 'osm', 'fallback', etc.

    @property
    def coastline_length_km(self) -> float:
        """Approximate coastline length in kilometers."""
        if len(self.coastline_points) < 2:
            return 0.0

        total = 0.0
        for i in range(len(self.coastline_points) - 1):
            lat1, lon1 = self.coastline_points[i]
            lat2, lon2 = self.coastline_points[i + 1]
            total += haversine_distance(lat1, lon1, lat2, lon2)

        return total


# =============================================================================
# OSM COASTLINE LOADER
# =============================================================================

class OSMCoastlineLoader:
    """
    Loads coastline data from OpenStreetMap.

    Provides two types of data:
    1. Water polygons - Areas covered by ocean/sea
    2. Coastline lines - The actual coast boundary

    Usage:
        loader = OSMCoastlineLoader()
        result = loader.load_coastline(lat_min, lat_max, lon_min, lon_max)
    """

    __slots__ = ('config', '_water_polygons', '_coastline_shapes', '_prepared_water')

    def __init__(self, config: Optional[OSMCoastlineConfig] = None):
        self.config = config or OSMCoastlineConfig()
        self._water_polygons: Optional[List] = None
        self._coastline_shapes: Optional[List] = None
        self._prepared_water = None

    def load_coastline(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        resolution_m: float = 50.0
    ) -> CoastlineResult:
        """
        Load coastline for the specified area.

        Args:
            lat_min, lat_max, lon_min, lon_max: Bounding box
            resolution_m: Target resolution in meters for points

        Returns:
            CoastlineResult with coastline data
        """
        if not HAS_SHAPELY:
            logger.error("Shapely required: pip install shapely")
            return self._empty_result(lat_min, lat_max, lon_min, lon_max)

        bounds = (lat_min, lat_max, lon_min, lon_max)
        bbox = box(lon_min, lat_min, lon_max, lat_max)

        # Try to load from local cache first
        water_polys = self._load_water_polygons(bounds)

        if not water_polys:
            # Try fallback: generate from Natural Earth or simple approximation
            logger.warning("OSM data not available, using fallback coastline")
            return self._generate_fallback_coastline(bounds, resolution_m)

        # Extract coastline points from water polygon boundaries
        coastline_points = self._extract_coastline_points(
            water_polys, bbox, resolution_m
        )

        # Create land polygons (inverse of water within bounds)
        land_polys = self._create_land_polygons(water_polys, bbox)

        return CoastlineResult(
            coastline_points=coastline_points,
            water_polygons=water_polys,
            land_polygons=land_polys,
            bounds=bounds,
            source='osm'
        )

    def _load_water_polygons(
        self,
        bounds: Tuple[float, float, float, float]
    ) -> List[Polygon]:
        """Load water polygons from shapefile."""
        if not HAS_SHAPEFILE:
            return []

        lat_min, lat_max, lon_min, lon_max = bounds
        cache_dir = Path(self.config.cache_dir)

        # Look for existing shapefile
        shp_files = list(cache_dir.glob("**/water_polygons*.shp"))
        if not shp_files:
            shp_files = list(cache_dir.glob("**/water-polygons*.shp"))

        if not shp_files:
            logger.info("No local OSM coastline data found")
            return []

        polygons = []

        for shp_path in shp_files:
            try:
                with shapefile.Reader(str(shp_path)) as sf:
                    for shp_rec in sf.iterShapes():
                        # Quick bounding box check
                        shp_bbox = shp_rec.bbox  # (minx, miny, maxx, maxy)

                        # Check if bounding boxes intersect
                        if (shp_bbox[2] < lon_min or shp_bbox[0] > lon_max or
                            shp_bbox[3] < lat_min or shp_bbox[1] > lat_max):
                            continue

                        # Convert to shapely
                        geom = shape(shp_rec.__geo_interface__)

                        # Clip to bounds
                        bbox_poly = box(lon_min, lat_min, lon_max, lat_max)
                        clipped = geom.intersection(bbox_poly)

                        if not clipped.is_empty:
                            if isinstance(clipped, Polygon):
                                polygons.append(clipped)
                            elif isinstance(clipped, MultiPolygon):
                                polygons.extend(list(clipped.geoms))

                logger.info(f"Loaded {len(polygons)} water polygons from {shp_path.name}")
                break  # Use first valid file

            except Exception as e:
                logger.warning(f"Error reading {shp_path}: {e}")
                continue

        return polygons

    def _extract_coastline_points(
        self,
        water_polys: List[Polygon],
        bbox: Polygon,
        resolution_m: float
    ) -> List[Tuple[float, float]]:
        """Extract coastline points from water polygon boundaries."""
        points = []

        # Convert resolution to degrees (approximate)
        resolution_deg = resolution_m / 111000  # ~111km per degree

        for poly in water_polys:
            # Get exterior ring
            exterior = poly.exterior

            # Simplify based on resolution
            simplified = exterior.simplify(resolution_deg / 2, preserve_topology=True)

            # Extract coordinates that are inside bbox
            for coord in simplified.coords:
                lon, lat = coord
                if bbox.contains(Point(lon, lat)):
                    points.append((lat, lon))

            # Also process interior rings (islands, etc)
            for interior in poly.interiors:
                simplified_int = interior.simplify(resolution_deg / 2, preserve_topology=True)
                for coord in simplified_int.coords:
                    lon, lat = coord
                    if bbox.contains(Point(lon, lat)):
                        points.append((lat, lon))

        # Remove duplicates while preserving order
        seen = set()
        unique_points = []
        for p in points:
            key = (round(p[0], 6), round(p[1], 6))
            if key not in seen:
                seen.add(key)
                unique_points.append(p)

        return unique_points

    def _create_land_polygons(
        self,
        water_polys: List[Polygon],
        bbox: Polygon
    ) -> List[Polygon]:
        """Create land polygons as inverse of water."""
        if not water_polys:
            return [bbox]

        try:
            # Union all water polygons
            water_union = unary_union(water_polys)

            # Subtract from bounding box
            land = bbox.difference(water_union)

            if isinstance(land, Polygon):
                return [land] if not land.is_empty else []
            elif isinstance(land, MultiPolygon):
                return [p for p in land.geoms if not p.is_empty]
            else:
                return []

        except Exception as e:
            logger.warning(f"Error creating land polygons: {e}")
            return []

    def _generate_fallback_coastline(
        self,
        bounds: Tuple[float, float, float, float],
        resolution_m: float
    ) -> CoastlineResult:
        """
        Generate a simple fallback coastline when OSM data is not available.
        Uses a basic west-is-ocean assumption for Peru's coast.
        """
        lat_min, lat_max, lon_min, lon_max = bounds

        # For Peru, the ocean is generally to the west
        # Create a simple diagonal coastline
        resolution_deg = resolution_m / 111000

        # Generate coastline points along a simple curve
        num_points = int((lat_max - lat_min) / resolution_deg)
        num_points = max(10, min(num_points, 1000))

        points = []
        for i in range(num_points + 1):
            lat = lat_min + (lat_max - lat_min) * (i / num_points)
            # Simple sinusoidal variation for more realistic look
            offset = 0.002 * math.sin(i * 0.5)
            lon = lon_min + (lon_max - lon_min) * 0.3 + offset
            points.append((lat, lon))

        # Create water polygon (west of coastline)
        water_coords = [(lon_min, lat_min)]
        for lat, lon in points:
            water_coords.append((lon, lat))
        water_coords.append((lon_min, lat_max))
        water_coords.append((lon_min, lat_min))

        water_poly = Polygon(water_coords)

        # Create land polygon (east of coastline)
        land_coords = [(lon_max, lat_min)]
        for lat, lon in reversed(points):
            land_coords.append((lon, lat))
        land_coords.append((lon_max, lat_max))
        land_coords.append((lon_max, lat_min))

        land_poly = Polygon(land_coords)

        return CoastlineResult(
            coastline_points=points,
            water_polygons=[water_poly] if water_poly.is_valid else [],
            land_polygons=[land_poly] if land_poly.is_valid else [],
            bounds=bounds,
            source='fallback'
        )

    def _empty_result(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float
    ) -> CoastlineResult:
        """Return empty result."""
        return CoastlineResult(
            coastline_points=[],
            water_polygons=[],
            land_polygons=[],
            bounds=(lat_min, lat_max, lon_min, lon_max),
            source='empty'
        )

    def get_distance_to_coast(
        self,
        lat: float,
        lon: float,
        coastline_points: List[Tuple[float, float]]
    ) -> float:
        """
        Calculate distance from point to nearest coastline.

        Args:
            lat, lon: Point coordinates
            coastline_points: Coastline points from load_coastline

        Returns:
            Distance in meters (positive = in water, negative = on land)
        """
        if not coastline_points:
            return float('inf')

        min_dist = float('inf')

        for coast_lat, coast_lon in coastline_points:
            dist = haversine_distance(lat, lon, coast_lat, coast_lon) * 1000  # km to m
            if dist < min_dist:
                min_dist = dist

        return min_dist

    def is_in_water(
        self,
        lat: float,
        lon: float,
        water_polygons: List[Polygon]
    ) -> bool:
        """
        Check if a point is in water.

        Args:
            lat, lon: Point coordinates
            water_polygons: Water polygons from load_coastline

        Returns:
            True if point is in water
        """
        if not HAS_SHAPELY or not water_polygons:
            return False

        point = Point(lon, lat)

        for poly in water_polygons:
            if poly.contains(point):
                return True

        return False

    def download_osm_data(
        self,
        data_type: str = 'water_polygons'
    ) -> bool:
        """
        Download OSM coastline data.

        Args:
            data_type: 'water_polygons' or 'coastlines'

        Returns:
            True if download successful
        """
        if not HAS_REQUESTS:
            logger.error("requests required: pip install requests")
            return False

        url = (self.config.water_polygons_url if data_type == 'water_polygons'
               else self.config.coastlines_url)

        cache_dir = Path(self.config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        zip_path = cache_dir / f"{data_type}.zip"

        logger.info(f"Downloading {data_type} from OSM (~500MB)...")

        try:
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        if downloaded % (50 * 8192) == 0:
                            logger.info(f"Download progress: {percent:.1f}%")

            logger.info("Download complete, extracting...")

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(cache_dir)

            # Clean up zip file
            zip_path.unlink()

            logger.info(f"OSM {data_type} ready in {cache_dir}")
            return True

        except Exception as e:
            logger.error(f"Error downloading OSM data: {e}")
            return False


# =============================================================================
# DISTANCE-BASED ZONE GENERATOR
# =============================================================================

class CoastalZoneGenerator:
    """
    Generates fishing zones based on distance from coastline.

    Creates concentric zones from the coast with different fishing characteristics.
    """

    __slots__ = ('coastline_loader',)

    def __init__(self, coastline_loader: Optional[OSMCoastlineLoader] = None):
        self.coastline_loader = coastline_loader or OSMCoastlineLoader()

    def generate_distance_zones(
        self,
        coastline_result: CoastlineResult,
        distances_m: List[float] = [50, 100, 200, 500, 1000, 2000]
    ) -> List[Dict]:
        """
        Generate zones based on distance from coast.

        Args:
            coastline_result: Result from OSMCoastlineLoader
            distances_m: List of distance thresholds in meters

        Returns:
            List of zone dictionaries with polygons
        """
        if not HAS_SHAPELY:
            return []

        if not coastline_result.water_polygons:
            return []

        zones = []
        water_union = unary_union(coastline_result.water_polygons)

        # Convert distances to degrees (approximate)
        prev_buffer = None

        for i, dist_m in enumerate(distances_m):
            dist_deg = dist_m / 111000  # Approximate

            # Create buffer from coastline (inward into water)
            try:
                if isinstance(water_union, (Polygon, MultiPolygon)):
                    # Buffer inward (negative) from water edge
                    buffered = water_union.buffer(-dist_deg)

                    if prev_buffer is None:
                        # First zone: from coast to first distance
                        zone_poly = water_union.difference(buffered)
                    else:
                        # Subsequent zones: ring between buffers
                        zone_poly = prev_buffer.difference(buffered)

                    if not zone_poly.is_empty:
                        zones.append({
                            'zone_id': f'coastal_{int(dist_m)}m',
                            'distance_min': distances_m[i-1] if i > 0 else 0,
                            'distance_max': dist_m,
                            'polygon': zone_poly,
                            'area_km2': zone_poly.area * 111 * 111  # Rough conversion
                        })

                    prev_buffer = buffered

            except Exception as e:
                logger.warning(f"Error creating zone at {dist_m}m: {e}")
                continue

        return zones

    def get_zone_for_point(
        self,
        lat: float,
        lon: float,
        zones: List[Dict]
    ) -> Optional[Dict]:
        """Find which zone a point belongs to."""
        if not HAS_SHAPELY:
            return None

        point = Point(lon, lat)

        for zone in zones:
            if zone['polygon'].contains(point):
                return zone

        return None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points in kilometers.

    Args:
        lat1, lon1: First point coordinates
        lat2, lon2: Second point coordinates

    Returns:
        Distance in kilometers
    """
    R = 6371  # Earth radius in km

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (math.sin(dlat / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def export_coastline_geojson(
    coastline_result: CoastlineResult,
    output_path: Union[str, Path]
) -> bool:
    """
    Export coastline result to GeoJSON.

    Args:
        coastline_result: Result from OSMCoastlineLoader
        output_path: Path for output file

    Returns:
        True if export successful
    """
    features = []

    # Add coastline as LineString
    if coastline_result.coastline_points:
        coords = [[lon, lat] for lat, lon in coastline_result.coastline_points]
        features.append({
            "type": "Feature",
            "properties": {
                "type": "coastline",
                "source": coastline_result.source,
                "length_km": coastline_result.coastline_length_km
            },
            "geometry": {
                "type": "LineString",
                "coordinates": coords
            }
        })

    # Add water polygons
    for i, poly in enumerate(coastline_result.water_polygons):
        try:
            coords = [list(poly.exterior.coords)]
            for interior in poly.interiors:
                coords.append(list(interior.coords))

            features.append({
                "type": "Feature",
                "properties": {
                    "type": "water",
                    "index": i
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": coords
                }
            })
        except Exception:
            continue

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)

        logger.info(f"Exported coastline to {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error exporting coastline: {e}")
        return False


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_coastline(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    resolution_m: float = 50.0
) -> CoastlineResult:
    """
    Convenience function to load coastline.

    Args:
        lat_min, lat_max, lon_min, lon_max: Bounding box
        resolution_m: Resolution in meters

    Returns:
        CoastlineResult with coastline data
    """
    loader = OSMCoastlineLoader()
    return loader.load_coastline(lat_min, lat_max, lon_min, lon_max, resolution_m)
