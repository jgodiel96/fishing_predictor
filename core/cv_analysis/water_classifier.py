"""
Water/Land Classifier for Fishing Predictor V8.

Uses OSM water polygons to accurately classify areas as water or land.
This approach does not depend on geographic assumptions (east/west).

The classifier can:
1. Use local OSM water polygon shapefiles if available
2. Query Overpass API for water features in the area
3. Provide accurate water/land classification for any location
"""

from __future__ import annotations

import logging
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union
import time

try:
    from shapely.geometry import shape, Point, Polygon, MultiPolygon, box
    from shapely.ops import unary_union
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import shapefile
    HAS_SHAPEFILE = True
except ImportError:
    HAS_SHAPEFILE = False

logger = logging.getLogger(__name__)


@dataclass
class WaterClassification:
    """Result of water/land classification for a point or area."""
    is_water: bool
    water_type: Optional[str] = None  # 'ocean', 'sea', 'bay', etc.
    confidence: float = 1.0
    source: str = 'unknown'  # 'osm_polygon', 'overpass', 'coastline'


class WaterClassifier:
    """
    Classifies areas as water or land using OSM data.

    Does NOT rely on geographic assumptions like "west = ocean".
    Uses actual OSM water polygon data for accurate classification.
    """

    __slots__ = ('cache_dir', '_water_polygons', '_land_polygons', '_bbox', '_source')

    def __init__(self, cache_dir: str = "data/coastlines"):
        self.cache_dir = Path(cache_dir)
        self._water_polygons: List[Polygon] = []
        self._land_polygons: List[Polygon] = []
        self._bbox: Optional[Tuple[float, float, float, float]] = None
        self._source: str = 'none'

    def load_area(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        use_overpass: bool = True
    ) -> bool:
        """
        Load water/land data for the specified area.

        Args:
            lat_min, lat_max, lon_min, lon_max: Bounding box
            use_overpass: Whether to query Overpass API if local data not available

        Returns:
            True if data was loaded successfully
        """
        self._bbox = (lat_min, lat_max, lon_min, lon_max)

        # Try local water polygons first
        if self._load_local_water_polygons(lat_min, lat_max, lon_min, lon_max):
            logger.info(f"Loaded {len(self._water_polygons)} water polygons from local files")
            return True

        # Try Overpass API
        if use_overpass and HAS_REQUESTS:
            if self._load_from_overpass(lat_min, lat_max, lon_min, lon_max):
                logger.info(f"Loaded {len(self._water_polygons)} water polygons from Overpass")
                return True

        logger.warning("No water polygon data available")
        return False

    def _load_local_water_polygons(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float
    ) -> bool:
        """Load water polygons from local OSM shapefiles."""
        if not HAS_SHAPEFILE or not HAS_SHAPELY:
            return False

        # Look for water polygon shapefiles
        shp_patterns = [
            "**/water_polygons*.shp",
            "**/water-polygons*.shp",
            "water-polygons-split-4326/**/*.shp"
        ]

        shp_files = []
        for pattern in shp_patterns:
            shp_files.extend(self.cache_dir.glob(pattern))

        if not shp_files:
            return False

        bbox_poly = box(lon_min, lat_min, lon_max, lat_max)
        polygons = []

        for shp_path in shp_files:
            try:
                logger.info(f"Loading water polygons from {shp_path}")
                with shapefile.Reader(str(shp_path)) as sf:
                    for shp_rec in sf.iterShapes():
                        shp_bbox = shp_rec.bbox

                        # Quick bbox check
                        if (shp_bbox[2] < lon_min or shp_bbox[0] > lon_max or
                            shp_bbox[3] < lat_min or shp_bbox[1] > lat_max):
                            continue

                        geom = shape(shp_rec.__geo_interface__)
                        clipped = geom.intersection(bbox_poly)

                        if not clipped.is_empty:
                            if isinstance(clipped, Polygon):
                                polygons.append(clipped)
                            elif isinstance(clipped, MultiPolygon):
                                polygons.extend(list(clipped.geoms))

                if polygons:
                    self._water_polygons = polygons
                    self._source = 'osm_polygon'
                    return True

            except Exception as e:
                logger.warning(f"Error reading {shp_path}: {e}")
                continue

        return False

    def _load_from_overpass(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float
    ) -> bool:
        """
        Load water features from Overpass API.

        Queries for:
        - natural=water (lakes, ponds, reservoirs)
        - natural=coastline (ocean boundaries)
        - waterway=* (rivers, streams)
        - place=sea/ocean
        """
        if not HAS_REQUESTS or not HAS_SHAPELY:
            return False

        # Overpass QL query for water features
        query = f"""
        [out:json][timeout:60];
        (
          // Ocean and sea areas
          way["natural"="coastline"]({lat_min},{lon_min},{lat_max},{lon_max});
          relation["natural"="coastline"]({lat_min},{lon_min},{lat_max},{lon_max});

          // Water bodies
          way["natural"="water"]({lat_min},{lon_min},{lat_max},{lon_max});
          relation["natural"="water"]({lat_min},{lon_min},{lat_max},{lon_max});

          // Seas and oceans (if tagged as areas)
          way["place"~"sea|ocean"]({lat_min},{lon_min},{lat_max},{lon_max});
          relation["place"~"sea|ocean"]({lat_min},{lon_min},{lat_max},{lon_max});
        );
        out body;
        >;
        out skel qt;
        """

        overpass_url = "https://overpass-api.de/api/interpreter"

        try:
            logger.info("Querying Overpass API for water features...")
            response = requests.post(
                overpass_url,
                data={'data': query},
                timeout=120
            )
            response.raise_for_status()
            data = response.json()

            # Parse OSM elements into polygons
            polygons = self._parse_overpass_response(data, lat_min, lat_max, lon_min, lon_max)

            if polygons:
                self._water_polygons = polygons
                self._source = 'overpass'
                return True

            # If no explicit water polygons found, try coastline-based approach
            # For coastal areas, the ocean is typically outside the coastline
            coastlines = self._extract_coastlines_from_overpass(data)
            if coastlines:
                water_poly = self._coastlines_to_water_smart(
                    coastlines, lat_min, lat_max, lon_min, lon_max
                )
                if water_poly:
                    self._water_polygons = water_poly
                    self._source = 'overpass_coastline'
                    return True

        except requests.exceptions.Timeout:
            logger.warning("Overpass API timeout")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Overpass API error: {e}")
        except Exception as e:
            logger.warning(f"Error parsing Overpass response: {e}")

        return False

    def _parse_overpass_response(
        self,
        data: Dict,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float
    ) -> List[Polygon]:
        """Parse Overpass JSON response into Shapely polygons."""
        if not HAS_SHAPELY:
            return []

        polygons = []

        # Build node lookup
        nodes = {}
        for elem in data.get('elements', []):
            if elem['type'] == 'node':
                nodes[elem['id']] = (elem['lon'], elem['lat'])

        # Process ways
        for elem in data.get('elements', []):
            if elem['type'] == 'way':
                tags = elem.get('tags', {})

                # Check if it's a water feature
                if tags.get('natural') == 'water' or tags.get('place') in ('sea', 'ocean'):
                    coords = []
                    for node_id in elem.get('nodes', []):
                        if node_id in nodes:
                            coords.append(nodes[node_id])

                    if len(coords) >= 4 and coords[0] == coords[-1]:
                        try:
                            poly = Polygon(coords)
                            if poly.is_valid:
                                polygons.append(poly)
                        except Exception:
                            pass

        return polygons

    def _extract_coastlines_from_overpass(self, data: Dict) -> List:
        """Extract coastline segments from Overpass response."""
        if not HAS_SHAPELY:
            return []

        from shapely.geometry import LineString

        lines = []
        nodes = {}

        for elem in data.get('elements', []):
            if elem['type'] == 'node':
                nodes[elem['id']] = (elem['lon'], elem['lat'])

        for elem in data.get('elements', []):
            if elem['type'] == 'way':
                tags = elem.get('tags', {})
                if tags.get('natural') == 'coastline':
                    coords = []
                    for node_id in elem.get('nodes', []):
                        if node_id in nodes:
                            coords.append(nodes[node_id])

                    if len(coords) >= 2:
                        try:
                            line = LineString(coords)
                            if line.is_valid:
                                lines.append(line)
                        except Exception:
                            pass

        return lines

    def _coastlines_to_water_smart(
        self,
        coastlines: List,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float
    ) -> List[Polygon]:
        """
        Convert coastlines to water polygon using OSM convention.

        OSM Convention: Coastlines are drawn with water on the RIGHT side
        when following the line direction. This is a global standard that
        doesn't depend on geographic assumptions.
        """
        if not HAS_SHAPELY or not coastlines:
            return []

        from shapely.ops import polygonize, linemerge

        try:
            # Merge all coastline segments
            merged = linemerge(coastlines)

            # Create bbox boundary
            bbox = box(lon_min, lat_min, lon_max, lat_max)
            bbox_boundary = bbox.boundary

            # Combine coastlines with bbox boundary
            if hasattr(merged, 'geoms'):
                all_lines = list(merged.geoms) + [bbox_boundary]
            else:
                all_lines = [merged, bbox_boundary]

            # Polygonize
            polygons = list(polygonize(all_lines))

            if not polygons:
                return []

            # Determine which polygons are water using sampling
            # Sample points and use heuristics to identify water
            water_polygons = []

            for poly in polygons:
                if poly.is_empty:
                    continue

                # Use centroid to test
                centroid = poly.centroid

                # Check if this polygon is likely water
                # Water polygons typically touch the bbox edge on the ocean side
                # and have coastline on the land side

                # For now, use a simple heuristic:
                # if the polygon touches the western or southern edge more,
                # it's likely water (this works for Peru but we need better logic)

                # Better approach: check which side of coastline the centroid is
                is_water = self._is_point_on_water_side(centroid.x, centroid.y, coastlines)

                if is_water:
                    water_polygons.append(poly)

            return water_polygons

        except Exception as e:
            logger.warning(f"Error converting coastlines to water: {e}")
            return []

    def _is_point_on_water_side(
        self,
        lon: float,
        lat: float,
        coastlines: List
    ) -> bool:
        """
        Determine if a point is on the water side of coastlines.

        Uses OSM convention: water is on the RIGHT side of coastline direction.
        """
        if not coastlines:
            return False

        # Find nearest coastline segment
        point = Point(lon, lat)
        min_dist = float('inf')
        nearest_line = None

        for line in coastlines:
            dist = point.distance(line)
            if dist < min_dist:
                min_dist = dist
                nearest_line = line

        if nearest_line is None:
            return False

        # Get the nearest point on the coastline
        from shapely.ops import nearest_points
        _, nearest_on_line = nearest_points(point, nearest_line)

        # Find the direction of the coastline at this point
        coords = list(nearest_line.coords)

        # Find which segment contains the nearest point
        for i in range(len(coords) - 1):
            seg_start = coords[i]
            seg_end = coords[i + 1]

            seg_line = Point(seg_start).distance(Point(seg_end))
            d1 = Point(seg_start).distance(nearest_on_line)
            d2 = Point(seg_end).distance(nearest_on_line)

            if abs(d1 + d2 - seg_line) < 0.0001:  # Point is on this segment
                # Calculate direction vector of segment
                dx = seg_end[0] - seg_start[0]
                dy = seg_end[1] - seg_start[1]

                # Calculate vector from line to point
                px = lon - nearest_on_line.x
                py = lat - nearest_on_line.y

                # Cross product: positive = left side, negative = right side
                # OSM convention: water is on RIGHT (negative cross product)
                cross = dx * py - dy * px

                return cross < 0  # Water is on right side

        return False

    def is_water(self, lat: float, lon: float) -> WaterClassification:
        """
        Check if a point is in water.

        Args:
            lat, lon: Point coordinates

        Returns:
            WaterClassification with result
        """
        if not HAS_SHAPELY:
            return WaterClassification(is_water=False, confidence=0, source='no_shapely')

        if not self._water_polygons:
            return WaterClassification(is_water=False, confidence=0, source='no_data')

        point = Point(lon, lat)

        for poly in self._water_polygons:
            if poly.contains(point):
                return WaterClassification(
                    is_water=True,
                    water_type='ocean',  # Could be refined based on tags
                    confidence=1.0,
                    source=self._source
                )

        return WaterClassification(
            is_water=False,
            confidence=1.0,
            source=self._source
        )

    def get_water_polygon(self) -> Optional[Polygon]:
        """Get the combined water polygon for the loaded area."""
        if not self._water_polygons:
            return None

        if len(self._water_polygons) == 1:
            return self._water_polygons[0]

        try:
            return unary_union(self._water_polygons)
        except Exception:
            return self._water_polygons[0]

    def get_water_polygons(self) -> List[Polygon]:
        """Get all water polygons for the loaded area."""
        return self._water_polygons.copy()

    def classify_substrate_potential(
        self,
        lat: float,
        lon: float,
        depth_m: float,
        distance_to_coast_m: float
    ) -> Dict[str, float]:
        """
        Estimate substrate type probabilities based on environmental factors.

        This is a simplified model. For accurate substrate data, use:
        - Bathymetric surveys
        - Sediment maps
        - GEBCO substrate data

        Returns:
            Dict with substrate type probabilities
        """
        # Simplified substrate model based on depth and distance
        # Real implementation would use actual substrate data

        substrates = {
            'rock': 0.0,
            'sand': 0.0,
            'mud': 0.0,
            'gravel': 0.0,
            'mixed': 0.0
        }

        # Very shallow/near shore: often rocky
        if distance_to_coast_m < 100:
            substrates['rock'] = 0.6
            substrates['sand'] = 0.2
            substrates['mixed'] = 0.2
        # Moderate distance: mixed
        elif distance_to_coast_m < 500:
            substrates['rock'] = 0.3
            substrates['sand'] = 0.4
            substrates['mixed'] = 0.3
        # Further out: often sandy/muddy
        else:
            substrates['sand'] = 0.5
            substrates['mud'] = 0.3
            substrates['mixed'] = 0.2

        # Depth adjustments
        if abs(depth_m) > 30:
            # Deep areas often muddy
            substrates['mud'] += 0.2
            substrates['rock'] = max(0, substrates['rock'] - 0.1)

        # Normalize
        total = sum(substrates.values())
        if total > 0:
            for k in substrates:
                substrates[k] /= total

        return substrates


def create_water_classifier(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    cache_dir: str = "data/coastlines"
) -> WaterClassifier:
    """
    Create and initialize a water classifier for the given area.

    Args:
        lat_min, lat_max, lon_min, lon_max: Bounding box
        cache_dir: Directory for cached data

    Returns:
        Initialized WaterClassifier
    """
    classifier = WaterClassifier(cache_dir)
    classifier.load_area(lat_min, lat_max, lon_min, lon_max)
    return classifier
