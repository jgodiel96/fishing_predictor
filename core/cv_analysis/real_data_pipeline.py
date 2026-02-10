"""
Real Data Pipeline for Fishing Predictor V8.

Generates fishing zones using verified data sources instead of CV detection:
- OSM Coastlines: Accurate coastline from OpenStreetMap
- GEBCO Bathymetry: Global bathymetry data (~450m resolution)
- Species Habitat Matrix: Scientific data on fish preferences

This approach is more reliable than CV detection from RGB imagery because:
1. OSM coastlines are verified by humans
2. GEBCO provides actual measured/interpolated depth data
3. No dependency on image quality, lighting, or spectral bands
"""

from __future__ import annotations

import logging
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Union
import math

import numpy as np

try:
    from shapely.geometry import Point, Polygon, MultiPolygon, box, mapping
    from shapely.ops import unary_union
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False

from .osm_coastline import OSMCoastlineLoader, CoastlineResult, haversine_distance
from .bathymetry import GEBCOBathymetry, BathymetryResult
from .species_zones import (
    SpeciesZone, SpeciesHabitat, SPECIES_DATABASE,
    DepthZone, SubstrateType
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class RealDataConfig:
    """Configuration for real data pipeline."""
    # Data paths
    gebco_file: Optional[str] = None
    osm_cache_dir: str = "data/coastlines"

    # Zone generation parameters
    distance_bands_m: Tuple[float, ...] = (0, 50, 100, 200, 500, 1000, 2000)
    depth_bands_m: Tuple[float, ...] = (0, 2, 5, 10, 20, 30, 50, 100)

    # Grid resolution
    grid_resolution_deg: float = 0.0005  # ~50m

    # Substrate estimation (based on typical Peru coast characteristics)
    rocky_coast_fraction: float = 0.4  # 40% rocky, 60% sandy typical


@dataclass
class RealDataZone:
    """A fishing zone generated from real data."""
    zone_id: str
    polygon: List[Tuple[float, float]]  # [(lat, lon), ...]
    center: Tuple[float, float]

    # Characteristics
    distance_from_coast_m: Tuple[float, float]  # (min, max)
    depth_range_m: Tuple[float, float]  # (min, max)
    avg_depth_m: float
    substrate: SubstrateType
    depth_zone: DepthZone

    # Species information
    species_scores: Dict[str, float]
    primary_species: str
    secondary_species: List[str]
    color: Tuple[int, int, int]

    # Metadata
    area_km2: float = 0.0
    data_source: str = "osm+gebco"

    def to_geojson_feature(self) -> Dict:
        """Convert to GeoJSON Feature."""
        coords = [[lon, lat] for lat, lon in self.polygon]
        if coords and coords[0] != coords[-1]:
            coords.append(coords[0])

        return {
            'type': 'Feature',
            'properties': {
                'zone_id': self.zone_id,
                'distance_from_coast_min': self.distance_from_coast_m[0],
                'distance_from_coast_max': self.distance_from_coast_m[1],
                'depth_min': self.depth_range_m[0],
                'depth_max': self.depth_range_m[1],
                'avg_depth': self.avg_depth_m,
                'substrate': self.substrate.value,
                'depth_zone': self.depth_zone.value,
                'primary_species': self.primary_species,
                'secondary_species': self.secondary_species,
                'species_scores': self.species_scores,
                'color': f'rgb({self.color[0]},{self.color[1]},{self.color[2]})',
                'area_km2': self.area_km2,
                'data_source': self.data_source
            },
            'geometry': {
                'type': 'Polygon',
                'coordinates': [coords]
            }
        }


@dataclass
class RealDataResult:
    """Complete result from real data pipeline."""
    coastline: CoastlineResult
    zones: List[RealDataZone]
    depth_grid: Optional[np.ndarray]
    depth_lats: Optional[np.ndarray]
    depth_lons: Optional[np.ndarray]
    bounds: Tuple[float, float, float, float]
    processing_info: Dict = field(default_factory=dict)

    def to_geojson(self) -> Dict:
        """Export all data to GeoJSON."""
        features = []

        # Add coastline
        if self.coastline.coastline_points:
            coords = [[lon, lat] for lat, lon in self.coastline.coastline_points]
            features.append({
                'type': 'Feature',
                'properties': {
                    'type': 'coastline',
                    'source': self.coastline.source,
                    'length_km': self.coastline.coastline_length_km
                },
                'geometry': {
                    'type': 'LineString',
                    'coordinates': coords
                }
            })

        # Add zones
        for zone in self.zones:
            features.append(zone.to_geojson_feature())

        return {
            'type': 'FeatureCollection',
            'properties': {
                'bounds': self.bounds,
                'total_zones': len(self.zones),
                'processing_info': self.processing_info
            },
            'features': features
        }

    def get_species_summary(self) -> Dict[str, Dict]:
        """Get summary by species."""
        summary = {}

        for species_id, habitat in SPECIES_DATABASE.items():
            zones_for_species = [
                z for z in self.zones
                if z.primary_species == species_id or species_id in z.secondary_species
            ]

            total_area = sum(z.area_km2 for z in zones_for_species)
            primary_zones = [z for z in zones_for_species if z.primary_species == species_id]

            summary[species_id] = {
                'name': habitat.name_es,
                'total_zones': len(zones_for_species),
                'primary_zones': len(primary_zones),
                'total_area_km2': round(total_area, 4),
                'color': habitat.color
            }

        return summary


# =============================================================================
# REAL DATA PIPELINE
# =============================================================================

class RealDataPipeline:
    """
    Pipeline for generating fishing zones from real data.

    Uses:
    - OSM coastlines for accurate coast boundary
    - GEBCO bathymetry for depth information
    - Species habitat matrix for zone classification

    This replaces CV-based detection which is unreliable with RGB-only imagery.
    """

    __slots__ = ('config', 'coastline_loader', 'gebco', '_coastline_cache')

    def __init__(self, config: Optional[RealDataConfig] = None):
        self.config = config or RealDataConfig()
        self.coastline_loader = OSMCoastlineLoader()
        self.gebco = GEBCOBathymetry(self.config.gebco_file) if self.config.gebco_file else GEBCOBathymetry()
        self._coastline_cache: Optional[CoastlineResult] = None

    def analyze_area(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        use_gebco: bool = True,
        substrate_hint: Optional[str] = None
    ) -> RealDataResult:
        """
        Analyze an area and generate fishing zones.

        Args:
            lat_min, lat_max, lon_min, lon_max: Bounding box
            use_gebco: Whether to use GEBCO for depth (if available)
            substrate_hint: Override substrate type ('rock', 'sand', 'mixed')

        Returns:
            RealDataResult with all generated data
        """
        bounds = (lat_min, lat_max, lon_min, lon_max)
        logger.info(f"Analyzing area: {bounds}")

        processing_info = {
            'bounds': bounds,
            'use_gebco': use_gebco,
            'substrate_hint': substrate_hint
        }

        # 1. Load coastline from OSM
        coastline = self.coastline_loader.load_coastline(
            lat_min, lat_max, lon_min, lon_max,
            resolution_m=50.0
        )
        self._coastline_cache = coastline

        processing_info['coastline_source'] = coastline.source
        processing_info['coastline_points'] = len(coastline.coastline_points)

        # 2. Get depth data from GEBCO (real bathymetry data)
        depth_grid, depth_lats, depth_lons = None, None, None

        if use_gebco and self.gebco._data is not None:
            depth_grid, depth_lats, depth_lons = self.gebco.get_depth_grid(
                lat_min, lat_max, lon_min, lon_max,
                resolution=self.config.grid_resolution_deg
            )
            processing_info['depth_source'] = 'gebco'
        else:
            # No GEBCO data - zones will be based on distance from OSM coastline only
            processing_info['depth_source'] = 'distance_based'
            logger.info("GEBCO bathymetry not available - using distance-based zones from OSM coastline")

        # 3. Generate zones
        zones = self._generate_zones(
            coastline, depth_grid, depth_lats, depth_lons,
            bounds, substrate_hint
        )

        processing_info['zones_generated'] = len(zones)

        return RealDataResult(
            coastline=coastline,
            zones=zones,
            depth_grid=depth_grid,
            depth_lats=depth_lats,
            depth_lons=depth_lons,
            bounds=bounds,
            processing_info=processing_info
        )

    def _generate_synthetic_depth(
        self,
        coastline: CoastlineResult,
        bounds: Tuple[float, float, float, float]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate synthetic depth based on distance from coastline.

        Uses a simple model: depth increases linearly with distance from coast.
        """
        lat_min, lat_max, lon_min, lon_max = bounds
        resolution = self.config.grid_resolution_deg

        lats = np.arange(lat_min, lat_max, resolution)
        lons = np.arange(lon_min, lon_max, resolution)

        depth_grid = np.full((len(lats), len(lons)), np.nan, dtype=np.float32)

        if not coastline.coastline_points:
            return depth_grid, lats, lons

        # For each grid point, calculate distance to coast and estimate depth
        coast_points = np.array(coastline.coastline_points)

        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                # Check if in water
                if coastline.water_polygons:
                    point = Point(lon, lat)
                    in_water = any(poly.contains(point) for poly in coastline.water_polygons)
                else:
                    # Fallback: assume west is water for Peru
                    in_water = self._simple_water_check(lat, lon, coastline)

                if in_water:
                    # Calculate distance to nearest coast point
                    distances = np.array([
                        haversine_distance(lat, lon, cp[0], cp[1])
                        for cp in coast_points[:100]  # Limit for performance
                    ])
                    min_dist_km = np.min(distances)

                    # Simple depth model: ~1m depth per 50m distance, max 100m
                    # This is a rough approximation for Peru's coastal shelf
                    depth_m = min(min_dist_km * 20, 100)  # 20m per km
                    depth_grid[i, j] = -depth_m  # Negative = underwater

        return depth_grid, lats.astype(np.float32), lons.astype(np.float32)

    def _simple_water_check(
        self,
        lat: float,
        lon: float,
        coastline: CoastlineResult
    ) -> bool:
        """Simple check if point is likely in water (for fallback coastline)."""
        if not coastline.coastline_points:
            return False

        # For Peru, ocean is generally west of coast
        # Find closest coastline point and check if we're west of it
        closest_coast_lon = None
        min_lat_diff = float('inf')

        for coast_lat, coast_lon in coastline.coastline_points:
            lat_diff = abs(lat - coast_lat)
            if lat_diff < min_lat_diff:
                min_lat_diff = lat_diff
                closest_coast_lon = coast_lon

        if closest_coast_lon is not None:
            return lon < closest_coast_lon  # West = water

        return False

    def _generate_zones(
        self,
        coastline: CoastlineResult,
        depth_grid: np.ndarray,
        depth_lats: np.ndarray,
        depth_lons: np.ndarray,
        bounds: Tuple[float, float, float, float],
        substrate_hint: Optional[str]
    ) -> List[RealDataZone]:
        """Generate fishing zones based on coastline and depth."""
        if not HAS_SHAPELY:
            logger.error("Shapely required for zone generation")
            return []

        lat_min, lat_max, lon_min, lon_max = bounds
        zones = []

        # Create water polygon
        if coastline.water_polygons:
            water_union = unary_union(coastline.water_polygons)
        else:
            # Fallback: create simple water polygon west of coastline
            water_union = self._create_fallback_water_polygon(coastline, bounds)

        if water_union is None or water_union.is_empty:
            logger.warning("No water area found")
            return []

        # Generate zones by distance bands
        distance_bands = self.config.distance_bands_m
        zone_id = 0

        for i in range(len(distance_bands) - 1):
            dist_min = distance_bands[i]
            dist_max = distance_bands[i + 1]

            # Convert to degrees (approximate)
            dist_min_deg = dist_min / 111000
            dist_max_deg = dist_max / 111000

            try:
                # Create ring zone by buffering
                if dist_min == 0:
                    outer = water_union
                else:
                    outer = water_union.buffer(-dist_min_deg)

                inner = water_union.buffer(-dist_max_deg)

                if outer.is_empty:
                    continue

                zone_poly = outer.difference(inner) if not inner.is_empty else outer

                if zone_poly.is_empty:
                    continue

                # Handle MultiPolygon
                polygons = [zone_poly] if isinstance(zone_poly, Polygon) else list(zone_poly.geoms)

                for poly in polygons:
                    if poly.is_empty or poly.area < 1e-8:
                        continue

                    # Get depth statistics for this zone
                    avg_depth, depth_range = self._get_zone_depth_stats(
                        poly, depth_grid, depth_lats, depth_lons,
                        distance_min_m=dist_min, distance_max_m=dist_max
                    )

                    # Determine substrate
                    substrate = self._estimate_substrate(
                        dist_min, dist_max, substrate_hint
                    )

                    # Calculate species scores
                    species_scores = self._calculate_species_scores(
                        substrate, avg_depth
                    )

                    # Get primary and secondary species
                    sorted_species = sorted(
                        species_scores.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )

                    if not sorted_species:
                        continue

                    primary = sorted_species[0][0]
                    secondary = [s[0] for s in sorted_species[1:4] if s[1] > 0.5]

                    # Get color from primary species
                    color = SPECIES_DATABASE[primary].color

                    # Extract polygon coordinates
                    coords = [(lat, lon) for lon, lat in poly.exterior.coords]

                    # Calculate center
                    centroid = poly.centroid
                    center = (centroid.y, centroid.x)

                    # Calculate area
                    area_km2 = poly.area * 111 * 111  # Rough conversion

                    zone = RealDataZone(
                        zone_id=f"zone_{zone_id:04d}",
                        polygon=coords,
                        center=center,
                        distance_from_coast_m=(dist_min, dist_max),
                        depth_range_m=depth_range,
                        avg_depth_m=avg_depth,
                        substrate=substrate,
                        depth_zone=self._get_depth_zone(avg_depth),
                        species_scores=species_scores,
                        primary_species=primary,
                        secondary_species=secondary,
                        color=color,
                        area_km2=area_km2
                    )

                    zones.append(zone)
                    zone_id += 1

            except Exception as e:
                logger.warning(f"Error creating zone at {dist_min}-{dist_max}m: {e}")
                continue

        logger.info(f"Generated {len(zones)} fishing zones")
        return zones

    def _create_fallback_water_polygon(
        self,
        coastline: CoastlineResult,
        bounds: Tuple[float, float, float, float]
    ) -> Optional[Polygon]:
        """Create water polygon when OSM data is not available."""
        if not coastline.coastline_points:
            return None

        lat_min, lat_max, lon_min, lon_max = bounds

        # Create polygon from coastline points (west is water)
        coords = [(lon_min, lat_min)]

        for lat, lon in coastline.coastline_points:
            coords.append((lon, lat))

        coords.append((lon_min, lat_max))
        coords.append((lon_min, lat_min))

        try:
            poly = Polygon(coords)
            return poly if poly.is_valid else poly.buffer(0)
        except Exception:
            return None

    def _get_zone_depth_stats(
        self,
        poly: Polygon,
        depth_grid: Optional[np.ndarray],
        depth_lats: Optional[np.ndarray],
        depth_lons: Optional[np.ndarray],
        distance_min_m: float = 0,
        distance_max_m: float = 100
    ) -> Tuple[float, Tuple[float, float]]:
        """
        Get depth statistics for a zone polygon.

        If GEBCO data available: uses real bathymetry
        If not: estimates from distance using Peru shelf model (~20m depth per km from coast)
        """
        # Try GEBCO first (real bathymetry data)
        if depth_grid is not None and len(depth_grid) > 0 and depth_lats is not None:
            depths = []
            minx, miny, maxx, maxy = poly.bounds

            for lat in depth_lats:
                if lat < miny or lat > maxy:
                    continue

                lat_idx = np.argmin(np.abs(depth_lats - lat))

                for lon in depth_lons:
                    if lon < minx or lon > maxx:
                        continue

                    if poly.contains(Point(lon, lat)):
                        lon_idx = np.argmin(np.abs(depth_lons - lon))

                        if 0 <= lat_idx < depth_grid.shape[0] and 0 <= lon_idx < depth_grid.shape[1]:
                            d = depth_grid[lat_idx, lon_idx]
                            if not np.isnan(d):
                                depths.append(d)

            if depths:
                avg = float(np.mean(depths))
                return avg, (float(np.min(depths)), float(np.max(depths)))

        # No GEBCO - estimate depth from distance to coast
        # Peru coastal shelf: approximately 20m depth per 1km from shore
        # This is based on typical Peruvian shelf bathymetry
        avg_dist_km = (distance_min_m + distance_max_m) / 2000.0  # Convert to km
        estimated_depth = -avg_dist_km * 20  # Negative = underwater, 20m per km

        min_depth = -(distance_min_m / 1000.0) * 20
        max_depth = -(distance_max_m / 1000.0) * 20

        return estimated_depth, (max_depth, min_depth)  # Note: max_depth is more negative

    def _estimate_substrate(
        self,
        dist_min: float,
        dist_max: float,
        substrate_hint: Optional[str]
    ) -> SubstrateType:
        """
        Estimate substrate type based on distance from coast.

        General pattern for Peru coast:
        - Very close (0-100m): Often rocky
        - Medium (100-500m): Mixed
        - Far (>500m): Sandy bottom
        """
        if substrate_hint:
            mapping = {
                'rock': SubstrateType.ROCK,
                'rocky': SubstrateType.ROCK,
                'sand': SubstrateType.SAND,
                'sandy': SubstrateType.SAND,
                'mixed': SubstrateType.MIXED
            }
            return mapping.get(substrate_hint.lower(), SubstrateType.MIXED)

        avg_dist = (dist_min + dist_max) / 2

        if avg_dist < 100:
            return SubstrateType.ROCK
        elif avg_dist < 500:
            return SubstrateType.MIXED
        else:
            return SubstrateType.SAND

    def _calculate_species_scores(
        self,
        substrate: SubstrateType,
        depth: float
    ) -> Dict[str, float]:
        """Calculate affinity scores for all species."""
        scores = {}

        for species_id, habitat in SPECIES_DATABASE.items():
            score = habitat.get_affinity(substrate, depth)
            scores[species_id] = round(score, 3)

        return scores

    def _get_depth_zone(self, depth: float) -> DepthZone:
        """Classify depth into zone."""
        abs_d = abs(depth)

        if abs_d <= 2:
            return DepthZone.INTERTIDAL
        elif abs_d <= 10:
            return DepthZone.SHALLOW
        elif abs_d <= 30:
            return DepthZone.MODERATE
        elif abs_d <= 100:
            return DepthZone.DEEP
        else:
            return DepthZone.VERY_DEEP

    def get_depth_at_point(
        self,
        lat: float,
        lon: float
    ) -> Optional[BathymetryResult]:
        """Get depth at a specific point."""
        if self.gebco._data is not None:
            depth, conf = self.gebco.get_depth(lat, lon)
            if not np.isnan(depth):
                return BathymetryResult(
                    depth=depth,
                    source='gebco',
                    confidence=conf
                )

        # Fallback: estimate from distance to coast
        if self._coastline_cache and self._coastline_cache.coastline_points:
            dist_km = min(
                haversine_distance(lat, lon, cp[0], cp[1])
                for cp in self._coastline_cache.coastline_points[:100]
            )
            estimated_depth = -min(dist_km * 20, 100)

            return BathymetryResult(
                depth=estimated_depth,
                source='estimated',
                confidence=0.3
            )

        return None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def analyze_fishing_area(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    gebco_file: Optional[str] = None,
    substrate_hint: Optional[str] = None
) -> RealDataResult:
    """
    Analyze a fishing area and generate zones.

    Args:
        lat_min, lat_max, lon_min, lon_max: Bounding box
        gebco_file: Path to GEBCO NetCDF file (optional)
        substrate_hint: Override substrate type

    Returns:
        RealDataResult with coastline and zones
    """
    config = RealDataConfig(gebco_file=gebco_file)
    pipeline = RealDataPipeline(config)

    return pipeline.analyze_area(
        lat_min, lat_max, lon_min, lon_max,
        use_gebco=gebco_file is not None,
        substrate_hint=substrate_hint
    )


def export_result_to_file(
    result: RealDataResult,
    output_path: Union[str, Path],
    format: str = 'geojson'
) -> bool:
    """
    Export analysis result to file.

    Args:
        result: RealDataResult from analyze_fishing_area
        output_path: Output file path
        format: 'geojson' or 'json'

    Returns:
        True if successful
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = result.to_geojson()

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported to {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error exporting: {e}")
        return False
