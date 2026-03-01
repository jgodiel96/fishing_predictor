"""
Domain module - Centralized domain constants and data structures.

This module contains all domain-specific data for the fishing prediction system,
using efficient immutable data structures (NamedTuple, frozenset) instead of dicts.

Based on research from:
- IMARPE (Instituto del Mar del Peru)
- Humboldt Current System studies
- Belkin-O'Reilly thermal front detection
- FAO fisheries reports for Peru
"""

from typing import NamedTuple, Tuple, FrozenSet
from dataclasses import dataclass
from enum import Enum, auto
import math


# =============================================================================
# DATE CONFIGURATION
# =============================================================================

SYSTEM_DATE = "2026-01-28"
TIMEZONE = "America/Lima"


# =============================================================================
# GEOGRAPHIC BOUNDS
# =============================================================================

class BoundingBox(NamedTuple):
    """Immutable geographic bounding box."""
    north: float
    south: float
    west: float
    east: float

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.north + self.south) / 2, (self.west + self.east) / 2)

    def contains(self, lat: float, lon: float) -> bool:
        return self.south <= lat <= self.north and self.west <= lon <= self.east


# Primary study area: Tacna-Ilo coast + Northern Chile extension
# Extended area for oceanographic analysis (currents, fish movement, thermal fronts)
# Recommendations are filtered to Peru only (see PERU_LIMIT)
STUDY_AREA = BoundingBox(
    north=-17.20,   # North: beyond Ilo
    south=-18.55,   # South: includes Chile for analysis context
    west=-71.60,    # West: coastal waters
    east=-70.25     # East: inland reference
)

# Peru-Chile border latitude (Concordia)
# Used to filter recommendations to Peru only
PERU_SOUTH_LIMIT = -18.35

# Peru-only area (for reference)
PERU_AREA = BoundingBox(
    north=-17.20,
    south=-18.35,   # Border with Chile
    west=-71.60,
    east=-70.30
)


# =============================================================================
# SUBSTRATE TYPES
# =============================================================================

class Substrate(Enum):
    """Coastal substrate classification."""
    ROCK = auto()
    SAND = auto()
    MIXED = auto()


# =============================================================================
# SPECIES DATA
# =============================================================================

class Species(NamedTuple):
    """Fish species with optimal conditions."""
    name: str
    temp_min: float
    temp_max: float
    substrate: Tuple[Substrate, ...]
    recommended_lures: str

    @property
    def temp_optimal(self) -> float:
        return (self.temp_min + self.temp_max) / 2

    def temp_score(self, sst: float) -> float:
        """Calculate temperature suitability score (0-1)."""
        if self.temp_min <= sst <= self.temp_max:
            center = self.temp_optimal
            half_range = (self.temp_max - self.temp_min) / 2
            return 1.0 - abs(sst - center) / half_range
        return 0.1


# Immutable species registry - based on IMARPE research
SPECIES: FrozenSet[Species] = frozenset({
    Species(
        name="Cabrilla",
        temp_min=16.0, temp_max=19.0,
        substrate=(Substrate.ROCK,),
        recommended_lures="Grubs 3\", jigs 15-25g"
    ),
    Species(
        name="Corvina",
        temp_min=15.0, temp_max=18.0,
        substrate=(Substrate.SAND, Substrate.MIXED),
        recommended_lures="Jigs metalicos 30-50g"
    ),
    Species(
        name="Robalo",
        temp_min=17.0, temp_max=21.0,
        substrate=(Substrate.ROCK, Substrate.MIXED),
        recommended_lures="Poppers 12cm, minnows"
    ),
    Species(
        name="Lenguado",
        temp_min=14.0, temp_max=17.0,
        substrate=(Substrate.SAND,),
        recommended_lures="Vinilos paddle tail"
    ),
    Species(
        name="Pejerrey",
        temp_min=14.0, temp_max=18.0,
        substrate=(Substrate.SAND, Substrate.MIXED),
        recommended_lures="Cucharillas pequenas"
    ),
    Species(
        name="Bonito",
        temp_min=17.0, temp_max=23.0,
        substrate=(Substrate.SAND, Substrate.MIXED),
        recommended_lures="Jigs metalicos 40-60g, poppers, cucharas"
    ),
})

# Quick lookup by name
SPECIES_BY_NAME = {sp.name: sp for sp in SPECIES}


# =============================================================================
# FISHING LOCATIONS (Hotspots)
# =============================================================================

class FishingLocation(NamedTuple):
    """Known fishing location with historical data."""
    name: str
    lat: float
    lon: float
    substrate: Substrate
    bonus_factor: float  # Historical productivity multiplier
    description: str

    def distance_to(self, lat: float, lon: float) -> float:
        """Haversine distance in meters."""
        R = 6_371_000
        phi1, phi2 = math.radians(self.lat), math.radians(lat)
        dphi = math.radians(lat - self.lat)
        dlam = math.radians(lon - self.lon)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
        return 2 * R * math.asin(math.sqrt(a))


# Historical hotspots - verified locations from IMARPE and local fishermen
# Coordinates corrected to match OSM coastline (2026-01-30)
HOTSPOTS: Tuple[FishingLocation, ...] = (
    # Moquegua - Ilo (north to south)
    FishingLocation("Ilo Puerto", -17.632, -71.348, Substrate.MIXED, 1.1, "Muelle y rocas adyacentes"),
    FishingLocation("Pozo Lizas", -17.642, -71.348, Substrate.ROCK, 1.2, "Pozas entre rocas, ideal spinning"),
    FishingLocation("Pocoma", -17.682, -71.377, Substrate.ROCK, 1.15, "Acantilados rocosos"),
    FishingLocation("Punta Coles", -17.702, -71.385, Substrate.ROCK, 1.35, "Reserva, rocas con mucha vida marina"),
    FishingLocation("Playa Media Luna", -17.732, -71.280, Substrate.SAND, 1.1, "Bahia arenosa en forma de media luna"),
    FishingLocation("Fundicion", -17.757, -71.233, Substrate.ROCK, 1.2, "Rocas grandes, estructura compleja"),
    FishingLocation("Pozo Redondo", -17.782, -71.189, Substrate.MIXED, 1.25, "Pozas naturales entre rocas"),
    FishingLocation("Punta Blanca", -17.812, -71.168, Substrate.ROCK, 1.3, "Punta rocosa, buena para robalo"),
    FishingLocation("Playa Blanca-Gentillar", -17.822, -71.140, Substrate.SAND, 1.30, "Playa abierta, corvina con dron 300m. Validado feb-2026: 140kg en 2 dias"),
    FishingLocation("Gentillar", -17.842, -71.113, Substrate.ROCK, 1.15, "Costa rocosa escarpada"),
    # Ite
    FishingLocation("Ite Sur", -17.872, -71.081, Substrate.ROCK, 1.2, "Formaciones rocosas, cabrilla"),
    FishingLocation("Ite Centro", -17.902, -70.998, Substrate.MIXED, 1.15, "Zona mixta productiva"),
    FishingLocation("Ite Norte", -17.932, -70.938, Substrate.SAND, 1.2, "Playa amplia, surgencia activa"),
    FishingLocation("Carlepe", -17.962, -70.905, Substrate.MIXED, 1.1, "Rocas y arena alternadas"),
    FishingLocation("Llostay", -17.96, -70.88, Substrate.ROCK, 1.2, "Chita y lorna, spinning. Validado feb-2026"),
    FishingLocation("Punta Mesa", -17.988, -70.889, Substrate.ROCK, 1.15, "Punta rocosa con pozas"),
    # Tacna
    FishingLocation("Vila Vila", -18.018, -70.876, Substrate.ROCK, 1.25, "Zona rocosa, buena estructura"),
    FishingLocation("Los Palos", -18.052, -70.807, Substrate.MIXED, 1.1, "Arena con rocas dispersas"),
    FishingLocation("Santa Rosa", -18.087, -70.759, Substrate.SAND, 1.15, "Playa extensa, buena para corvina"),
    FishingLocation("Boca del Rio", -18.1205, -70.728, Substrate.SAND, 1.2, "Desembocadura, playa arenosa"),
)


def get_nearby_hotspots(lat: float, lon: float, radius_m: float = 10_000) -> Tuple[FishingLocation, ...]:
    """Get hotspots within radius of a point."""
    return tuple(h for h in HOTSPOTS if h.distance_to(lat, lon) <= radius_m)


def get_species_for_substrate(substrate: Substrate) -> Tuple[Species, ...]:
    """Get species that prefer given substrate."""
    return tuple(sp for sp in SPECIES if substrate in sp.substrate)


# =============================================================================
# OCEANOGRAPHIC THRESHOLDS
# =============================================================================

class OceanographicThresholds(NamedTuple):
    """Immutable thresholds for oceanographic analysis."""
    # SST (Sea Surface Temperature)
    sst_optimal_min: float = 14.0
    sst_optimal_max: float = 24.0
    sst_optimal_center: float = 17.5

    # Thermal fronts (Belkin-O'Reilly 2009)
    gradient_weak: float = 0.04      # C/km for weak front
    gradient_strong: float = 0.10   # C/km for strong front
    gradient_threshold: float = 0.45  # C difference for front detection

    # Chlorophyll-a productivity
    chl_high: float = 2.0   # mg/m3 high productivity
    chl_low: float = 0.5    # mg/m3 low productivity

    # Safety thresholds
    wave_max: float = 2.0       # meters max for shore fishing
    wind_max: float = 25.0      # km/h max
    safety_threshold: float = 0.5

    # Spatial
    max_distance_km: float = 10.0   # km from coast of interest


THRESHOLDS = OceanographicThresholds()


# =============================================================================
# SCORING WEIGHTS
# =============================================================================

class ScoringWeights(NamedTuple):
    """Immutable weights for fishing score calculation."""
    front_proximity: float = 0.25
    chlorophyll: float = 0.20
    upwelling: float = 0.15
    fishing_activity: float = 0.15
    golden_hour: float = 0.10
    safety: float = 0.10
    lunar: float = 0.05

    def validate(self) -> bool:
        """Verify weights sum to 1.0."""
        total = sum(self)
        return abs(total - 1.0) < 0.001


WEIGHTS = ScoringWeights()


# =============================================================================
# FEATURE NAMES (32 features for ML)
# =============================================================================

FEATURE_NAMES: Tuple[str, ...] = (
    # SST features (6)
    'sst', 'sst_anomaly', 'sst_optimal_score', 'sst_species_score',
    'sst_variability', 'sst_trend',
    # Thermal front features (5)
    'gradient_magnitude', 'gradient_direction_sin', 'gradient_direction_cos',
    'is_thermal_front', 'front_intensity',
    # Current features (6)
    'current_speed', 'current_u', 'current_v',
    'current_convergence', 'current_shear', 'current_toward_coast',
    # Wave features (3)
    'wave_height', 'wave_period', 'wave_favorable',
    # Upwelling features (3)
    'upwelling_index', 'ekman_transport', 'upwelling_favorable',
    # Spatial features (4)
    'distance_to_coast', 'depth_proxy', 'coastal_zone', 'offshore_zone',
    # Historical features (2)
    'hotspot_distance', 'hotspot_similarity',
    # Temporal features (3)
    'hour_score', 'moon_score', 'season_score',
)

N_FEATURES = len(FEATURE_NAMES)  # 32


# =============================================================================
# API ENDPOINTS
# =============================================================================

class APIEndpoints(NamedTuple):
    """Immutable API endpoint configuration."""
    erddap_base: str = "https://coastwatch.pfeg.noaa.gov/erddap"
    openmeteo_marine: str = "https://marine-api.open-meteo.com/v1/marine"
    openmeteo_weather: str = "https://api.open-meteo.com/v1/forecast"
    gfw_api: str = "https://gateway.api.globalfishingwatch.org"


ENDPOINTS = APIEndpoints()


# =============================================================================
# ERDDAP DATASETS
# =============================================================================

class ERDDAPDataset(NamedTuple):
    """ERDDAP dataset configuration."""
    dataset_id: str
    variable: str
    description: str


ERDDAP_DATASETS: Tuple[ERDDAPDataset, ...] = (
    ERDDAPDataset("jplMURSST41", "analysed_sst", "MUR SST daily 1km"),
    ERDDAPDataset("erdMH1sstd8day", "sst", "MODIS SST 8-day composite"),
    ERDDAPDataset("nesdisVHNSQchlaDaily", "chlor_a", "VIIRS Chlorophyll-a daily"),
    ERDDAPDataset("erdMH1chla8day", "chlorophyll", "MODIS Chlorophyll-a 8-day"),
)


# =============================================================================
# VISUALIZATION
# =============================================================================

class ScoreCategory(NamedTuple):
    """Score category with color."""
    name: str
    min_score: float
    max_score: float
    color: str


SCORE_CATEGORIES: Tuple[ScoreCategory, ...] = (
    ScoreCategory("Pobre", 0, 20, "#ff4444"),
    ScoreCategory("Bajo", 20, 40, "#ff8c00"),
    ScoreCategory("Promedio", 40, 60, "#ffff00"),
    ScoreCategory("Bueno", 60, 80, "#90ee90"),
    ScoreCategory("Excelente", 80, 100, "#228b22"),
)


def get_score_category(score: float) -> ScoreCategory:
    """Get category for a score value."""
    for cat in SCORE_CATEGORIES:
        if cat.min_score <= score < cat.max_score:
            return cat
    return SCORE_CATEGORIES[-1]  # Excelente for score >= 100


def get_score_color(score: float) -> str:
    """Get color for a score value."""
    return get_score_category(score).color


# =============================================================================
# PREDICTION HORIZONS
# =============================================================================

PREDICTION_HORIZONS: Tuple[int, ...] = (0, 24, 48, 72)  # hours


# =============================================================================
# CACHE CONFIGURATION
# =============================================================================

CACHE_TTL_HOURS = 6
