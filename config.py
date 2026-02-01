"""
System configuration module.

This module provides runtime configuration that may vary between environments.
For domain constants (species, hotspots, thresholds), use domain.py instead.
"""

import os
from pathlib import Path

# Import domain constants for backward compatibility
from domain import (
    STUDY_AREA,
    HOTSPOTS,
    SPECIES,
    SPECIES_BY_NAME,
    THRESHOLDS,
    WEIGHTS,
    ENDPOINTS,
    FEATURE_NAMES,
    N_FEATURES,
    SCORE_CATEGORIES,
    PREDICTION_HORIZONS,
    CACHE_TTL_HOURS,
    SYSTEM_DATE,
    TIMEZONE,
    get_score_color,
    get_score_category,
    get_species_for_substrate,
    get_nearby_hotspots,
    Substrate,
    BoundingBox,
)


# =============================================================================
# LEGACY COMPATIBILITY - DEPRECATED
# Use domain.py types directly instead
# =============================================================================

# Deprecated: Use STUDY_AREA from domain.py
BBOX = {
    "north": STUDY_AREA.north,
    "south": STUDY_AREA.south,
    "west": STUDY_AREA.west,
    "east": STUDY_AREA.east,
}

# Deprecated: Use HOTSPOTS from domain.py
LOCATIONS = {
    loc.name: {
        "lat": loc.lat,
        "lon": loc.lon,
        "tipo": loc.substrate.name.lower(),
        "descripcion": loc.description,
    }
    for loc in HOTSPOTS
}

# Deprecated: Use WEIGHTS from domain.py
SCORING_WEIGHTS = {
    "front_proximity": WEIGHTS.front_proximity,
    "chlorophyll_score": WEIGHTS.chlorophyll,
    "upwelling_index": WEIGHTS.upwelling,
    "fishing_vessel_proxy": WEIGHTS.fishing_activity,
    "golden_hour": WEIGHTS.golden_hour,
    "safety_score": WEIGHTS.safety,
    "lunar_score": WEIGHTS.lunar,
}

# Deprecated: Use THRESHOLDS from domain.py
SAFETY_PENALTY_MULTIPLIER = 0.3
SAFETY_THRESHOLD = THRESHOLDS.safety_threshold


# =============================================================================
# ERDDAP DATASETS - Legacy format
# =============================================================================

ERDDAP_DATASETS = {
    "sst_daily": "jplMURSST41",
    "sst_8day": "erdMH1sstd8day",
    "chlorophyll_daily": "nesdisVHNSQchlaDaily",
    "chlorophyll_8day": "erdMH1chla8day",
}


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
OUTPUT_DIR = PROJECT_ROOT / "output"
HISTORICAL_DIR = PROJECT_ROOT / "data" / "historical"
GOLD_DIR = PROJECT_ROOT / "data" / "gold"

# Ensure directories exist
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GOLD_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_OUTPUT_FILE = "fishing_map.html"
# V5: Use refined coastline from Gold layer (6,732 points, 50m max spacing)
COASTLINE_FILE = GOLD_DIR / "coastline" / "coastline_v1.geojson"


# =============================================================================
# GRID CONFIGURATION
# =============================================================================

GRID_RESOLUTION_M = 500  # meters
GRID_RESOLUTION_DEG = GRID_RESOLUTION_M / 111_000  # approximate degrees


# =============================================================================
# MAP CENTER
# =============================================================================

MAP_CENTER = {"lat": STUDY_AREA.center[0], "lon": STUDY_AREA.center[1]}


# =============================================================================
# API ENDPOINTS - Legacy format
# =============================================================================

class APIEndpointsLegacy:
    """Legacy API endpoints class. Use ENDPOINTS from domain.py instead."""
    ERDDAP_BASE = ENDPOINTS.erddap_base
    OPENMETEO_MARINE = ENDPOINTS.openmeteo_marine
    OPENMETEO_WEATHER = ENDPOINTS.openmeteo_weather
    GFW_API = ENDPOINTS.gfw_api


# =============================================================================
# ENVIRONMENT VARIABLES
# =============================================================================

GFW_API_KEY = os.environ.get("GFW_API_KEY", "")
COPERNICUS_USER = os.environ.get("COPERNICUS_USER", "")
COPERNICUS_PASS = os.environ.get("COPERNICUS_PASS", "")
