"""
System configuration module.

Runtime configuration that may vary between environments.
For domain constants (species, hotspots, thresholds), use domain.py.
"""

import os
from pathlib import Path

# Re-export domain constants for convenience
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

# Coastline v8: Extended OSM data including Canepa/Sama
# 20 segments, 281.4 km, 7741 points
COASTLINE_FILE = GOLD_DIR / "coastline" / "coastline_v8_extended.geojson"


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
# ERDDAP DATASETS
# =============================================================================

ERDDAP_DATASETS = {
    "sst_daily": "jplMURSST41",
    "sst_8day": "erdMH1sstd8day",
    "chlorophyll_daily": "nesdisVHNSQchlaDaily",
    "chlorophyll_8day": "erdMH1chla8day",
}


# =============================================================================
# ENVIRONMENT VARIABLES
# =============================================================================

GFW_API_KEY = os.environ.get("GFW_API_KEY", "")
COPERNICUS_USER = os.environ.get("COPERNICUS_USER", "")
COPERNICUS_PASS = os.environ.get("COPERNICUS_PASS", "")
