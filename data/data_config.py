"""
Data architecture configuration module.

Implements Bronze/Silver/Gold (Data Lakehouse) architecture:
- Bronze (raw/): Immutable raw data, partitioned by month
- Silver (processed/): Regenerable consolidated databases
- Gold (analytics/): ML-ready, versioned datasets

Usage:
    from data.data_config import DataConfig

    # Access paths
    gfw_dir = DataConfig.RAW_GFW
    training_db = DataConfig.FISHING_DB
"""

from pathlib import Path
from typing import Dict, Any
import os

# Load environment variables from .env file automatically
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent.parent / '.env'
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass  # dotenv not installed, use system environment variables


class DataConfig:
    """Centralized configuration for all data paths and settings."""

    # ==========================================================================
    # BASE PATHS
    # ==========================================================================

    PROJECT_ROOT = Path(__file__).parent.parent
    BASE_DIR = PROJECT_ROOT / "data"

    # ==========================================================================
    # BRONZE LAYER (Raw) - IMMUTABLE
    # ==========================================================================
    # Rule: NEVER modify existing files, only ADD new monthly partitions
    # Format: YYYY-MM.parquet
    # Each source has its own _manifest.json for tracking

    RAW_DIR = BASE_DIR / "raw"

    # Global Fishing Watch - AIS-based fishing activity
    RAW_GFW = RAW_DIR / "gfw"

    # Open-Meteo ERA5 - Waves, wind, marine conditions
    RAW_OPEN_METEO = RAW_DIR / "open_meteo"

    # SST from multiple sources
    RAW_SST_DIR = RAW_DIR / "sst"
    RAW_SST_NOAA = RAW_SST_DIR / "noaa"           # NOAA OISST
    RAW_SST_EARTHDATA = RAW_SST_DIR / "earthdata"  # NASA MUR SST (high-res)
    RAW_SST_COPERNICUS = RAW_SST_DIR / "copernicus"  # Copernicus Marine

    # SSS (Salinity) from Copernicus - Variable #1 según papers V2
    RAW_SSS_DIR = RAW_DIR / "sss"
    RAW_SSS_COPERNICUS = RAW_SSS_DIR / "copernicus"

    # SLA (Sea Level Anomaly) from Copernicus - Variable #2 según papers V2
    RAW_SLA_DIR = RAW_DIR / "sla"
    RAW_SLA_COPERNICUS = RAW_SLA_DIR / "copernicus"

    # Chlorophyll-a from Copernicus - Variable #3 (V7)
    RAW_CHLA_DIR = RAW_DIR / "chla"
    RAW_CHLA_COPERNICUS = RAW_CHLA_DIR / "copernicus"

    # User-submitted sightings (append-only)
    RAW_USER_SIGHTINGS = RAW_DIR / "user_sightings" / "sightings.jsonl"

    # ==========================================================================
    # SILVER LAYER (Processed) - REGENERABLE FROM BRONZE
    # ==========================================================================
    # Rule: Can be regenerated from Bronze at any time
    # Scripts in data/consolidator.py handle this

    PROCESSED_DIR = BASE_DIR / "processed"

    # Consolidated fishing data (all GFW + user sightings)
    FISHING_DB = PROCESSED_DIR / "fishing_consolidated.db"

    # Consolidated marine conditions (all Open-Meteo + IMARPE)
    MARINE_DB = PROCESSED_DIR / "marine_consolidated.db"

    # Pre-computed training features
    TRAINING_FEATURES = PROCESSED_DIR / "training_features.parquet"

    # Consolidation log for tracking regeneration
    CONSOLIDATION_LOG = PROCESSED_DIR / "_consolidation_log.json"

    # ==========================================================================
    # GOLD LAYER (Analytics) - VERSIONED, ML-READY
    # ==========================================================================
    # Rule: Versioned by date, keep last N versions

    ANALYTICS_DIR = BASE_DIR / "analytics"

    # Current active training dataset
    CURRENT_DIR = ANALYTICS_DIR / "current"
    CURRENT_TRAINING = CURRENT_DIR / "training_dataset.parquet"
    MODEL_METADATA = CURRENT_DIR / "model_metadata.json"

    # Historical versions for reproducibility
    VERSIONS_DIR = ANALYTICS_DIR / "versions"

    # Latest predictions
    PREDICTIONS_DIR = ANALYTICS_DIR / "predictions"
    LATEST_PREDICTIONS = PREDICTIONS_DIR / "latest.json"

    # Number of versions to keep
    MAX_VERSIONS = 5

    # ==========================================================================
    # LEGACY COMPATIBILITY
    # ==========================================================================
    # These paths point to old locations for backward compatibility
    # TODO: Remove after full migration

    LEGACY_DB = BASE_DIR / "real_only" / "real_data_100.db"
    LEGACY_HISTORICAL_DIR = BASE_DIR / "historical"
    LEGACY_CACHE_DIR = BASE_DIR / "cache"

    # Timeline database (used by models/timeline.py)
    # Points to Silver layer after migration
    TIMELINE_DB = FISHING_DB

    # ==========================================================================
    # METADATA
    # ==========================================================================

    METADATA_DIR = BASE_DIR / "metadata"
    SOURCES_FILE = METADATA_DIR / "sources.json"
    SCHEMA_FILE = METADATA_DIR / "schema.json"
    REGION_FILE = METADATA_DIR / "region.json"
    CHANGELOG_FILE = METADATA_DIR / "changelog.md"

    # ==========================================================================
    # BATHYMETRY (GEBCO)
    # ==========================================================================

    BATHYMETRY_DIR = BASE_DIR / "bathymetry"
    GEBCO_FILE = BATHYMETRY_DIR / "GEBCO_2025_peru.nc"  # Full Peru coast

    # GEBCO region bounds (full Peru coast)
    GEBCO_BOUNDS = {
        'lat_min': -20.0,  # South (Chile border)
        'lat_max': -3.0,   # North (Ecuador border)
        'lon_min': -82.0,  # West (offshore)
        'lon_max': -70.0,  # East (coast)
        'name': 'Peru Full Coast'
    }

    # ==========================================================================
    # COASTLINES (OSM)
    # ==========================================================================

    COASTLINES_DIR = BASE_DIR / "coastlines"
    WATER_POLYGONS_DIR = COASTLINES_DIR / "water-polygons-split-4326"
    WATER_POLYGONS_SHP = WATER_POLYGONS_DIR / "water_polygons.shp"

    # ==========================================================================
    # CACHE (Temporary - can be deleted)
    # ==========================================================================

    CACHE_DIR = BASE_DIR / "cache"
    API_CACHE_DIR = CACHE_DIR / "api_responses"

    # ==========================================================================
    # GEOGRAPHIC REGION (Tacna-Ilo, Peru)
    # ==========================================================================

    REGION = {
        'lat_min': -18.3,
        'lat_max': -17.3,
        'lon_min': -71.5,
        'lon_max': -70.2,  # Expanded to include Playa Canepa
        'name': 'Tacna-Ilo-Sama, Peru',
        'description': 'Coastal fishing zone from Ilo to Sama (includes Playa Canepa)'
    }

    # Grid resolution for spatial analysis
    GRID_RESOLUTION = 0.1  # degrees (~11km)

    # ==========================================================================
    # DATA SOURCES CONFIGURATION
    # ==========================================================================

    SOURCES = {
        'gfw': {
            'name': 'Global Fishing Watch',
            'type': 'fishing',
            'api': 'https://gateway.api.globalfishingwatch.org/v3/4wings/report',
            'auth': 'GFW_API_KEY',
            'format': 'parquet',
            'partition': 'monthly'
        },
        'open_meteo': {
            'name': 'Open-Meteo ERA5',
            'type': 'marine',
            'api': 'https://marine-api.open-meteo.com/v1/marine',
            'auth': None,
            'format': 'parquet',
            'partition': 'monthly'
        },
        'noaa_sst': {
            'name': 'NOAA OISST',
            'type': 'sst',
            'api': 'https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg_LonPM180',
            'auth': None,
            'format': 'parquet',
            'partition': 'monthly'
        },
        'earthdata_sst': {
            'name': 'NASA Earthdata MUR SST',
            'type': 'sst',
            'api': 'https://cmr.earthdata.nasa.gov/search/granules.json',
            'auth': 'EARTHDATA_USER',
            'format': 'parquet',
            'partition': 'monthly'
        },
        'copernicus_sst': {
            'name': 'Copernicus Marine SST',
            'type': 'sst',
            'api': 'copernicusmarine',
            'auth': 'COPERNICUS_USER',
            'format': 'parquet',
            'partition': 'monthly'
        },
        'imarpe': {
            'name': 'IMARPE Climatology',
            'type': 'sst',
            'api': None,  # Hardcoded climatological data
            'auth': None,
            'format': 'internal',
            'partition': None
        },
        'copernicus_sss': {
            'name': 'Copernicus Marine SSS',
            'type': 'physics',
            'api': 'copernicusmarine',
            'auth': 'COPERNICUS_USER',
            'format': 'parquet',
            'partition': 'monthly'
        },
        'copernicus_sla': {
            'name': 'Copernicus Marine SLA',
            'type': 'physics',
            'api': 'copernicusmarine',
            'auth': 'COPERNICUS_USER',
            'format': 'parquet',
            'partition': 'monthly'
        },
        'copernicus_chla': {
            'name': 'Copernicus Marine Chlorophyll-a',
            'type': 'biology',
            'api': 'copernicusmarine',
            'auth': 'COPERNICUS_USER',
            'dataset_id': 'cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D',
            'variable': 'CHL',
            'format': 'parquet',
            'partition': 'monthly'
        }
    }

    # ==========================================================================
    # TEMPORAL RANGE
    # ==========================================================================

    # Target data range for all sources
    DATA_START_YEAR = 2020
    DATA_START_MONTH = 1
    DATA_END_YEAR = 2026
    DATA_END_MONTH = 1

    # ==========================================================================
    # ENVIRONMENT VARIABLES
    # ==========================================================================

    @classmethod
    def get_api_key(cls, source: str) -> str:
        """Get API key for a data source from environment variables."""
        source_config = cls.SOURCES.get(source, {})
        auth_var = source_config.get('auth')
        if auth_var:
            return os.environ.get(auth_var, '')
        return ''

    @classmethod
    def get_gfw_api_key(cls) -> str:
        """Get Global Fishing Watch API key."""
        return os.environ.get('GFW_API_KEY', '')

    @classmethod
    def get_earthdata_credentials(cls) -> tuple:
        """Get NASA Earthdata credentials (user, password)."""
        return (
            os.environ.get('EARTHDATA_USER', ''),
            os.environ.get('EARTHDATA_PASS', '')
        )

    @classmethod
    def get_copernicus_credentials(cls) -> tuple:
        """Get Copernicus Marine credentials (user, password)."""
        return (
            os.environ.get('COPERNICUS_USER', ''),
            os.environ.get('COPERNICUS_PASS', '')
        )

    # ==========================================================================
    # UTILITY METHODS
    # ==========================================================================

    @classmethod
    def ensure_directories(cls) -> None:
        """Create all required directories if they don't exist."""
        directories = [
            cls.RAW_GFW,
            cls.RAW_OPEN_METEO,
            cls.RAW_SST_NOAA,
            cls.RAW_SST_EARTHDATA,
            cls.RAW_SST_COPERNICUS,
            cls.RAW_SSS_COPERNICUS,
            cls.RAW_SLA_COPERNICUS,
            cls.RAW_CHLA_COPERNICUS,
            cls.RAW_DIR / "user_sightings",
            cls.PROCESSED_DIR,
            cls.CURRENT_DIR,
            cls.VERSIONS_DIR,
            cls.PREDICTIONS_DIR,
            cls.METADATA_DIR,
            cls.API_CACHE_DIR,
            cls.BATHYMETRY_DIR,
            cls.COASTLINES_DIR,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_gebco_path(cls) -> Path:
        """Get the path to the GEBCO bathymetry file."""
        return cls.GEBCO_FILE

    @classmethod
    def has_gebco_data(cls) -> bool:
        """Check if GEBCO data is available."""
        return cls.GEBCO_FILE.exists()

    @classmethod
    def has_water_polygons(cls) -> bool:
        """Check if OSM water polygons are available."""
        return cls.WATER_POLYGONS_SHP.exists()

    @classmethod
    def get_raw_path(cls, source: str, year: int, month: int) -> Path:
        """Get the path for a raw data file.

        Args:
            source: Data source name (gfw, open_meteo, noaa_sst, etc.)
            year: Year (2020-2026)
            month: Month (1-12)

        Returns:
            Path to the parquet file
        """
        source_dirs = {
            'gfw': cls.RAW_GFW,
            'open_meteo': cls.RAW_OPEN_METEO,
            'noaa_sst': cls.RAW_SST_NOAA,
            'earthdata_sst': cls.RAW_SST_EARTHDATA,
            'copernicus_sst': cls.RAW_SST_COPERNICUS,
            'copernicus_sss': cls.RAW_SSS_COPERNICUS,
            'copernicus_sla': cls.RAW_SLA_COPERNICUS,
            'copernicus_chla': cls.RAW_CHLA_COPERNICUS,
        }
        base_dir = source_dirs.get(source)
        if not base_dir:
            raise ValueError(f"Unknown source: {source}")
        return base_dir / f"{year}-{month:02d}.parquet"

    @classmethod
    def get_manifest_path(cls, source: str) -> Path:
        """Get the path for a source's manifest file."""
        source_dirs = {
            'gfw': cls.RAW_GFW,
            'open_meteo': cls.RAW_OPEN_METEO,
            'noaa_sst': cls.RAW_SST_NOAA,
            'earthdata_sst': cls.RAW_SST_EARTHDATA,
            'copernicus_sst': cls.RAW_SST_COPERNICUS,
            'copernicus_sss': cls.RAW_SSS_COPERNICUS,
            'copernicus_sla': cls.RAW_SLA_COPERNICUS,
            'copernicus_chla': cls.RAW_CHLA_COPERNICUS,
        }
        base_dir = source_dirs.get(source)
        if not base_dir:
            raise ValueError(f"Unknown source: {source}")
        return base_dir / "_manifest.json"

    @classmethod
    def get_version_dir(cls, version_date: str) -> Path:
        """Get the directory for a specific version.

        Args:
            version_date: Version date string (e.g., '20260129')

        Returns:
            Path to version directory
        """
        return cls.VERSIONS_DIR / f"v{version_date}"

    @classmethod
    def list_raw_files(cls, source: str) -> list:
        """List all raw parquet files for a source.

        Args:
            source: Data source name

        Returns:
            List of Path objects sorted by filename
        """
        source_dirs = {
            'gfw': cls.RAW_GFW,
            'open_meteo': cls.RAW_OPEN_METEO,
            'noaa_sst': cls.RAW_SST_NOAA,
            'earthdata_sst': cls.RAW_SST_EARTHDATA,
            'copernicus_sst': cls.RAW_SST_COPERNICUS,
            'copernicus_sss': cls.RAW_SSS_COPERNICUS,
            'copernicus_sla': cls.RAW_SLA_COPERNICUS,
            'copernicus_chla': cls.RAW_CHLA_COPERNICUS,
        }
        base_dir = source_dirs.get(source)
        if not base_dir or not base_dir.exists():
            return []
        return sorted(base_dir.glob("*.parquet"))


# Convenience exports
RAW_DIR = DataConfig.RAW_DIR
RAW_GFW = DataConfig.RAW_GFW
RAW_OPEN_METEO = DataConfig.RAW_OPEN_METEO
RAW_SST_NOAA = DataConfig.RAW_SST_NOAA
RAW_SST_EARTHDATA = DataConfig.RAW_SST_EARTHDATA
RAW_SST_COPERNICUS = DataConfig.RAW_SST_COPERNICUS
RAW_SSS_COPERNICUS = DataConfig.RAW_SSS_COPERNICUS
RAW_SLA_COPERNICUS = DataConfig.RAW_SLA_COPERNICUS
RAW_CHLA_COPERNICUS = DataConfig.RAW_CHLA_COPERNICUS

PROCESSED_DIR = DataConfig.PROCESSED_DIR
FISHING_DB = DataConfig.FISHING_DB
MARINE_DB = DataConfig.MARINE_DB
TRAINING_FEATURES = DataConfig.TRAINING_FEATURES

ANALYTICS_DIR = DataConfig.ANALYTICS_DIR
CURRENT_TRAINING = DataConfig.CURRENT_TRAINING

LEGACY_DB = DataConfig.LEGACY_DB
TIMELINE_DB = DataConfig.TIMELINE_DB

REGION = DataConfig.REGION

# Bathymetry and coastlines
BATHYMETRY_DIR = DataConfig.BATHYMETRY_DIR
GEBCO_FILE = DataConfig.GEBCO_FILE
GEBCO_BOUNDS = DataConfig.GEBCO_BOUNDS
COASTLINES_DIR = DataConfig.COASTLINES_DIR
WATER_POLYGONS_SHP = DataConfig.WATER_POLYGONS_SHP
