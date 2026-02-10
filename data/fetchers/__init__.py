"""
Data Fetchers Module.

Provides access to historical and real-time oceanographic data:
- historical_fetcher: Multi-year data from Copernicus, NOAA, GFW
- tide_fetcher: Astronomical tide calculations
- copernicus_physics_fetcher: SSS and SLA from Copernicus
- copernicus_chlorophyll_fetcher: Chlorophyll-a from Copernicus (V7)
- sst_historical_provider: Historical SST with anomalies (V7)
- gfw_hotspot_generator: Dynamic hotspots from GFW (V7)
"""

from .historical_fetcher import (
    HistoricalDataFetcher,
    FishMovementPredictor,
    HistoricalDataPoint
)

# Optional imports with graceful fallback
try:
    from .tide_fetcher import TideFetcher
except ImportError:
    TideFetcher = None

try:
    from .copernicus_physics_fetcher import CopernicusPhysicsFetcher
except ImportError:
    CopernicusPhysicsFetcher = None

try:
    from .copernicus_chlorophyll_fetcher import ChlorophyllFetcher
except ImportError:
    ChlorophyllFetcher = None

try:
    from .sst_historical_provider import SSTHistoricalProvider
except ImportError:
    SSTHistoricalProvider = None

try:
    from .gfw_hotspot_generator import GFWHotspotGenerator
except ImportError:
    GFWHotspotGenerator = None

__all__ = [
    'HistoricalDataFetcher',
    'FishMovementPredictor',
    'HistoricalDataPoint',
    'TideFetcher',
    'CopernicusPhysicsFetcher',
    'ChlorophyllFetcher',
    'SSTHistoricalProvider',
    'GFWHotspotGenerator',
]
