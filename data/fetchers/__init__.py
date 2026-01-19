"""
Data Fetchers Module.

Provides access to historical and real-time oceanographic data:
- historical_fetcher: Multi-year data from Copernicus, NOAA, GFW
- (future) real_time_fetcher: Live data streams
"""

from .historical_fetcher import (
    HistoricalDataFetcher,
    FishMovementPredictor,
    HistoricalDataPoint
)

__all__ = [
    'HistoricalDataFetcher',
    'FishMovementPredictor',
    'HistoricalDataPoint'
]
