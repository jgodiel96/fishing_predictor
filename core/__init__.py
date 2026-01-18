"""
Core modules for Fishing Spot Predictor.
"""

from .coastline_real import RealCoastline, CoastPoint, load_coastline
from .weather_solunar import (
    WeatherConditions,
    SolunarData,
    WeatherFetcher,
    SolunarCalculator,
    get_fishing_conditions
)

__all__ = [
    'RealCoastline',
    'CoastPoint',
    'load_coastline',
    'WeatherConditions',
    'SolunarData',
    'WeatherFetcher',
    'SolunarCalculator',
    'get_fishing_conditions'
]
