"""
Core modules for Fishing Spot Predictor.
Robust, error-proof architecture.
"""

from .coastline import CoastlineModel, CoastlineSegment
from .transects import TransectAnalyzer, Transect
from .fish_movement import FishMovementPredictor, MovementVector
from .scoring import ScoringEngine

__all__ = [
    'CoastlineModel',
    'CoastlineSegment',
    'TransectAnalyzer',
    'Transect',
    'FishMovementPredictor',
    'MovementVector',
    'ScoringEngine'
]
