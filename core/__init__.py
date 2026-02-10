"""
Core modules for Fishing Spot Predictor.

Includes:
- coastline_real: Coastline data and interpolation
- weather_solunar: Weather and solunar calculations
- marine_data: SST, currents, thermal fronts
- cv_analysis: Computer Vision analysis (V8)
"""

from .coastline_real import RealCoastline, CoastPoint, load_coastline
from .weather_solunar import (
    WeatherConditions,
    SolunarData,
    WeatherFetcher,
    SolunarCalculator,
    get_fishing_conditions
)
from .marine_data import (
    MarineDataFetcher,
    ThermalFrontDetector,
    FishZonePredictor,
    MarinePoint,
    ThermalFront,
    CurrentVector,
    SSTHistory
)

# V8: Computer Vision Analysis
try:
    from .cv_analysis import (
        CVAnalysisPipeline,
        CVAnalysisResult,
        CoastlineDetectorCV,
        SubstrateClassifier,
        SubstrateType,
        SatelliteBathymetry,
        BathymetryFusion,
        SpeciesZoneGenerator,
        SpeciesZone,
        SPECIES_DATABASE,
    )
    HAS_CV_ANALYSIS = True
except ImportError:
    HAS_CV_ANALYSIS = False

__all__ = [
    'RealCoastline',
    'CoastPoint',
    'load_coastline',
    'WeatherConditions',
    'SolunarData',
    'WeatherFetcher',
    'SolunarCalculator',
    'get_fishing_conditions',
    'MarineDataFetcher',
    'ThermalFrontDetector',
    'FishZonePredictor',
    'MarinePoint',
    'ThermalFront',
    'CurrentVector',
    'SSTHistory',
    # V8 CV Analysis
    'HAS_CV_ANALYSIS',
]

# Add CV exports if available
if HAS_CV_ANALYSIS:
    __all__.extend([
        'CVAnalysisPipeline',
        'CVAnalysisResult',
        'CoastlineDetectorCV',
        'SubstrateClassifier',
        'SubstrateType',
        'SatelliteBathymetry',
        'BathymetryFusion',
        'SpeciesZoneGenerator',
        'SpeciesZone',
        'SPECIES_DATABASE',
    ])
