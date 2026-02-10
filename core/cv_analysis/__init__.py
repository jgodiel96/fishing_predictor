"""
Computer Vision Analysis Module for Fishing Predictor V8.

Components:
- CoastlineDetectorCV: Precise coastline detection using HSV analysis
- SubstrateClassifier: Rock/Sand/Mixed classification from imagery
- SatelliteBathymetry: Depth estimation from Blue/Green band ratio (Stumpf)
- GEBCOBathymetry: Global bathymetry from GEBCO data
- BathymetryFusion: Combines SDB and GEBCO intelligently
- SpeciesZoneGenerator: Generate fishing zones by species habitat
- CVAnalysisPipeline: Full pipeline integrating all components

Usage:
    from core.cv_analysis import CVAnalysisPipeline

    pipeline = CVAnalysisPipeline()
    result = pipeline.analyze_area(lat_min, lat_max, lon_min, lon_max)
    result.save_geojson(Path('output.geojson'))
"""

from .coastline_detector import (
    CoastlineDetectorCV,
    TileConfig,
    detect_coastline_cv,
)
from .substrate_classifier import (
    SubstrateClassifier,
    SubstrateType,
    SubstrateResult,
    classify_substrate_from_image,
)
from .bathymetry import (
    SatelliteBathymetry,
    GEBCOBathymetry,
    BathymetryFusion,
    BathymetryResult,
    SDBConfig,
    estimate_depth_from_image,
    get_depth_zones,
)
from .species_zones import (
    SpeciesZoneGenerator,
    SpeciesZone,
    SpeciesHabitat,
    DepthZone,
    SPECIES_DATABASE,
    generate_species_zones,
    get_species_at_point,
)
from .pipeline import (
    CVAnalysisPipeline,
    CVAnalysisResult,
    analyze_fishing_area,
)

__all__ = [
    # Coastline
    'CoastlineDetectorCV',
    'TileConfig',
    'detect_coastline_cv',
    # Substrate
    'SubstrateClassifier',
    'SubstrateType',
    'SubstrateResult',
    'classify_substrate_from_image',
    # Bathymetry
    'SatelliteBathymetry',
    'GEBCOBathymetry',
    'BathymetryFusion',
    'BathymetryResult',
    'SDBConfig',
    'estimate_depth_from_image',
    'get_depth_zones',
    # Species Zones
    'SpeciesZoneGenerator',
    'SpeciesZone',
    'SpeciesHabitat',
    'DepthZone',
    'SPECIES_DATABASE',
    'generate_species_zones',
    'get_species_at_point',
    # Pipeline
    'CVAnalysisPipeline',
    'CVAnalysisResult',
    'analyze_fishing_area',
]
