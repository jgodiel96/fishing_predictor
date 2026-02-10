"""
Computer Vision Analysis Module for Fishing Predictor V8.

Two approaches available:

1. REAL DATA PIPELINE (Recommended):
   - OSMCoastlineLoader: Verified coastline from OpenStreetMap
   - GEBCOBathymetry: Global bathymetry data (~450m resolution)
   - RealDataPipeline: Generates zones from verified data sources

2. CV-BASED PIPELINE (Experimental):
   - CoastlineDetectorCV: HSV-based coastline detection (RGB only)
   - SubstrateClassifier: Rock/Sand/Mixed classification
   - SatelliteBathymetry: Stumpf algorithm for depth (requires calibration)
   - CVAnalysisPipeline: Full CV pipeline

The Real Data approach is more reliable because:
- OSM coastlines are verified by humans
- GEBCO provides actual measured depth data
- No dependency on image quality or spectral bands

Usage (Recommended):
    from core.cv_analysis import RealDataPipeline

    pipeline = RealDataPipeline()
    result = pipeline.analyze_area(lat_min, lat_max, lon_min, lon_max)

Usage (CV-based):
    from core.cv_analysis import CVAnalysisPipeline

    pipeline = CVAnalysisPipeline()
    result = pipeline.analyze_area(lat_min, lat_max, lon_min, lon_max)
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
)
from .osm_coastline import (
    OSMCoastlineLoader,
    OSMCoastlineConfig,
    CoastlineResult,
    CoastalZoneGenerator,
    load_coastline,
    export_coastline_geojson,
    haversine_distance,
)
from .real_data_pipeline import (
    RealDataPipeline,
    RealDataConfig,
    RealDataResult,
    RealDataZone,
    analyze_fishing_area,
    export_result_to_file,
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
    # CV Pipeline (experimental)
    'CVAnalysisPipeline',
    'CVAnalysisResult',
    # OSM Coastline (real data)
    'OSMCoastlineLoader',
    'OSMCoastlineConfig',
    'CoastlineResult',
    'CoastalZoneGenerator',
    'load_coastline',
    'export_coastline_geojson',
    'haversine_distance',
    # Real Data Pipeline (recommended)
    'RealDataPipeline',
    'RealDataConfig',
    'RealDataResult',
    'RealDataZone',
    'analyze_fishing_area',
    'export_result_to_file',
]
