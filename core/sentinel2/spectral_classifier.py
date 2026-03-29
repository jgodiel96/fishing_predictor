"""
Spectral substrate classifier using Sentinel-2 multispectral indices.

Classifies coastal pixels at 20m resolution into water/rock/sand/mixed
using a 3-index decision tree: MNDWI + B11/B12 ratio + BSI.

Based on validated literature:
- MNDWI water/land: Xu 2006
- B11/B12 lithological discrimination: 74.5% accuracy (van der Werff 2018)
- BSI bare soil index: Rikimaru et al. 2002

Uses B8A (865nm, 20m native) instead of B08 (842nm, 10m) to avoid
SWIR resampling artifacts.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict
from pathlib import Path

try:
    import rasterio
    from rasterio.transform import Affine
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


# Substrate class codes (uint8)
NODATA = 0
WATER = 1
ROCK = 2
SAND = 3
MIXED = 4

SUBSTRATE_NAMES = {NODATA: 'nodata', WATER: 'water', ROCK: 'rock',
                   SAND: 'sand', MIXED: 'mixed'}

# SCL valid classes (Sen2Cor Scene Classification)
SCL_VALID = frozenset({4, 5, 6, 7})  # vegetation, bare_soil, water, unclassified


@dataclass
class ClassifierConfig:
    """Thresholds for spectral classification. Calibrate with ground truth."""
    # MNDWI threshold for water/land
    mndwi_water_threshold: float = 0.0

    # B11/B12 ratio thresholds for rock vs sand
    # Igneous rock (Peru coast): low B11 relative → ratio < swir_rock_max
    swir_rock_max: float = 1.5
    # Sand: high B11 → ratio > swir_sand_min
    swir_sand_min: float = 1.5

    # BSI threshold for bare soil detection
    bsi_sand_min: float = 0.1

    # Brightness thresholds (mean of B02+B03+B04)
    brightness_rock_max: float = 800.0
    brightness_sand_min: float = 1400.0


@dataclass
class ClassificationResult:
    """Result of classifying a single Sentinel-2 scene."""
    date: str
    substrate_map: np.ndarray       # uint8 2D array (NODATA/WATER/ROCK/SAND/MIXED)
    cloud_mask: np.ndarray          # bool 2D array (True = cloudy/invalid)
    mndwi: np.ndarray               # float32 2D array
    transform: Optional[object] = None  # rasterio Affine
    crs: Optional[str] = None
    cloud_pct: float = 0.0
    n_rock: int = 0
    n_sand: int = 0
    n_mixed: int = 0
    n_water: int = 0


class SpectralClassifier:
    """
    Classifies Sentinel-2 pixels into water/rock/sand/mixed
    using a 3-index spectral decision tree at 20m resolution.
    """

    def __init__(self, config: Optional[ClassifierConfig] = None):
        self.config = config or ClassifierConfig()

    def classify_scene(self, scene_path: str, date: str) -> ClassificationResult:
        """Classify a multiband GeoTIFF scene.

        Expects bands in order: B02, B03, B04, B8A, B11, B12, SCL
        (7 bands total, all at 20m resolution).
        """
        if not RASTERIO_AVAILABLE:
            raise ImportError("rasterio required: pip install rasterio")

        with rasterio.open(scene_path) as src:
            bands = src.read()  # (7, H, W)
            transform = src.transform
            crs = str(src.crs)

        b02 = bands[0].astype(np.float32)  # Blue
        b03 = bands[1].astype(np.float32)  # Green
        b04 = bands[2].astype(np.float32)  # Red
        b8a = bands[3].astype(np.float32)  # NIR narrow
        b11 = bands[4].astype(np.float32)  # SWIR-1
        b12 = bands[5].astype(np.float32)  # SWIR-2
        scl = bands[6].astype(np.uint8)    # Scene Classification

        return self.classify_bands(b02, b03, b04, b8a, b11, b12, scl,
                                   date, transform, crs)

    def classify_bands(self, b02: np.ndarray, b03: np.ndarray,
                       b04: np.ndarray, b8a: np.ndarray,
                       b11: np.ndarray, b12: np.ndarray,
                       scl: np.ndarray, date: str,
                       transform=None, crs: str = None) -> ClassificationResult:
        """Classify from individual band arrays (all 20m, same shape)."""
        cfg = self.config
        eps = 1e-10

        # Step 0: Cloud mask from SCL
        cloud_mask = ~np.isin(scl, list(SCL_VALID))
        valid = ~cloud_mask
        cloud_pct = np.sum(cloud_mask) / cloud_mask.size * 100

        # Step 1: MNDWI — water vs land
        mndwi = (b03 - b11) / (b03 + b11 + eps)
        is_water = mndwi > cfg.mndwi_water_threshold

        # Step 2: B11/B12 ratio — rock vs sand mineralogy
        swir_ratio = b11 / (b12 + eps)

        # Step 3: BSI — bare soil index
        bsi = ((b11 + b04) - (b8a + b02)) / ((b11 + b04) + (b8a + b02) + eps)

        # Step 4: Brightness
        brightness = (b02 + b03 + b04) / 3.0

        # Classification
        result = np.full(b02.shape, NODATA, dtype=np.uint8)

        # Water
        water_mask = valid & is_water
        result[water_mask] = WATER

        # Land pixels
        land = valid & ~is_water

        # Rock: dark, low SWIR ratio
        rock_mask = land & (brightness < cfg.brightness_rock_max) & (swir_ratio < cfg.swir_rock_max)
        result[rock_mask] = ROCK

        # Sand: bright, high BSI
        sand_mask = land & (brightness > cfg.brightness_sand_min) & (bsi > cfg.bsi_sand_min)
        result[sand_mask] = SAND

        # Mixed: land pixels not clearly rock or sand
        mixed_mask = land & (result == NODATA)
        result[mixed_mask] = MIXED

        return ClassificationResult(
            date=date,
            substrate_map=result,
            cloud_mask=cloud_mask,
            mndwi=mndwi,
            transform=transform,
            crs=crs,
            cloud_pct=round(cloud_pct, 1),
            n_rock=int(np.sum(result == ROCK)),
            n_sand=int(np.sum(result == SAND)),
            n_mixed=int(np.sum(result == MIXED)),
            n_water=int(np.sum(result == WATER))
        )

    def save_classification(self, result: ClassificationResult, output_path: str):
        """Save classification as categorical GeoTIFF (uint8)."""
        if not RASTERIO_AVAILABLE:
            raise ImportError("rasterio required")

        h, w = result.substrate_map.shape
        with rasterio.open(
            output_path, 'w', driver='GTiff',
            height=h, width=w, count=1, dtype='uint8',
            crs=result.crs, transform=result.transform,
            compress='lzw'
        ) as dst:
            dst.write(result.substrate_map, 1)

    def validate_against_ground_truth(self, result: ClassificationResult,
                                       ground_truth: list) -> Dict:
        """Validate classification against known substrate points.

        Args:
            ground_truth: list of dicts with 'lat', 'lon', 'substrate'
                         where substrate is 'rock', 'sand', or 'mixed'
        """
        if not RASTERIO_AVAILABLE or result.transform is None:
            return {'error': 'Cannot validate without georeferencing'}

        name_to_code = {'rock': ROCK, 'sand': SAND, 'mixed': MIXED, 'water': WATER}
        correct = 0
        total = 0
        confusion = {}

        for pt in ground_truth:
            col, row = ~result.transform * (pt['lon'], pt['lat'])
            row, col = int(row), int(col)
            h, w = result.substrate_map.shape
            if 0 <= row < h and 0 <= col < w:
                predicted = result.substrate_map[row, col]
                if predicted == NODATA:
                    continue
                expected = name_to_code.get(pt['substrate'], NODATA)
                if expected == NODATA:
                    continue
                total += 1
                if predicted == expected:
                    correct += 1
                key = (SUBSTRATE_NAMES[expected], SUBSTRATE_NAMES[predicted])
                confusion[key] = confusion.get(key, 0) + 1

        accuracy = correct / total if total > 0 else 0
        return {
            'overall_accuracy': round(accuracy, 3),
            'correct': correct,
            'total': total,
            'confusion': confusion
        }
