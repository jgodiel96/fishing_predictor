"""
Habitat accessibility score provider for the ML pipeline.

Reads precomputed stability_map.parquet (Gold layer) and provides
spatial lookup for 6 new features per fishing spot:
- habitat_accessibility (0-1)
- substrate_stability (0-1)
- waterline_anomaly (-3 to +3 std)
- seasonal_burial_risk (0-1)
- days_since_substrate_change (normalized 0-1)
- sand_advance_rate (normalized 0-1)

Uses scipy.spatial.cKDTree for fast nearest-neighbor lookup
across 224k+ fishing spots.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from pathlib import Path

try:
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class HabitatFeatures:
    """6 habitat features for a single fishing spot."""
    habitat_accessibility: float     # 0-1
    substrate_stability: float       # 0-1
    waterline_anomaly: float         # -3 to +3
    seasonal_burial_risk: float      # 0-1
    days_since_change_norm: float    # 0-1 (normalized)
    sand_advance_rate_norm: float    # 0-1 (normalized)


class HabitatAccessibilityProvider:
    """
    Provides habitat accessibility features for the ML pipeline.
    Loads precomputed stability map and performs spatial lookups.
    """

    def __init__(self, stability_map_path: Optional[str] = None):
        from data.data_config import DataConfig

        self.path = Path(stability_map_path) if stability_map_path else DataConfig.STABILITY_MAP
        self._data: Optional[pd.DataFrame] = None
        self._tree: Optional[object] = None
        self._loaded = False

        # Normalization constants
        self._max_days_since_change = 3650.0  # 10 years
        self._max_sand_rate = 1000.0  # m²/year

    def load(self) -> bool:
        """Load stability map from parquet. Returns True if successful."""
        if self._loaded:
            return True

        if not self.path.exists():
            return False

        try:
            self._data = pd.read_parquet(self.path)
            if len(self._data) == 0:
                return False

            # Build KDTree for spatial lookup
            if SCIPY_AVAILABLE:
                coords = self._data[['lat', 'lon']].values
                mean_lat = np.mean(coords[:, 0])
                cos_lat = np.cos(np.radians(mean_lat))
                # Convert to meters for distance calculation
                coords_m = np.column_stack([
                    coords[:, 0] * 111000.0,
                    coords[:, 1] * 111000.0 * cos_lat
                ])
                self._tree = cKDTree(coords_m)

                # Precompute normalization
                if 'days_since_last_change' in self._data.columns:
                    max_days = self._data['days_since_last_change'].max()
                    if max_days > 0:
                        self._max_days_since_change = float(max_days)
                if 'sand_advance_rate_m2_yr' in self._data.columns:
                    max_rate = self._data['sand_advance_rate_m2_yr'].max()
                    if max_rate > 0:
                        self._max_sand_rate = float(max_rate)

            self._loaded = True
            print(f"[OK] Sentinel-2 stability map: {len(self._data)} pixels cargados")
            return True

        except Exception as e:
            print(f"[WARN] Error cargando stability map: {e}")
            return False

    def get_features(self, lat: float, lon: float) -> HabitatFeatures:
        """Get 6 habitat features for a single spot via spatial lookup."""
        if not self._loaded or self._tree is None:
            return self._default_features()

        # Find nearest pixel
        mean_lat = np.mean(self._data['lat'].values)
        cos_lat = np.cos(np.radians(mean_lat))
        query_m = np.array([[lat * 111000.0, lon * 111000.0 * cos_lat]])
        dist, idx = self._tree.query(query_m)

        # If nearest pixel is > 2km away, return defaults
        if dist[0] > 2000:
            return self._default_features()

        row = self._data.iloc[idx[0]]
        return self._row_to_features(row)

    def get_features_batch(self, lats: np.ndarray,
                           lons: np.ndarray) -> np.ndarray:
        """Get 6 habitat features for all spots at once (vectorized).

        Returns:
            ndarray of shape (N, 6) with features for each spot
        """
        N = len(lats)
        result = np.full((N, 6), 0.5)  # Default neutral values

        if not self._loaded or self._tree is None:
            return result

        mean_lat = np.mean(self._data['lat'].values)
        cos_lat = np.cos(np.radians(mean_lat))

        query_m = np.column_stack([
            lats * 111000.0,
            lons * 111000.0 * cos_lat
        ])
        dists, indices = self._tree.query(query_m)

        # Only use results within 2km
        valid = dists < 2000
        valid_idx = indices[valid]

        if np.sum(valid) == 0:
            return result

        rows = self._data.iloc[valid_idx]

        # Compute features vectorized
        result[valid, 0] = self._compute_accessibility(rows)
        result[valid, 1] = rows['stability_score'].values
        result[valid, 2] = 0.0  # waterline_anomaly (needs current waterline)
        result[valid, 3] = rows['is_seasonal_burial'].astype(float).values

        if 'days_since_last_change' in rows.columns:
            result[valid, 4] = np.clip(
                rows['days_since_last_change'].values / self._max_days_since_change, 0, 1
            )
        if 'sand_advance_rate_m2_yr' in rows.columns:
            result[valid, 5] = np.clip(
                rows['sand_advance_rate_m2_yr'].values / self._max_sand_rate, 0, 1
            )

        return result

    def _compute_accessibility(self, rows: pd.DataFrame) -> np.ndarray:
        """Compute habitat accessibility score from substrate data."""
        substrate = rows['dominant_substrate'].values
        stability = rows['stability_score'].values

        # Base score by substrate
        base = np.where(substrate == 'rock', 1.0,
               np.where(substrate == 'mixed', 0.6,
               np.where(substrate == 'sand', 0.2, 0.3)))

        # Stability bonus
        bonus = stability * 0.2

        # Burial penalty
        is_burial = rows['is_seasonal_burial'].values.astype(float)
        penalty = is_burial * 0.15

        return np.clip(base + bonus - penalty, 0.0, 1.0)

    def _row_to_features(self, row) -> HabitatFeatures:
        """Convert a DataFrame row to HabitatFeatures."""
        substrate = row.get('dominant_substrate', 'mixed')
        base = {'rock': 1.0, 'mixed': 0.6, 'sand': 0.2, 'water': 0.3}.get(substrate, 0.5)
        stability = row.get('stability_score', 0.5)
        is_burial = row.get('is_seasonal_burial', False)

        accessibility = min(1.0, base + stability * 0.2 - (0.15 if is_burial else 0.0))

        days_since = row.get('days_since_last_change', 0)
        sand_rate = row.get('sand_advance_rate_m2_yr', 0)

        return HabitatFeatures(
            habitat_accessibility=round(accessibility, 3),
            substrate_stability=round(stability, 3),
            waterline_anomaly=0.0,
            seasonal_burial_risk=1.0 if is_burial else 0.0,
            days_since_change_norm=round(min(days_since / self._max_days_since_change, 1.0), 3),
            sand_advance_rate_norm=round(min(sand_rate / self._max_sand_rate, 1.0), 3)
        )

    @staticmethod
    def _default_features() -> HabitatFeatures:
        return HabitatFeatures(
            habitat_accessibility=0.5,
            substrate_stability=0.5,
            waterline_anomaly=0.0,
            seasonal_burial_risk=0.0,
            days_since_change_norm=0.5,
            sand_advance_rate_norm=0.0
        )
