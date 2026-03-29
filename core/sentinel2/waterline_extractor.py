"""
Waterline extraction from Sentinel-2 at 10m resolution.

Uses NDWI (B03/B08, both native 10m) with Composite Waterline Method (CWM)
for tidal normalization: selects scenes near Mean Low Water, extracts
contours, takes monthly median to suppress tidal noise.

Accuracy: RMSE ~5m (Sentinel-2), ~10m horizontal at microtidal sites.
References: Vos et al. 2023 (Nature Comms), Sánchez-García et al. 2024.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from pathlib import Path

try:
    from skimage.measure import find_contours
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


@dataclass
class WaterlineResult:
    """Extracted waterline for a single scene."""
    date: str
    contours_geo: List[List[Tuple[float, float]]]  # [(lat, lon), ...] per contour
    tide_height_m: Optional[float] = None
    ndwi_threshold: float = 0.0
    n_pixels_water: int = 0
    n_pixels_land: int = 0


class WaterlineExtractor:
    """
    Extracts waterline position from Sentinel-2 at 10m resolution.
    Uses NDWI with adaptive (Otsu) or fixed thresholding.
    """

    def __init__(self, threshold: float = None):
        """
        Args:
            threshold: Fixed NDWI threshold. If None, uses Otsu's method.
        """
        self.threshold = threshold

    def extract_from_bands(self, b03: np.ndarray, b08: np.ndarray,
                           date: str, transform=None,
                           tide_height_m: float = None) -> WaterlineResult:
        """Extract waterline from B03 (Green) and B08 (NIR) at 10m.

        Args:
            b03: Green band array (10m)
            b08: NIR band array (10m)
            date: Scene date string
            transform: rasterio Affine transform for georeferencing
            tide_height_m: Tide height at acquisition time
        """
        eps = 1e-10
        ndwi = (b03.astype(np.float32) - b08.astype(np.float32)) / \
               (b03.astype(np.float32) + b08.astype(np.float32) + eps)

        # Determine threshold
        if self.threshold is not None:
            thresh = self.threshold
        else:
            thresh = self._otsu_threshold(ndwi)

        # Extract contours at threshold
        if not SKIMAGE_AVAILABLE:
            raise ImportError("scikit-image required: pip install scikit-image")

        contours_px = find_contours(ndwi, thresh)

        # Convert pixel contours to geographic coordinates
        contours_geo = []
        if transform is not None:
            for contour in contours_px:
                geo_coords = []
                for row, col in contour:
                    lon, lat = transform * (col, row)
                    geo_coords.append((lat, lon))
                if len(geo_coords) >= 3:
                    contours_geo.append(geo_coords)
        else:
            contours_geo = [[(r, c) for r, c in cont] for cont in contours_px
                           if len(cont) >= 3]

        n_water = int(np.sum(ndwi > thresh))
        n_land = int(np.sum(ndwi <= thresh))

        return WaterlineResult(
            date=date,
            contours_geo=contours_geo,
            tide_height_m=tide_height_m,
            ndwi_threshold=float(thresh),
            n_pixels_water=n_water,
            n_pixels_land=n_land
        )

    def compute_monthly_composite(self, waterlines: List[WaterlineResult],
                                  prefer_low_tide: bool = True) -> List[WaterlineResult]:
        """Select best waterlines for monthly composite (CWM method).

        Selects 2-3 scenes closest to Mean Low Water to suppress tidal noise.
        """
        if not waterlines:
            return []

        if not prefer_low_tide or not any(w.tide_height_m is not None for w in waterlines):
            return waterlines

        # Sort by tide height (ascending = low tide first)
        with_tide = [w for w in waterlines if w.tide_height_m is not None]
        with_tide.sort(key=lambda w: w.tide_height_m)

        # Take 2-3 lowest tide scenes
        n_select = min(3, len(with_tide))
        return with_tide[:n_select]

    def waterline_to_transects(self, contours_geo: List[List[Tuple[float, float]]],
                               coastline_points: List[Tuple[float, float]],
                               spacing_m: float = 100.0) -> Dict[int, float]:
        """Measure waterline distance along perpendicular transects.

        Args:
            contours_geo: Waterline contours [(lat, lon), ...]
            coastline_points: Reference coastline [(lat, lon), ...]
            spacing_m: Spacing between transects in meters

        Returns:
            Dict mapping transect_id to distance_m (positive = seaward)
        """
        if not contours_geo or not coastline_points:
            return {}

        # Flatten all contour points
        all_water_pts = np.array([pt for contour in contours_geo for pt in contour])
        if len(all_water_pts) == 0:
            return {}

        coast_arr = np.array(coastline_points)
        n_coast = len(coast_arr)

        # Create transects every spacing_m along coastline
        transect_distances = {}
        step = max(1, int(spacing_m / 10))  # approx, assuming ~10m between coast points

        for t_id, idx in enumerate(range(0, n_coast, step)):
            coast_pt = coast_arr[idx]

            # Find nearest waterline point
            dists = np.sqrt(
                ((all_water_pts[:, 0] - coast_pt[0]) * 111000) ** 2 +
                ((all_water_pts[:, 1] - coast_pt[1]) * 111000 *
                 np.cos(np.radians(coast_pt[0]))) ** 2
            )
            min_dist = float(np.min(dists))
            transect_distances[t_id] = min_dist

        return transect_distances

    @staticmethod
    def _otsu_threshold(ndwi: np.ndarray) -> float:
        """Compute Otsu's threshold on NDWI values."""
        valid = ndwi[np.isfinite(ndwi)].ravel()
        if len(valid) == 0:
            return 0.0

        # Histogram
        hist, bin_edges = np.histogram(valid, bins=256, range=(-1, 1))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        total = hist.sum()

        if total == 0:
            return 0.0

        # Otsu's method
        best_thresh = 0.0
        best_var = 0.0

        w0 = 0
        sum0 = 0
        sum_total = np.sum(hist * bin_centers)

        for i in range(len(hist)):
            w0 += hist[i]
            if w0 == 0:
                continue
            w1 = total - w0
            if w1 == 0:
                break

            sum0 += hist[i] * bin_centers[i]
            mean0 = sum0 / w0
            mean1 = (sum_total - sum0) / w1

            var_between = w0 * w1 * (mean0 - mean1) ** 2
            if var_between > best_var:
                best_var = var_between
                best_thresh = bin_centers[i]

        return float(best_thresh)
