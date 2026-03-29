"""
Sand displacement tracking between consecutive Sentinel-2 scenes.

Compares substrate classifications scene-to-scene to detect:
- Direction and speed of sand advance/retreat
- Burial events (rock → sand transitions)
- Exposure events (sand → rock transitions)
- Sand corridors (consistent movement patterns over years)

Uses ~649 consecutive scene pairs over 10 years to build a complete
picture of how sand moves along the coast.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from core.sentinel2.spectral_classifier import ROCK, SAND, MIXED, WATER, NODATA


@dataclass
class DisplacementResult:
    """Sand displacement between two consecutive scenes."""
    date_from: str
    date_to: str
    days_between: int
    burial_area_m2: float        # Area of rock → sand transitions
    exposure_area_m2: float      # Area of sand → rock transitions
    net_change_m2: float         # burial - exposure (positive = net burial)
    sand_direction_deg: float    # Mean direction of sand advance (degrees from N)
    n_burial_pixels: int
    n_exposure_pixels: int
    burial_centroids: List[Tuple[float, float]]    # (lat, lon) centers
    exposure_centroids: List[Tuple[float, float]]   # (lat, lon) centers


@dataclass
class PixelDisplacementMetrics:
    """Per-pixel displacement metrics over the full time series."""
    lat: float
    lon: float
    burial_events_count: int
    exposure_events_count: int
    mean_burial_duration_days: float
    max_burial_duration_days: float
    last_transition_date: str
    days_since_last_change: int
    sand_advance_direction_deg: float
    sand_advance_rate_m2_yr: float


class DisplacementTracker:
    """Tracks sand displacement between consecutive Sentinel-2 classifications."""

    def __init__(self, pixel_size_m: float = 20.0):
        self.pixel_size_m = pixel_size_m
        self.pixel_area_m2 = pixel_size_m ** 2

    def compute_displacement(self, substrate_t0: np.ndarray,
                             substrate_t1: np.ndarray,
                             date_t0: str, date_t1: str,
                             transform=None) -> DisplacementResult:
        """Compute displacement between two consecutive substrate maps.

        Args:
            substrate_t0: Classified substrate map at time t0 (uint8)
            substrate_t1: Classified substrate map at time t1 (uint8)
            date_t0, date_t1: Date strings (YYYY-MM-DD)
            transform: rasterio Affine for georeferencing
        """
        # Valid pixels in both scenes
        valid = (substrate_t0 != NODATA) & (substrate_t1 != NODATA)

        # Burial: rock/mixed → sand
        burial = valid & np.isin(substrate_t0, [ROCK, MIXED]) & (substrate_t1 == SAND)
        # Exposure: sand → rock/mixed
        exposure = valid & (substrate_t0 == SAND) & np.isin(substrate_t1, [ROCK, MIXED])

        n_burial = int(np.sum(burial))
        n_exposure = int(np.sum(exposure))

        burial_area = n_burial * self.pixel_area_m2
        exposure_area = n_exposure * self.pixel_area_m2

        # Compute centroids of change zones
        burial_centroids = self._get_centroids(burial, transform)
        exposure_centroids = self._get_centroids(exposure, transform)

        # Mean direction of sand advance
        direction = self._compute_advance_direction(burial, transform)

        # Days between scenes
        from datetime import datetime
        d0 = datetime.strptime(date_t0[:10], '%Y-%m-%d')
        d1 = datetime.strptime(date_t1[:10], '%Y-%m-%d')
        days = (d1 - d0).days

        return DisplacementResult(
            date_from=date_t0,
            date_to=date_t1,
            days_between=max(days, 1),
            burial_area_m2=burial_area,
            exposure_area_m2=exposure_area,
            net_change_m2=burial_area - exposure_area,
            sand_direction_deg=direction,
            n_burial_pixels=n_burial,
            n_exposure_pixels=n_exposure,
            burial_centroids=burial_centroids,
            exposure_centroids=exposure_centroids
        )

    def compute_timeseries(self, classifications: List[Dict],
                           transform=None) -> List[DisplacementResult]:
        """Compute displacement for all consecutive scene pairs.

        Args:
            classifications: Sorted list of {'date': str, 'substrate_map': ndarray}
        """
        results = []
        for i in range(len(classifications) - 1):
            t0 = classifications[i]
            t1 = classifications[i + 1]
            result = self.compute_displacement(
                t0['substrate_map'], t1['substrate_map'],
                t0['date'], t1['date'], transform
            )
            results.append(result)

            if (i + 1) % 50 == 0:
                print(f"      Desplazamiento: {i+1}/{len(classifications)-1} pares procesados")

        return results

    def compute_pixel_metrics(self, classifications: List[Dict],
                              row: int, col: int,
                              lat: float, lon: float,
                              reference_date: str = None) -> PixelDisplacementMetrics:
        """Compute displacement metrics for a single pixel over time.

        Args:
            classifications: Sorted list of {'date': str, 'substrate_map': ndarray}
            row, col: Pixel indices
            reference_date: Date to compute days_since_last_change from
        """
        from datetime import datetime

        values = []
        dates = []
        for clf in classifications:
            h, w = clf['substrate_map'].shape
            if 0 <= row < h and 0 <= col < w:
                val = clf['substrate_map'][row, col]
                if val != NODATA:
                    values.append(val)
                    dates.append(clf['date'])

        if len(values) < 2:
            return self._empty_metrics(lat, lon)

        # Track transitions
        burial_events = 0
        exposure_events = 0
        burial_durations = []
        current_burial_start = None
        last_transition = dates[0]
        directions = []

        for i in range(1, len(values)):
            prev, curr = values[i-1], values[i]

            # Rock/mixed → sand = burial
            if prev in (ROCK, MIXED) and curr == SAND:
                burial_events += 1
                current_burial_start = dates[i]
                last_transition = dates[i]

            # Sand → rock/mixed = exposure
            elif prev == SAND and curr in (ROCK, MIXED):
                exposure_events += 1
                if current_burial_start:
                    d_start = datetime.strptime(current_burial_start[:10], '%Y-%m-%d')
                    d_end = datetime.strptime(dates[i][:10], '%Y-%m-%d')
                    burial_durations.append((d_end - d_start).days)
                    current_burial_start = None
                last_transition = dates[i]

        # Days since last change
        if reference_date:
            ref = datetime.strptime(reference_date[:10], '%Y-%m-%d')
        else:
            ref = datetime.strptime(dates[-1][:10], '%Y-%m-%d')
        last_trans_dt = datetime.strptime(last_transition[:10], '%Y-%m-%d')
        days_since = (ref - last_trans_dt).days

        # Mean burial duration
        mean_duration = float(np.mean(burial_durations)) if burial_durations else 0.0
        max_duration = float(np.max(burial_durations)) if burial_durations else 0.0

        # Annual sand advance rate
        total_days = (datetime.strptime(dates[-1][:10], '%Y-%m-%d') -
                      datetime.strptime(dates[0][:10], '%Y-%m-%d')).days
        years = max(total_days / 365.25, 0.1)
        rate = (burial_events * self.pixel_area_m2) / years

        return PixelDisplacementMetrics(
            lat=lat, lon=lon,
            burial_events_count=burial_events,
            exposure_events_count=exposure_events,
            mean_burial_duration_days=round(mean_duration, 1),
            max_burial_duration_days=round(max_duration, 1),
            last_transition_date=last_transition,
            days_since_last_change=days_since,
            sand_advance_direction_deg=0.0,  # Computed from spatial context
            sand_advance_rate_m2_yr=round(rate, 1)
        )

    def generate_sand_corridors(self, displacements: List[DisplacementResult],
                                min_events: int = 10) -> List[Dict]:
        """Identify consistent sand movement corridors from displacement series.

        Returns GeoJSON-compatible features for sand corridors.
        """
        if not displacements:
            return []

        # Aggregate burial centroids by location
        from collections import defaultdict
        location_events = defaultdict(list)

        for d in displacements:
            for lat, lon in d.burial_centroids:
                # Grid to ~200m cells
                grid_lat = round(lat, 3)
                grid_lon = round(lon, 3)
                location_events[(grid_lat, grid_lon)].append({
                    'date': d.date_to,
                    'direction': d.sand_direction_deg,
                    'area': d.burial_area_m2
                })

        corridors = []
        for (lat, lon), events in location_events.items():
            if len(events) < min_events:
                continue

            directions = [e['direction'] for e in events if e['direction'] != 0]
            if not directions:
                continue

            # Circular mean direction
            sin_sum = np.sum(np.sin(np.radians(directions)))
            cos_sum = np.sum(np.cos(np.radians(directions)))
            mean_dir = np.degrees(np.arctan2(sin_sum, cos_sum)) % 360

            # Direction consistency (0 = random, 1 = perfectly consistent)
            r = np.sqrt(sin_sum**2 + cos_sum**2) / len(directions)

            if r > 0.3:  # At least moderately consistent direction
                corridors.append({
                    'type': 'Feature',
                    'geometry': {
                        'type': 'Point',
                        'coordinates': [lon, lat]
                    },
                    'properties': {
                        'mean_direction_deg': round(mean_dir, 1),
                        'consistency': round(r, 3),
                        'n_events': len(events),
                        'total_area_m2': sum(e['area'] for e in events),
                        'first_event': events[0]['date'],
                        'last_event': events[-1]['date']
                    }
                })

        return corridors

    def _get_centroids(self, mask: np.ndarray,
                       transform=None) -> List[Tuple[float, float]]:
        """Get centroids of connected regions in a binary mask."""
        if not np.any(mask):
            return []

        try:
            from scipy.ndimage import label
            labeled, n_features = label(mask)
        except ImportError:
            # Fallback: single centroid of all True pixels
            rows, cols = np.where(mask)
            if len(rows) == 0:
                return []
            r, c = np.mean(rows), np.mean(cols)
            if transform:
                lon, lat = transform * (c, r)
                return [(lat, lon)]
            return [(r, c)]

        centroids = []
        for i in range(1, n_features + 1):
            rows, cols = np.where(labeled == i)
            r, c = np.mean(rows), np.mean(cols)
            if transform:
                lon, lat = transform * (c, r)
                centroids.append((lat, lon))
            else:
                centroids.append((r, c))

        return centroids

    def _compute_advance_direction(self, burial_mask: np.ndarray,
                                   transform=None) -> float:
        """Compute mean direction of sand advance from burial pixels."""
        if not np.any(burial_mask):
            return 0.0

        rows, cols = np.where(burial_mask)
        if len(rows) < 2:
            return 0.0

        # Direction from centroid of burial zone relative to coast (assumed east)
        # Simple approximation: direction from mean position
        if transform:
            center_lon, center_lat = transform * (np.mean(cols), np.mean(rows))
            # For Peru south coast, sand generally moves north-south along coast
            # Use the spread of burial pixels to estimate direction
            lons = np.array([transform * (c, r) for r, c in zip(rows, cols)])
            if len(lons) > 1:
                dlat = np.std([p[1] for p in lons])
                dlon = np.std([p[0] for p in lons])
                return float(np.degrees(np.arctan2(dlon, dlat))) % 360

        return 0.0

    @staticmethod
    def _empty_metrics(lat: float, lon: float) -> PixelDisplacementMetrics:
        return PixelDisplacementMetrics(
            lat=lat, lon=lon,
            burial_events_count=0, exposure_events_count=0,
            mean_burial_duration_days=0.0, max_burial_duration_days=0.0,
            last_transition_date='', days_since_last_change=0,
            sand_advance_direction_deg=0.0, sand_advance_rate_m2_yr=0.0
        )
