"""
Temporal stability analysis from Sentinel-2 classification time series.

Computes per-pixel stability metrics over 10 years (2015-2025):
- Substrate frequency (rock/sand/water per pixel)
- Seasonal patterns (DJF vs JJA)
- ENSO-aware metrics (El Niño vs La Niña)
- Waterline trends (accretion/erosion rate m/year)

References:
- Sánchez-García et al. 2024 (beach rotation detection)
- Intertidal monitoring 2025 (93.6% classification accuracy)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from core.sentinel2.spectral_classifier import ROCK, SAND, MIXED, WATER, NODATA

# ENSO periods for Peru south coast
ELNINO_PERIODS = [
    ('2015-03', '2016-06'),  # Strong El Niño
    ('2023-06', '2024-04'),  # Moderate El Niño
]
LANINA_PERIODS = [
    ('2020-08', '2022-12'),  # Extended La Niña
]

# Seasons (Southern Hemisphere)
SEASON_MONTHS = {
    'DJF': {12, 1, 2},   # Summer
    'MAM': {3, 4, 5},    # Autumn
    'JJA': {6, 7, 8},    # Winter
    'SON': {9, 10, 11},  # Spring
}


@dataclass
class PixelStability:
    """Temporal stability metrics for a single pixel over 10 years."""
    lat: float
    lon: float
    dominant_substrate: str
    stability_score: float
    rock_frequency: float
    sand_frequency: float
    water_frequency: float
    mixed_frequency: float
    is_seasonal_burial: bool
    elnino_substrate: str
    lanina_substrate: str
    mean_waterline_distance_m: float
    waterline_std_m: float
    waterline_trend_m_yr: float
    n_observations: int
    first_observation: str
    last_observation: str


class TemporalAnalyzer:
    """Analyzes substrate classification time series for stability patterns."""

    def compute_pixel_stability(self, classifications: List[Dict],
                                lat: float, lon: float,
                                row: int, col: int) -> PixelStability:
        """Compute stability metrics for a single pixel across all scenes.

        Args:
            classifications: List of {'date': str, 'substrate_map': ndarray}
            lat, lon: Geographic coordinates of pixel
            row, col: Pixel indices in substrate maps
        """
        # Collect substrate values across time
        values = []
        dates = []
        for clf in classifications:
            h, w = clf['substrate_map'].shape
            if 0 <= row < h and 0 <= col < w:
                val = clf['substrate_map'][row, col]
                if val != NODATA:
                    values.append(val)
                    dates.append(clf['date'])

        n_obs = len(values)
        if n_obs == 0:
            return self._empty_stability(lat, lon)

        values = np.array(values)
        code_to_name = {ROCK: 'rock', SAND: 'sand', MIXED: 'mixed', WATER: 'water'}

        # Frequencies
        rock_freq = np.mean(values == ROCK)
        sand_freq = np.mean(values == SAND)
        water_freq = np.mean(values == WATER)
        mixed_freq = np.mean(values == MIXED)

        # Dominant substrate
        freqs = {'rock': rock_freq, 'sand': sand_freq, 'mixed': mixed_freq, 'water': water_freq}
        dominant = max(freqs, key=freqs.get)

        # Stability score
        stability = max(freqs.values())

        # Seasonal burial detection
        summer_vals = [v for v, d in zip(values, dates)
                       if self._get_month(d) in SEASON_MONTHS['DJF']]
        winter_vals = [v for v, d in zip(values, dates)
                       if self._get_month(d) in SEASON_MONTHS['JJA']]

        is_burial = False
        if summer_vals and winter_vals:
            summer_rock = np.mean(np.array(summer_vals) == ROCK)
            winter_rock = np.mean(np.array(winter_vals) == ROCK)
            # Seasonal burial: rock in summer, sand in winter
            is_burial = (summer_rock > 0.5) and (winter_rock < 0.3)

        # ENSO substrate
        elnino_vals = [v for v, d in zip(values, dates)
                       if self._is_in_period(d, ELNINO_PERIODS)]
        lanina_vals = [v for v, d in zip(values, dates)
                       if self._is_in_period(d, LANINA_PERIODS)]

        elnino_sub = self._mode_substrate(elnino_vals) if elnino_vals else dominant
        lanina_sub = self._mode_substrate(lanina_vals) if lanina_vals else dominant

        return PixelStability(
            lat=lat, lon=lon,
            dominant_substrate=dominant,
            stability_score=round(stability, 3),
            rock_frequency=round(rock_freq, 3),
            sand_frequency=round(sand_freq, 3),
            water_frequency=round(water_freq, 3),
            mixed_frequency=round(mixed_freq, 3),
            is_seasonal_burial=is_burial,
            elnino_substrate=elnino_sub,
            lanina_substrate=lanina_sub,
            mean_waterline_distance_m=0.0,  # Computed separately
            waterline_std_m=0.0,
            waterline_trend_m_yr=0.0,
            n_observations=n_obs,
            first_observation=dates[0],
            last_observation=dates[-1]
        )

    def compute_stability_map(self, classifications: List[Dict],
                              transform) -> pd.DataFrame:
        """Compute stability metrics for all coastal pixels.

        Args:
            classifications: List of {'date': str, 'substrate_map': ndarray}
            transform: rasterio Affine transform

        Returns:
            DataFrame with one row per pixel, columns matching PixelStability
        """
        if not classifications:
            return pd.DataFrame()

        ref_shape = classifications[0]['substrate_map'].shape
        h, w = ref_shape

        # Find coastal pixels (pixels that are sometimes water, sometimes land)
        ever_water = np.zeros(ref_shape, dtype=bool)
        ever_land = np.zeros(ref_shape, dtype=bool)
        for clf in classifications:
            smap = clf['substrate_map']
            ever_water |= (smap == WATER)
            ever_land |= ((smap == ROCK) | (smap == SAND) | (smap == MIXED))

        # Coastal pixels: those that switch between water and land
        # Plus a buffer of land pixels near the coast
        coastal = ever_water & ever_land
        # Also include purely land pixels adjacent to water (habitat zone)
        from scipy.ndimage import binary_dilation
        coastal_expanded = binary_dilation(coastal, iterations=5) & ever_land

        rows_of_interest, cols_of_interest = np.where(coastal_expanded)
        print(f"[INFO] Sentinel-2: {len(rows_of_interest)} pixels costeros a analizar")

        results = []
        for i, (r, c) in enumerate(zip(rows_of_interest, cols_of_interest)):
            lon, lat = transform * (c + 0.5, r + 0.5)
            stability = self.compute_pixel_stability(classifications, lat, lon, r, c)
            results.append(stability.__dict__)

            if (i + 1) % 10000 == 0:
                print(f"      {i+1}/{len(rows_of_interest)} pixels procesados")

        return pd.DataFrame(results)

    def generate_seasonal_composite(self, classifications: List[Dict],
                                    season: str, year: int) -> Optional[np.ndarray]:
        """Generate mode substrate composite for a season/year.

        Args:
            season: 'DJF', 'MAM', 'JJA', or 'SON'
            year: Year (DJF uses Dec of previous year)
        """
        months = SEASON_MONTHS[season]
        matching = []

        for clf in classifications:
            d = clf['date']
            m = self._get_month(d)
            y = int(d[:4])

            if season == 'DJF':
                if (m == 12 and y == year - 1) or (m in {1, 2} and y == year):
                    matching.append(clf['substrate_map'])
            elif m in months and y == year:
                matching.append(clf['substrate_map'])

        if not matching:
            return None

        # Stack and compute mode per pixel
        stack = np.stack(matching, axis=0)
        from scipy.stats import mode as scipy_mode
        mode_result = scipy_mode(stack, axis=0, keepdims=False)
        return mode_result.mode.astype(np.uint8)

    def compute_waterline_trends(self, transect_timeseries: pd.DataFrame) -> pd.DataFrame:
        """Compute waterline migration trends per transect.

        Args:
            transect_timeseries: DataFrame with columns:
                transect_id, date, distance_m

        Returns:
            DataFrame with trend per transect (m/year)
        """
        results = []
        for t_id, group in transect_timeseries.groupby('transect_id'):
            if len(group) < 6:
                continue

            dates = pd.to_datetime(group['date'])
            days = (dates - dates.min()).dt.days.values.astype(float)
            distances = group['distance_m'].values

            if np.std(distances) < 0.1:
                trend = 0.0
            else:
                # Linear regression
                coeffs = np.polyfit(days, distances, 1)
                trend = coeffs[0] * 365.25  # m/year

            results.append({
                'transect_id': t_id,
                'mean_distance_m': float(np.mean(distances)),
                'std_distance_m': float(np.std(distances)),
                'trend_m_yr': round(trend, 3),
                'n_observations': len(group)
            })

        return pd.DataFrame(results)

    @staticmethod
    def _get_month(date_str: str) -> int:
        return int(date_str[5:7])

    @staticmethod
    def _is_in_period(date_str: str, periods: List[Tuple[str, str]]) -> bool:
        d = date_str[:7]  # YYYY-MM
        for start, end in periods:
            if start <= d <= end:
                return True
        return False

    @staticmethod
    def _mode_substrate(values: list) -> str:
        code_to_name = {ROCK: 'rock', SAND: 'sand', MIXED: 'mixed', WATER: 'water'}
        arr = np.array(values)
        counts = {name: np.sum(arr == code) for code, name in code_to_name.items()}
        return max(counts, key=counts.get)

    @staticmethod
    def _empty_stability(lat: float, lon: float) -> PixelStability:
        return PixelStability(
            lat=lat, lon=lon, dominant_substrate='unknown',
            stability_score=0.0, rock_frequency=0.0, sand_frequency=0.0,
            water_frequency=0.0, mixed_frequency=0.0, is_seasonal_burial=False,
            elnino_substrate='unknown', lanina_substrate='unknown',
            mean_waterline_distance_m=0.0, waterline_std_m=0.0,
            waterline_trend_m_yr=0.0, n_observations=0,
            first_observation='', last_observation=''
        )
