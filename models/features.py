"""
Advanced Feature Extraction for Fishing Zone Prediction.

Based on state-of-the-art research:
- Belkin-O'Reilly (2009): Thermal front detection
- INCOIS PFZ methodology: Multi-parameter approach
- Humboldt Current research: SST 14-24°C optimal for anchoveta
- Upwelling dynamics: Wind-driven nutrient concentration
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import math

# Import domain constants
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from domain import (
    HOTSPOTS,
    SPECIES_BY_NAME,
    THRESHOLDS,
    FEATURE_NAMES,
    N_FEATURES,
)


@dataclass
class MarineFeatures:
    """Complete feature set for a marine point."""
    # Location
    lat: float
    lon: float

    # Primary oceanographic
    sst: float
    sst_anomaly: float
    current_speed: float
    current_direction: float
    wave_height: float
    wave_period: float

    # Derived spatial
    distance_to_coast: float
    depth_proxy: float  # Estimated from distance

    # Thermal front indicators (Belkin-O'Reilly)
    gradient_magnitude: float
    gradient_direction: float
    is_thermal_front: bool
    front_intensity: float

    # Upwelling indicators
    upwelling_index: float
    ekman_transport: float

    # Current dynamics
    current_convergence: float
    current_shear: float

    # Productivity proxy
    productivity_index: float

    # Historical pattern
    historical_hotspot_distance: float
    hotspot_similarity: float

    # Temporal
    hour_sin: float
    hour_cos: float
    day_of_year_sin: float
    day_of_year_cos: float
    moon_phase: float
    is_major_period: bool


class FeatureExtractor:
    """
    State-of-the-art feature extraction for fishing prediction.

    Features based on:
    1. SST patterns (NOAA OISST methodology)
    2. Thermal fronts (Cayula-Cornillon SIED, Belkin-O'Reilly BOA)
    3. Upwelling dynamics (Ekman transport)
    4. Current patterns (convergence zones)
    5. Temporal cycles (solunar, seasonal)
    6. Historical patterns (known hotspots)

    Uses centralized domain constants from domain.py
    """

    def __init__(self):
        self.features_cache: List[MarineFeatures] = []
        self.feature_matrix: Optional[np.ndarray] = None
        self.sst_field: Dict[Tuple[float, float], float] = {}
        self.current_field: Dict[Tuple[float, float], Tuple[float, float]] = {}

        # Use centralized feature names from domain.py
        self.feature_names = list(FEATURE_NAMES)

    # Domain constants - delegated to domain.py
    @property
    def SST_OPTIMAL(self) -> Dict[str, float]:
        return {
            'min': THRESHOLDS.sst_optimal_min,
            'max': THRESHOLDS.sst_optimal_max,
            'center': THRESHOLDS.sst_optimal_center
        }

    @property
    def GRADIENT_THRESHOLD(self) -> float:
        return THRESHOLDS.gradient_weak

    @property
    def STRONG_FRONT_THRESHOLD(self) -> float:
        return THRESHOLDS.gradient_strong

    def extract_from_marine_points(
        self,
        marine_points: List,
        coastline_points: List[Tuple[float, float]],
        solunar_data: Optional[Dict] = None,
        weather_data: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Extract comprehensive feature matrix from marine data.

        Args:
            marine_points: List of MarinePoint objects
            coastline_points: Coastline coordinates
            solunar_data: Optional solunar conditions
            weather_data: Optional weather conditions

        Returns:
            Feature matrix (n_samples, 32 features)
        """
        self.features_cache = []
        self._build_sst_field(marine_points)
        self._build_current_field(marine_points)

        features_list = []
        now = datetime.now()

        for point in marine_points:
            if point.sst is None:
                continue

            # Extract all features
            feat = self._extract_point_features(
                point, coastline_points, solunar_data, weather_data, now
            )
            self.features_cache.append(feat)
            features_list.append(self._to_vector(feat))

        self.feature_matrix = np.array(features_list) if features_list else np.array([])
        return self.feature_matrix

    def _extract_point_features(
        self,
        point,
        coastline: List[Tuple[float, float]],
        solunar: Optional[Dict],
        weather: Optional[Dict],
        now: datetime
    ) -> MarineFeatures:
        """Extract all features for a single point."""

        # Distance to coast
        dist_coast = self._min_distance_to_coast(point.lat, point.lon, coastline)

        # Depth proxy (deeper offshore)
        depth_proxy = min(200, dist_coast / 50)  # Rough estimate

        # Thermal front analysis (Belkin-O'Reilly)
        gradient_mag, gradient_dir = self._calculate_gradient(point.lat, point.lon)
        is_front = gradient_mag >= self.GRADIENT_THRESHOLD
        front_intensity = min(1.0, gradient_mag / self.STRONG_FRONT_THRESHOLD)

        # Current dynamics
        convergence = self._calculate_convergence(point.lat, point.lon)
        shear = self._calculate_shear(point.lat, point.lon)

        # Upwelling index (simplified Ekman transport)
        upwelling, ekman = self._calculate_upwelling(
            point.lat, point.lon,
            weather.get('wind_speed', 10) if weather else 10,
            weather.get('wind_direction', 180) if weather else 180
        )

        # Productivity proxy (SST + upwelling + fronts)
        productivity = self._estimate_productivity(
            point.sst, upwelling, front_intensity
        )

        # Historical hotspot analysis
        hotspot_dist, hotspot_sim = self._analyze_hotspots(point.lat, point.lon, point.sst)

        # Temporal features
        hour_sin = np.sin(2 * np.pi * now.hour / 24)
        hour_cos = np.cos(2 * np.pi * now.hour / 24)
        doy = now.timetuple().tm_yday
        doy_sin = np.sin(2 * np.pi * doy / 365)
        doy_cos = np.cos(2 * np.pi * doy / 365)

        # Handle moon_illumination (may be "50%", 50, or 0.5)
        moon_raw = solunar.get('moon_illumination', 50) if solunar else 50
        try:
            if isinstance(moon_raw, str):
                moon_raw = moon_raw.replace('%', '').strip()
            moon_phase = float(moon_raw)
            if moon_phase > 1:  # Percentage format
                moon_phase /= 100
        except (ValueError, TypeError):
            moon_phase = 0.5
        moon_phase = max(0, min(1, moon_phase))  # Clamp to [0, 1]
        is_major = self._is_major_period(now, solunar)

        return MarineFeatures(
            lat=point.lat,
            lon=point.lon,
            sst=point.sst,
            sst_anomaly=abs(point.sst - self.SST_OPTIMAL['center']),
            current_speed=point.current_speed or 0,
            current_direction=point.current_direction or 0,
            wave_height=point.wave_height or 1.0,
            wave_period=point.wave_period or 8.0,
            distance_to_coast=dist_coast,
            depth_proxy=depth_proxy,
            gradient_magnitude=gradient_mag,
            gradient_direction=gradient_dir,
            is_thermal_front=is_front,
            front_intensity=front_intensity,
            upwelling_index=upwelling,
            ekman_transport=ekman,
            current_convergence=convergence,
            current_shear=shear,
            productivity_index=productivity,
            historical_hotspot_distance=hotspot_dist,
            hotspot_similarity=hotspot_sim,
            hour_sin=hour_sin,
            hour_cos=hour_cos,
            day_of_year_sin=doy_sin,
            day_of_year_cos=doy_cos,
            moon_phase=moon_phase,
            is_major_period=is_major
        )

    def _to_vector(self, feat: MarineFeatures) -> List[float]:
        """Convert MarineFeatures to 32-element vector."""
        # Current components
        rad = np.radians(feat.current_direction)
        current_u = feat.current_speed * np.sin(rad)
        current_v = feat.current_speed * np.cos(rad)

        # SST scores
        sst_optimal = self._sst_optimal_score(feat.sst)
        sst_species = self._sst_species_score(feat.sst)

        # SST variability (from gradient)
        sst_variability = feat.gradient_magnitude * 10

        # SST trend (simplified - using gradient direction toward coast)
        sst_trend = np.cos(np.radians(feat.gradient_direction - 270)) * feat.gradient_magnitude

        # Wave favorable
        wave_favorable = 1.0 if feat.wave_height < 2.0 else 0.0

        # Upwelling favorable
        upwelling_favorable = 1.0 if feat.upwelling_index > 0.3 else 0.0

        # Current toward coast (west = toward coast in Peru)
        current_toward = max(0, -current_u)  # Negative u = toward west/coast

        # Spatial zones
        coastal_zone = 1.0 if feat.distance_to_coast < 5000 else 0.0
        offshore_zone = 1.0 if feat.distance_to_coast > 15000 else 0.0

        # Temporal scores
        hour_score = self._hour_score(feat.hour_sin, feat.hour_cos)
        moon_score = self._moon_score(feat.moon_phase)
        season_score = self._season_score(feat.day_of_year_sin, feat.day_of_year_cos)

        return [
            # SST features (6)
            feat.sst,
            feat.sst_anomaly,
            sst_optimal,
            sst_species,
            sst_variability,
            sst_trend,
            # Thermal front features (5)
            feat.gradient_magnitude * 100,  # Scale
            np.sin(np.radians(feat.gradient_direction)),
            np.cos(np.radians(feat.gradient_direction)),
            1.0 if feat.is_thermal_front else 0.0,
            feat.front_intensity,
            # Current features (6)
            feat.current_speed,
            current_u,
            current_v,
            feat.current_convergence,
            feat.current_shear,
            current_toward,
            # Wave features (3)
            feat.wave_height,
            feat.wave_period,
            wave_favorable,
            # Upwelling features (3)
            feat.upwelling_index,
            feat.ekman_transport,
            upwelling_favorable,
            # Spatial features (4)
            feat.distance_to_coast / 1000,  # km
            feat.depth_proxy,
            coastal_zone,
            offshore_zone,
            # Historical features (2)
            feat.historical_hotspot_distance / 1000,  # km
            feat.hotspot_similarity,
            # Temporal features (3)
            hour_score,
            moon_score,
            season_score
        ]

    # === SST Analysis ===

    def _sst_optimal_score(self, sst: float) -> float:
        """Score based on optimal SST range."""
        if self.SST_OPTIMAL['min'] <= sst <= self.SST_OPTIMAL['max']:
            # Gaussian around center
            center = self.SST_OPTIMAL['center']
            sigma = 2.5
            return np.exp(-((sst - center) ** 2) / (2 * sigma ** 2))
        return 0.1  # Penalty for out of range

    def _sst_species_score(self, sst: float) -> float:
        """Score based on species-specific optimal SST using domain.py species data."""
        scores = [sp.temp_score(sst) for sp in SPECIES_BY_NAME.values()]
        return max(scores) if scores else 0.1

    # === Thermal Front Detection (Belkin-O'Reilly) ===

    def _build_sst_field(self, points: List):
        """Build SST field for gradient calculation."""
        self.sst_field = {}
        for p in points:
            if p.sst is not None:
                key = (round(p.lat, 2), round(p.lon, 2))
                self.sst_field[key] = p.sst

    def _calculate_gradient(self, lat: float, lon: float, delta: float = 0.1) -> Tuple[float, float]:
        """
        Calculate SST gradient using Sobel-like operator.
        Based on Belkin-O'Reilly (2009) methodology.
        """
        key = (round(lat, 2), round(lon, 2))
        if key not in self.sst_field:
            return 0.0, 0.0

        # Get neighboring values
        neighbors = {
            'n': (round(lat + delta, 2), round(lon, 2)),
            's': (round(lat - delta, 2), round(lon, 2)),
            'e': (round(lat, 2), round(lon + delta, 2)),
            'w': (round(lat, 2), round(lon - delta, 2)),
        }

        # Calculate gradients
        dx, dy = 0.0, 0.0
        center = self.sst_field[key]

        if neighbors['e'] in self.sst_field and neighbors['w'] in self.sst_field:
            dx = (self.sst_field[neighbors['e']] - self.sst_field[neighbors['w']]) / (2 * delta * 111)
        if neighbors['n'] in self.sst_field and neighbors['s'] in self.sst_field:
            dy = (self.sst_field[neighbors['n']] - self.sst_field[neighbors['s']]) / (2 * delta * 111)

        magnitude = np.sqrt(dx**2 + dy**2)
        direction = np.degrees(np.arctan2(dx, dy)) % 360

        return magnitude, direction

    # === Current Dynamics ===

    def _build_current_field(self, points: List):
        """Build current vector field."""
        self.current_field = {}
        for p in points:
            if p.current_speed is not None and p.current_direction is not None:
                key = (round(p.lat, 2), round(p.lon, 2))
                rad = np.radians(p.current_direction)
                u = p.current_speed * np.sin(rad)
                v = p.current_speed * np.cos(rad)
                self.current_field[key] = (u, v)

    def _calculate_convergence(self, lat: float, lon: float, delta: float = 0.1) -> float:
        """
        Calculate current convergence (negative divergence).
        Convergence zones attract fish.
        """
        key = (round(lat, 2), round(lon, 2))
        if key not in self.current_field:
            return 0.0

        neighbors = {
            'e': (round(lat, 2), round(lon + delta, 2)),
            'w': (round(lat, 2), round(lon - delta, 2)),
            'n': (round(lat + delta, 2), round(lon, 2)),
            's': (round(lat - delta, 2), round(lon, 2)),
        }

        du_dx, dv_dy = 0.0, 0.0

        if neighbors['e'] in self.current_field and neighbors['w'] in self.current_field:
            du_dx = (self.current_field[neighbors['e']][0] - self.current_field[neighbors['w']][0]) / (2 * delta)
        if neighbors['n'] in self.current_field and neighbors['s'] in self.current_field:
            dv_dy = (self.current_field[neighbors['n']][1] - self.current_field[neighbors['s']][1]) / (2 * delta)

        divergence = du_dx + dv_dy
        return -divergence  # Convergence is negative divergence

    def _calculate_shear(self, lat: float, lon: float, delta: float = 0.1) -> float:
        """Calculate current shear (vorticity)."""
        key = (round(lat, 2), round(lon, 2))
        if key not in self.current_field:
            return 0.0

        neighbors = {
            'e': (round(lat, 2), round(lon + delta, 2)),
            'w': (round(lat, 2), round(lon - delta, 2)),
            'n': (round(lat + delta, 2), round(lon, 2)),
            's': (round(lat - delta, 2), round(lon, 2)),
        }

        dv_dx, du_dy = 0.0, 0.0

        if neighbors['e'] in self.current_field and neighbors['w'] in self.current_field:
            dv_dx = (self.current_field[neighbors['e']][1] - self.current_field[neighbors['w']][1]) / (2 * delta)
        if neighbors['n'] in self.current_field and neighbors['s'] in self.current_field:
            du_dy = (self.current_field[neighbors['n']][0] - self.current_field[neighbors['s']][0]) / (2 * delta)

        vorticity = dv_dx - du_dy
        return abs(vorticity)

    # === Upwelling Analysis ===

    def _calculate_upwelling(
        self,
        lat: float,
        lon: float,
        wind_speed: float,
        wind_direction: float
    ) -> Tuple[float, float]:
        """
        Calculate upwelling index based on Ekman transport.

        In Peru (southern hemisphere), upwelling occurs when wind
        blows from south (parallel to coast), causing offshore
        Ekman transport and upwelling of cold, nutrient-rich water.
        """
        # Handle None values with regional defaults
        if wind_speed is None:
            wind_speed = 5.0  # Moderate wind default for Peru coast
        if wind_direction is None:
            wind_direction = 180.0  # South wind default

        # Coriolis parameter
        omega = 7.2921e-5  # Earth's rotation rate
        f = 2 * omega * np.sin(np.radians(lat))

        # Wind stress (simplified)
        rho_air = 1.225  # kg/m³
        Cd = 1.3e-3  # Drag coefficient
        tau = rho_air * Cd * wind_speed**2

        # Ekman transport (perpendicular to wind, to the left in SH)
        # Positive = offshore = upwelling favorable
        wind_rad = np.radians(wind_direction)

        # Coast orientation (roughly N-S for Peru)
        coast_angle = 180  # South

        # Angle difference
        angle_diff = (wind_direction - coast_angle + 360) % 360

        # Upwelling favorable when wind parallel to coast (from south)
        # causing Ekman transport offshore
        if 135 <= angle_diff <= 225:  # Wind roughly from south
            upwelling_index = tau * abs(np.sin(np.radians(angle_diff - 180))) / 0.1
        else:
            upwelling_index = 0.0

        ekman_transport = tau / abs(f) if f != 0 else 0.0

        return min(1.0, upwelling_index), ekman_transport

    # === Productivity ===

    def _estimate_productivity(
        self,
        sst: float,
        upwelling: float,
        front_intensity: float
    ) -> float:
        """
        Estimate productivity index.
        Based on: cold SST + upwelling + fronts = high productivity
        """
        # Cold water bonus (upwelled water is cold and nutrient-rich)
        cold_bonus = max(0, (18 - sst) / 4) if sst < 18 else 0

        # Combined productivity
        productivity = (
            0.4 * upwelling +
            0.3 * front_intensity +
            0.3 * cold_bonus
        )

        return min(1.0, productivity)

    # === Historical Patterns ===

    def _analyze_hotspots(self, lat: float, lon: float, sst: float) -> Tuple[float, float]:
        """Analyze proximity and similarity to historical hotspots from domain.py."""
        min_dist = float('inf')
        max_similarity = 0.0

        for hotspot in HOTSPOTS:
            dist = hotspot.distance_to(lat, lon)
            min_dist = min(min_dist, dist)

            # Similarity based on distance and SST match
            if dist < 10_000:  # Within 10km
                proximity = 1 - (dist / 10_000)
                sst_match = self._sst_optimal_score(sst)
                similarity = proximity * sst_match * hotspot.bonus_factor
                max_similarity = max(max_similarity, similarity)

        return min_dist, max_similarity

    # === Temporal Features ===

    def _hour_score(self, hour_sin: float, hour_cos: float) -> float:
        """Score based on time of day. Dawn/dusk best."""
        # Peak at 6am and 6pm
        dawn_score = np.exp(-((hour_sin - 0.5)**2 + (hour_cos - 0.866)**2) / 0.5)
        dusk_score = np.exp(-((hour_sin - (-0.5))**2 + (hour_cos - 0.866)**2) / 0.5)
        return max(dawn_score, dusk_score)

    def _moon_score(self, moon_phase: float) -> float:
        """Score based on moon phase. New/full moon best."""
        # Best at new moon (0) and full moon (1)
        if moon_phase < 0.1 or moon_phase > 0.9:
            return 1.0
        elif 0.4 < moon_phase < 0.6:
            return 0.8  # Full moon also good
        return 0.5

    def _season_score(self, doy_sin: float, doy_cos: float) -> float:
        """Score based on season. Summer/fall best for Humboldt."""
        # In Peru, Feb-May typically best (after upwelling season)
        # DOY 32-152 approximately
        # This corresponds to doy_sin roughly 0.5-1.0 and doy_cos 0.5 to -0.5
        if doy_sin > 0.3 and doy_cos > -0.5:
            return 0.9
        return 0.6

    def _is_major_period(self, now: datetime, solunar: Optional[Dict]) -> bool:
        """Check if current time is in a major solunar period."""
        if not solunar:
            return False
        # Simplified: major periods around moonrise/moonset
        return solunar.get('is_major_period', False)

    # === Utilities ===

    def _min_distance_to_coast(self, lat: float, lon: float, coastline: List[Tuple]) -> float:
        if not coastline:
            return 10000.0
        return min(
            self._haversine(lat, lon, c_lat, c_lon)
            for c_lat, c_lon in coastline[::10]
        )

    def _haversine(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371000
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlam = np.radians(lon2 - lon1)
        a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam/2)**2
        return 2 * R * np.arcsin(np.sqrt(a))

    def get_feature_descriptions(self) -> Dict[str, str]:
        """Return descriptions for all features."""
        return {
            'sst': 'Temperatura superficial del mar (°C)',
            'sst_anomaly': 'Desviación del SST óptimo',
            'sst_optimal_score': 'Score SST en rango óptimo (0-1)',
            'sst_species_score': 'Score SST para especies objetivo',
            'sst_variability': 'Variabilidad espacial de SST',
            'sst_trend': 'Tendencia SST hacia la costa',
            'gradient_magnitude': 'Magnitud del gradiente térmico',
            'gradient_direction_sin': 'Dirección gradiente (sin)',
            'gradient_direction_cos': 'Dirección gradiente (cos)',
            'is_thermal_front': 'Es frente térmico (0/1)',
            'front_intensity': 'Intensidad del frente (0-1)',
            'current_speed': 'Velocidad de corriente (m/s)',
            'current_u': 'Componente E-O de corriente',
            'current_v': 'Componente N-S de corriente',
            'current_convergence': 'Convergencia de corrientes',
            'current_shear': 'Cizalladura de corrientes',
            'current_toward_coast': 'Corriente hacia la costa',
            'wave_height': 'Altura de olas (m)',
            'wave_period': 'Período de olas (s)',
            'wave_favorable': 'Condiciones de olas favorables',
            'upwelling_index': 'Índice de surgencia',
            'ekman_transport': 'Transporte de Ekman',
            'upwelling_favorable': 'Surgencia favorable',
            'distance_to_coast': 'Distancia a la costa (km)',
            'depth_proxy': 'Profundidad estimada (m)',
            'coastal_zone': 'En zona costera (<5km)',
            'offshore_zone': 'En zona oceánica (>15km)',
            'hotspot_distance': 'Distancia a hotspot histórico (km)',
            'hotspot_similarity': 'Similitud con hotspots',
            'hour_score': 'Score por hora del día',
            'moon_score': 'Score por fase lunar',
            'season_score': 'Score por temporada'
        }
