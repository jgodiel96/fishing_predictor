"""
Analysis Controller - Orchestrates data fetching, ML prediction, and visualization.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

# Import centralized configuration
from config import LEGACY_DB, STUDY_AREA
from domain import PERU_SOUTH_LIMIT

try:
    from data.data_config import DataConfig
except ImportError:
    DataConfig = None

from models.predictor import FishingPredictor
from models.features import FeatureExtractor
from models.timeline import TimelineAnalyzer, generate_timeline_data
from models.anchovy_migration import AnchovyMigrationModel, get_anchovy_predictions
from views.map_view import MapView, MapConfig
from core.marine_data import MarineDataFetcher, FishZonePredictor
from core.weather_solunar import get_fishing_conditions

# Tide calculations (V3)
try:
    from data.fetchers.tide_fetcher import TideFetcher
    TIDES_AVAILABLE = True
except ImportError:
    TIDES_AVAILABLE = False

# SSS and SLA from Copernicus (V4)
try:
    from data.fetchers.copernicus_physics_fetcher import CopernicusPhysicsFetcher
    PHYSICS_AVAILABLE = True
except ImportError:
    PHYSICS_AVAILABLE = False

# Chlorophyll-a from Copernicus (V7)
try:
    from data.fetchers.copernicus_chlorophyll_fetcher import ChlorophyllFetcher
    CHLA_AVAILABLE = True
except ImportError:
    CHLA_AVAILABLE = False

# Historical SST provider (V7)
try:
    from data.fetchers.sst_historical_provider import SSTHistoricalProvider
    SST_HISTORICAL_AVAILABLE = True
except ImportError:
    SST_HISTORICAL_AVAILABLE = False

# GFW Dynamic Hotspots (V7)
try:
    from data.fetchers.gfw_hotspot_generator import GFWHotspotGenerator
    GFW_HOTSPOTS_AVAILABLE = True
except ImportError:
    GFW_HOTSPOTS_AVAILABLE = False

# Optional: Historical data for supervised learning
try:
    from data.fetchers.historical_fetcher import HistoricalDataFetcher, FishMovementPredictor
    HISTORICAL_AVAILABLE = True
except ImportError:
    HISTORICAL_AVAILABLE = False

# Copernicus Data Provider (currents, waves, SST from downloaded data)
try:
    from core.copernicus_data_provider import CopernicusDataProvider, convert_to_marine_points
    COPERNICUS_DATA_AVAILABLE = True
except ImportError:
    COPERNICUS_DATA_AVAILABLE = False

# Timeline data availability
try:
    _db_path = LEGACY_DB if DataConfig is None else DataConfig.LEGACY_DB
    TIMELINE_AVAILABLE = _db_path.exists()
except:
    TIMELINE_AVAILABLE = False

# Hourly predictions generator (V6)
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
    from generate_hourly_predictions import HourlyPredictionGenerator
    HOURLY_PREDICTIONS_AVAILABLE = True
except ImportError:
    HOURLY_PREDICTIONS_AVAILABLE = False


class AnalysisController:
    """
    Main controller for fishing spot analysis.

    Responsibilities:
    - Load coastline data
    - Fetch marine data
    - Run ML predictions
    - Coordinate visualization
    """

    SPECIES_BY_SUBSTRATE = {
        "roca": ["Cabrilla", "Pintadilla", "Robalo"],
        "arena": ["Corvina", "Lenguado", "Pejerrey"],
        "mixto": ["Corvina", "Cabrilla", "Robalo"]
    }

    SPECIES_LURES = {
        "Cabrilla": "Grubs 3\", jigs 15-25g",
        "Pintadilla": "Vinilos pequenos, jigs ligeros",
        "Robalo": "Poppers 12cm, minnows",
        "Corvina": "Jigs metalicos 30-50g",
        "Lenguado": "Vinilos paddle tail",
        "Pejerrey": "Cucharillas pequenas"
    }

    def __init__(self):
        # Data
        self.coastline_points: List[Tuple[float, float]] = []
        self.sampled_spots: List[Dict] = []
        self.fish_zones: List[Dict] = []
        self.flow_lines: List[List[Tuple[float, float]]] = []
        self.current_vectors: List = []
        self.marine_points: List = []
        self.user_location: Optional[Dict] = None  # For proximity search

        # Components
        self.marine_fetcher: Optional[MarineDataFetcher] = None
        self.feature_extractor = FeatureExtractor()
        self.predictor = FishingPredictor(n_components=8, n_clusters=6)  # More components for 32 features
        self.map_view = MapView()

        # Fish movement prediction
        self.movement_predictor: Optional['FishMovementPredictor'] = None
        self.future_hotspots: List[Dict] = []

        # Anchovy migration model
        self.anchovy_model: Optional[AnchovyMigrationModel] = None
        self.anchovy_zones: List[Dict] = []

        # Tide data (V4)
        self.tide_fetcher: Optional['TideFetcher'] = None
        self.tide_data: Dict = {}

        # SSS and SLA data (V4)
        self.physics_fetcher: Optional['CopernicusPhysicsFetcher'] = None
        self.sss_score: float = 0.5  # Neutral fallback
        self.sla_score: float = 0.5  # Neutral fallback

        # Chlorophyll-a data (V7)
        self.chla_fetcher: Optional['ChlorophyllFetcher'] = None
        self.chla_score: float = 0.5  # Neutral fallback
        self.chla_value: Optional[float] = None

        # SST Historical data (V7)
        self.sst_provider: Optional['SSTHistoricalProvider'] = None
        self.sst_historical_score: float = 0.5  # Neutral fallback
        self.sst_anomaly: float = 0.0

        # GFW Dynamic Hotspots (V7)
        self.gfw_generator: Optional['GFWHotspotGenerator'] = None
        self.dynamic_hotspots: List = []
        self.gfw_bonus: float = 0.0

        # Analysis date (V4) - can be overridden for historical analysis
        self.analysis_datetime: datetime = datetime.now()

        # Historical data for supervised learning
        self.historical_fetcher: Optional['HistoricalDataFetcher'] = None
        self.is_supervised_mode: bool = False

        # Copernicus data provider (real oceanographic data)
        self.copernicus_provider: Optional['CopernicusDataProvider'] = None
        if COPERNICUS_DATA_AVAILABLE:
            self.copernicus_provider = CopernicusDataProvider()

        # Results
        self.ml_predictions: List = []
        self.pca_analysis: Dict = {}
        self.conditions: Dict = {}

    def load_coastline(self, geojson_path: str) -> int:
        """Load coastline from GeoJSON file.

        Supports both LineString and MultiLineString geometries.
        Preserves segment order (no sorting) to maintain coastline continuity.
        """
        with open(geojson_path, 'r') as f:
            data = json.load(f)

        self.coastline_points = []
        self.coastline_segments = []  # Store separate segments

        for feature in data.get('features', []):
            geom = feature.get('geometry', {})

            if geom.get('type') == 'LineString':
                segment = [(coord[1], coord[0]) for coord in geom.get('coordinates', [])]
                self.coastline_segments.append(segment)
                self.coastline_points.extend(segment)

            elif geom.get('type') == 'MultiLineString':
                for line in geom.get('coordinates', []):
                    segment = [(coord[1], coord[0]) for coord in line]
                    self.coastline_segments.append(segment)
                    self.coastline_points.extend(segment)

        # Remove duplicates within each segment while preserving order
        self.coastline_points = self._remove_duplicates(self.coastline_points)

        return len(self.coastline_points)

    def sample_fishing_spots(self, num_spots: int = None, spacing_m: int = 750, max_spots: int = 200) -> List[Dict]:
        """
        Sample spots along the coastline with specified spacing.

        Samples from each coastline segment separately to maintain geographic
        continuity and avoid creating connections between distant segments.

        Args:
            num_spots: Fixed number of spots (if provided, overrides spacing)
            spacing_m: Target spacing between spots in meters (default: 750m)
                      Use 500-1000m for high-resolution analysis
            max_spots: Maximum number of spots to sample (default: 200)

        Returns:
            List of sampled fishing spots
        """
        if not self.coastline_points:
            return []

        # Use segments if available, otherwise treat all points as one segment
        segments = getattr(self, 'coastline_segments', None)
        if not segments or len(segments) == 0:
            segments = [self.coastline_points]

        # Calculate length of each segment
        segment_lengths = []
        for segment in segments:
            seg_length = 0
            for i in range(1, len(segment)):
                p1 = segment[i-1]
                p2 = segment[i]
                dist = self._distance_m(p1[0], p1[1], p2[0], p2[1])
                # Only count connections < 10km (ignore gaps within segment)
                if dist < 10000:
                    seg_length += dist
            segment_lengths.append(seg_length)

        total_length_m = sum(segment_lengths)

        # Calculate number of spots based on spacing
        if num_spots is None:
            num_spots = max(10, min(max_spots, int(total_length_m / spacing_m)))

        actual_spacing = total_length_m / num_spots if num_spots > 0 else spacing_m
        print(f"[INFO] Costa: {total_length_m/1000:.1f}km ({len(segments)} segmentos), espaciado: {actual_spacing:.0f}m, spots: {num_spots}")

        # Distribute spots proportionally to each segment's length
        self.sampled_spots = []
        spot_id = 1

        for seg_idx, (segment, seg_length) in enumerate(zip(segments, segment_lengths)):
            if seg_length == 0 or len(segment) < 2:
                continue

            # Calculate spots for this segment proportionally
            seg_spots = max(1, int(num_spots * (seg_length / total_length_m)))

            # Sample points within this segment
            if len(segment) <= seg_spots:
                seg_indices = range(len(segment))
            else:
                seg_indices = np.linspace(0, len(segment) - 1, seg_spots, dtype=int)

            for local_idx in seg_indices:
                lat, lon = segment[local_idx]

                # Filter: only include spots within STUDY_AREA
                if not STUDY_AREA.contains(lat, lon):
                    continue

                bearing = self._perpendicular_to_sea_segment(segment, local_idx)

                self.sampled_spots.append({
                    'id': spot_id,
                    'lat': lat,
                    'lon': lon,
                    'bearing_to_sea': bearing,
                    'score': 0,
                    'distance_to_fish': 0,
                    'direction_to_fish': 0,
                    'species': self._get_species(lat),
                    'segment': seg_idx
                })
                spot_id += 1

        return self.sampled_spots

    def _add_focus_zone_spots(
        self,
        center_lat: float,
        center_lon: float,
        radius_km: float = 2.0,
        spacing_m: int = 500,
        zone_name: str = "Focus Zone"
    ) -> List[Dict]:
        """
        Add dense sampling spots around a focus zone.

        Uses segments to calculate correct perpendicular bearings.

        Args:
            center_lat, center_lon: Center of focus zone
            radius_km: Radius of focus zone in km
            spacing_m: Spacing between spots in meters (500-1000m recommended)
            zone_name: Name of the zone for labeling

        Returns:
            List of new spots added
        """
        segments = getattr(self, 'coastline_segments', None)
        if not segments:
            segments = [self.coastline_points]

        # Find coastline points within the focus zone, tracking their segment
        zone_points = []  # (segment, local_idx, lat, lon)
        for seg_idx, segment in enumerate(segments):
            for local_idx, (lat, lon) in enumerate(segment):
                dist = self._distance_m(lat, lon, center_lat, center_lon)
                if dist <= radius_km * 1000:
                    zone_points.append((segment, local_idx, lat, lon))

        if not zone_points:
            print(f"[WARN] No hay puntos de costa en zona {zone_name}")
            return []

        # Calculate number of spots based on zone coastline length
        zone_length_m = 0
        for i in range(1, len(zone_points)):
            p1 = zone_points[i-1]
            p2 = zone_points[i]
            # Only count if same segment or close
            dist = self._distance_m(p1[2], p1[3], p2[2], p2[3])
            if dist < 1000:  # Don't count jumps between segments
                zone_length_m += dist

        num_zone_spots = max(5, int(zone_length_m / spacing_m))

        # Sample evenly within zone
        if len(zone_points) <= num_zone_spots:
            zone_indices = range(len(zone_points))
        else:
            zone_indices = np.linspace(0, len(zone_points) - 1, num_zone_spots, dtype=int)

        new_spots = []
        start_id = len(self.sampled_spots) + 1

        for j, zi in enumerate(zone_indices):
            segment, local_idx, lat, lon = zone_points[zi]
            bearing = self._perpendicular_to_sea_segment(segment, local_idx)

            spot = {
                'id': start_id + j,
                'lat': lat,
                'lon': lon,
                'bearing_to_sea': bearing,
                'score': 0,
                'distance_to_fish': 0,
                'direction_to_fish': 0,
                'species': self._get_species(lat),
                'zone': zone_name
            }
            new_spots.append(spot)
            self.sampled_spots.append(spot)

        return new_spots

    def fetch_marine_data(self) -> int:
        """Fetch marine data and generate flow lines following coast contour.

        Priority:
        1. Copernicus data (if available for analysis date)
        2. Open-Meteo API (fallback for real-time)

        Samples from each coastline segment separately to ensure correct
        perpendicular bearings (no cross-segment calculations).
        """
        # Try Copernicus data first (better quality)
        if COPERNICUS_DATA_AVAILABLE and self.copernicus_provider:
            analysis_date = self.analysis_datetime.strftime('%Y-%m-%d')
            copernicus_points = self._fetch_copernicus_marine_data(analysis_date)
            if copernicus_points:
                return len(self.current_vectors)

        # Fallback to Open-Meteo sampling
        print("[INFO] Generando muestreo paralelo a la costa (Open-Meteo)...")

        # Use segments if available
        segments = getattr(self, 'coastline_segments', None)
        if not segments or len(segments) == 0:
            segments = [self.coastline_points]

        # Sample ~25 points total, distributed across segments by length
        target_samples = 25
        offshore_km = [3, 8, 15, 25]
        sample_points = []

        # Calculate total coastline length
        segment_lengths = []
        for segment in segments:
            seg_len = 0
            for i in range(1, len(segment)):
                seg_len += self._distance_m(segment[i-1][0], segment[i-1][1],
                                           segment[i][0], segment[i][1])
            segment_lengths.append(seg_len)

        total_length = sum(segment_lengths)
        if total_length == 0:
            return 0

        # Sample from each segment proportionally
        for seg_idx, (segment, seg_len) in enumerate(zip(segments, segment_lengths)):
            if len(segment) < 3 or seg_len < 1000:  # Skip very short segments
                continue

            # Number of samples for this segment
            seg_samples = max(1, int(target_samples * (seg_len / total_length)))
            step = max(1, len(segment) // seg_samples)

            for local_idx in range(0, len(segment), step):
                lat, lon = segment[local_idx]
                bearing = self._perpendicular_to_sea_segment(segment, local_idx)

                for dist in offshore_km:
                    rad = np.radians(bearing)
                    new_lat = lat + dist / 111.0 * np.cos(rad)
                    new_lon = lon + dist / 111.0 * np.sin(rad) / np.cos(np.radians(lat))
                    sample_points.append((new_lat, new_lon))

        print(f"[INFO] Muestreando {len(sample_points)} puntos desde {len(segments)} segmentos...")

        self.marine_fetcher = MarineDataFetcher()
        self.current_vectors = self.marine_fetcher.fetch_current_vectors(sample_points)
        self.flow_lines = self.marine_fetcher.get_flow_lines(num_steps=5, step_km=4.0)
        self.marine_points = self.marine_fetcher.sampled_points

        print(f"[OK] {len(self.current_vectors)} vectores, {len(self.flow_lines)} lineas de flujo")
        return len(self.current_vectors)

    def _fetch_copernicus_marine_data(self, date: str) -> bool:
        """
        Fetch marine data from Copernicus (SST, currents, waves).

        Args:
            date: Date in YYYY-MM-DD format

        Returns:
            True if data was loaded successfully
        """
        from core.marine_data import MarinePoint, CurrentVector

        print(f"[INFO] Cargando datos de Copernicus para {date}...")

        ocean_points = self.copernicus_provider.get_data_for_date(date)

        if not ocean_points:
            print("[WARN] No hay datos de Copernicus, usando Open-Meteo...")
            return False

        # Get statistics
        stats = self.copernicus_provider.get_statistics(date)
        print(f"      SST: {stats['sst']['count']} pts ({stats['sst']['min']:.1f}°C - {stats['sst']['max']:.1f}°C)")
        print(f"      Corrientes: {stats['currents']['count']} pts (media: {stats['currents']['mean_speed']:.3f} m/s)")
        print(f"      Olas: {stats['waves']['count']} pts (altura media: {stats['waves']['mean_height']:.2f} m)")

        # Convert to MarinePoint format
        self.marine_points = []
        self.current_vectors = []

        for op in ocean_points:
            if op.sst is None:
                continue

            # Create MarinePoint
            mp = MarinePoint(
                lat=op.lat,
                lon=op.lon,
                sst=op.sst,
                wave_height=op.wave_height if op.wave_height else 1.0,
                wave_period=op.wave_period if op.wave_period else 8.0,
                current_speed=op.current_speed if op.current_speed else 0.1,
                current_direction=op.current_direction if op.current_direction else 180.0,
                timestamp=date
            )
            self.marine_points.append(mp)

            # Create CurrentVector for visualization
            if op.uo is not None and op.vo is not None:
                speed = op.current_speed or np.sqrt(op.uo**2 + op.vo**2)
                direction = op.current_direction or np.degrees(np.arctan2(op.vo, op.uo))
                self.current_vectors.append(CurrentVector(
                    lat=op.lat,
                    lon=op.lon,
                    u=op.uo,
                    v=op.vo,
                    speed=speed,
                    direction=direction
                ))

        # Generate flow lines from current vectors
        self.flow_lines = self._generate_flow_lines_from_vectors()

        # Initialize marine_fetcher for compatibility
        self.marine_fetcher = MarineDataFetcher(use_copernicus=False)
        self.marine_fetcher.current_vectors = self.current_vectors
        self.marine_fetcher.sampled_points = self.marine_points

        print(f"[OK] Copernicus: {len(self.marine_points)} puntos, {len(self.current_vectors)} vectores")
        return True

    def _generate_flow_lines_from_vectors(self, num_steps: int = 5, step_km: float = 4.0) -> List:
        """Generate flow lines from current vectors for visualization."""
        flow_lines = []
        step_deg = step_km / 111.0

        for vector in self.current_vectors:
            line = [(vector.lat, vector.lon)]
            lat, lon = vector.lat, vector.lon
            direction = vector.direction

            for _ in range(num_steps):
                rad = np.radians(direction)
                lat += step_deg * np.cos(rad)
                lon += step_deg * np.sin(rad) / np.cos(np.radians(lat))
                line.append((lat, lon))

            flow_lines.append(line)

        return flow_lines

    def generate_fish_zones(self, num_zones: int = 6) -> List[Dict]:
        """Generate fish zones using real data."""
        predictor = FishZonePredictor()
        self.fish_zones = predictor.predict_zones(self.coastline_points, num_zones)
        return self.fish_zones

    def predict_anchovy_migration(self, target_date: str = None, target_hour: int = 17) -> List[Dict]:
        """
        Predict anchovy concentration zones based on migration patterns.

        Args:
            target_date: Target date (default: today)
            target_hour: Hour of day (default: 5 PM peak time)

        Returns:
            List of predicted anchovy zones
        """
        if not target_date:
            target_date = datetime.now().strftime('%Y-%m-%d')

        print(f"[INFO] Prediciendo migracion de anchoveta para {target_date} {target_hour}:00...")

        try:
            self.anchovy_model = AnchovyMigrationModel()
            self.anchovy_zones = self.anchovy_model.predict_concentration_zones(
                target_date, target_hour, num_zones=8
            )

            # Add anchovy zones to fish_zones with special marker
            for i, zone in enumerate(self.anchovy_zones):
                self.fish_zones.append({
                    'id': f'anchovy_{i+1}',
                    'lat': zone['lat'],
                    'lon': zone['lon'],
                    'intensity': min(zone['score'] / 100, 1.0),
                    'radius': 400,  # Larger radius for anchovy schools
                    'cause': f"Anchoveta (score: {zone['score']:.0f})",
                    'sst': zone['avg_sst'],
                    'is_anchovy': True,
                    'historical_hours': zone['historical_hours'],
                    'migration_applied': zone['migration_applied']
                })

            print(f"[OK] {len(self.anchovy_zones)} zonas de anchoveta predichas")
            return self.anchovy_zones

        except Exception as e:
            print(f"[WARN] Error en prediccion de anchoveta: {e}")
            return []

    def run_ml_prediction(self) -> Dict:
        """Run ML prediction using extracted features (32 advanced features)."""
        if not self.marine_points:
            print("[WARN] No marine data for ML prediction")
            return {}

        print("[INFO] Extrayendo 32 features avanzados para ML...")

        # Get conditions for temporal features
        solunar_data = self.conditions.get('solunar', {}) if self.conditions else None
        weather_data = self.conditions.get('weather', {}) if self.conditions else None

        # Extract features with solunar and weather data
        X = self.feature_extractor.extract_from_marine_points(
            self.marine_points,
            self.coastline_points,
            solunar_data=solunar_data,
            weather_data=weather_data
        )

        if X.shape[0] < 5:
            print("[WARN] Insuficientes datos para ML")
            return {}

        print(f"[OK] {X.shape[0]} muestras, {X.shape[1]} features avanzados")

        # Train predictor (unsupervised - uses domain knowledge scoring)
        print("[INFO] Entrenando modelo ML...")
        self.predictor.fit_unsupervised(
            X,
            feature_names=self.feature_extractor.feature_names
        )

        # Get predictions
        self.ml_predictions = self.predictor.predict(X)

        # Update marine points with predictions
        for i, pred in enumerate(self.ml_predictions):
            if i < len(self.feature_extractor.features_cache):
                feat = self.feature_extractor.features_cache[i]
                pred.lat = feat.lat
                pred.lon = feat.lon

        # PCA analysis
        self.pca_analysis = self.predictor.get_pca_analysis()

        print(self.predictor.get_model_summary())
        return self.pca_analysis

    def analyze_spots(self, target_hour: int = None) -> List[Dict]:
        """Analyze and score fishing spots with tide, SSS, SLA integration (V4/V6).

        Args:
            target_hour: Hour of day (0-23) for tide/time calculations.
                        If None, uses current hour from analysis_datetime.
        """
        if not self.fish_zones:
            self.generate_fish_zones()

        # Determine target hour
        if target_hour is None:
            target_hour = self.analysis_datetime.hour

        # Get environmental scores for specific hour
        tide_score, tide_phase, hour_score = self._get_hourly_scores(target_hour)

        # SSS and SLA scores (V4)
        sss_score = self.sss_score  # Set by _fetch_physics_data
        sla_score = self.sla_score  # Set by _fetch_physics_data

        for spot in self.sampled_spots:
            best_score, best_dist, best_dir = 0, float('inf'), 0

            for zone in self.fish_zones:
                dist = self._distance_m(spot['lat'], spot['lon'], zone['lat'], zone['lon'])
                direction = self._bearing(spot['lat'], spot['lon'], zone['lat'], zone['lon'])

                # Movement factor
                ideal = (direction + 180) % 360
                angle_diff = abs(zone.get('movement_direction', 90) - ideal)
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                movement_factor = 1.0 + (1.0 - angle_diff / 180) * 0.4

                # Score calculation
                dist_score = max(0, 100 - dist / 8)
                intensity_score = zone.get('intensity', 0.5) * 40
                zone_score = (dist_score * 0.6 + intensity_score * 0.4) * movement_factor

                if zone_score > best_score:
                    best_score, best_dist, best_dir = zone_score, dist, direction

            # Apply ML boost if available
            ml_boost = self._get_ml_boost(spot['lat'], spot['lon'])

            # Environmental bonuses (V4/V6/V7):
            # Tide: up to ±15 points (marea entrante/saliente = bueno)
            tide_bonus = (tide_score - 0.5) * 30
            # Hour: up to ±12 points (alba/ocaso = excelente)
            hour_bonus = (hour_score - 0.5) * 24
            # SSS: up to ±10 points (salinity optimal range)
            sss_bonus = (sss_score - 0.5) * 20
            # SLA: up to ±8 points (negative SLA = upwelling = good)
            sla_bonus = (sla_score - 0.5) * 16

            # V7 bonuses:
            # Chlorophyll-a: up to ±8 points (productividad primaria)
            chla_bonus = (self.chla_score - 0.5) * 16
            # SST Historical: up to ±6 points (anomalias termicas)
            sst_hist_bonus = (self.sst_historical_score - 0.5) * 12
            # GFW Hotspots: up to +10 points (proximidad a zona de pesca real)
            gfw_bonus = self._get_gfw_bonus(spot['lat'], spot['lon'])

            total_bonus = (tide_bonus + hour_bonus + sss_bonus + sla_bonus +
                          chla_bonus + sst_hist_bonus + gfw_bonus)

            spot['score'] = min(100, max(0, best_score + ml_boost + total_bonus))
            spot['distance_to_fish'] = best_dist
            spot['direction_to_fish'] = best_dir
            spot['tide_phase'] = tide_phase
            spot['tide_score'] = tide_score
            spot['hour_score'] = hour_score
            spot['target_hour'] = target_hour
            spot['sss_score'] = sss_score
            spot['sla_score'] = sla_score

            # V7 fields
            spot['chla_score'] = self.chla_score
            spot['chla_value'] = self.chla_value
            spot['sst_historical_score'] = self.sst_historical_score
            spot['sst_anomaly'] = self.sst_anomaly
            spot['gfw_bonus'] = gfw_bonus

        self.sampled_spots.sort(key=lambda s: s['score'], reverse=True)
        return self.sampled_spots

    def get_conditions(self) -> Dict:
        """Get weather, solunar, tide, SSS and SLA conditions (V4)."""
        if not self.coastline_points:
            return {}

        center_lat = np.mean([p[0] for p in self.coastline_points])
        center_lon = np.mean([p[1] for p in self.coastline_points])

        self.conditions = get_fishing_conditions(center_lat, center_lon)

        # Add tide data (V4)
        self._fetch_tide_data(center_lat, center_lon)
        if self.tide_data:
            self.conditions['tide'] = self.tide_data

        # Add SSS and SLA data (V4)
        physics_data = self._fetch_physics_data(center_lat, center_lon)
        self.conditions['physics'] = physics_data

        # Add Chlorophyll-a data (V7)
        chla_data = self._fetch_chla_data(center_lat, center_lon)
        self.conditions['chlorophyll'] = chla_data

        # Add SST Historical data (V7)
        sst_hist_data = self._fetch_sst_historical_data(center_lat, center_lon)
        self.conditions['sst_historical'] = sst_hist_data

        # Generate GFW Dynamic Hotspots (V7)
        self._generate_dynamic_hotspots()
        self.conditions['gfw_hotspots'] = {
            'count': len(self.dynamic_hotspots),
            'available': GFW_HOTSPOTS_AVAILABLE
        }

        return self.conditions

    def _fetch_tide_data(self, lat: float, lon: float) -> Dict:
        """Fetch tide data for current time and location."""
        if not TIDES_AVAILABLE:
            self.tide_data = {}
            return {}

        try:
            if not self.tide_fetcher:
                self.tide_fetcher = TideFetcher()

            # Use analysis datetime (can be current or specified date)
            now = self.analysis_datetime
            today = now.strftime('%Y-%m-%d')

            # Get tide state for analysis time
            tide_state = self.tide_fetcher.get_tidal_state(now, lat, lon)

            # Get today's extremes
            extremes = self.tide_fetcher.get_tide_extremes_for_date(today, lat, lon)

            # Get best fishing hours
            best_hours = self.tide_fetcher.get_best_fishing_hours(today, lat, lon, top_n=5)

            # Convert TidalState namedtuple to dict if needed
            if hasattr(tide_state, '_asdict'):
                tide_dict = tide_state._asdict()
            else:
                tide_dict = tide_state if isinstance(tide_state, dict) else {}

            self.tide_data = {
                'current': tide_dict,
                'extremes': extremes,
                'best_hours': best_hours,
                'tide_score': tide_dict.get('fishing_score', 0.5),
                'tide_phase': tide_dict.get('phase', 'unknown'),
                'tide_height': tide_dict.get('height', 0)
            }

            return self.tide_data

        except Exception as e:
            print(f"[WARN] Error obteniendo mareas: {e}")
            self.tide_data = {'tide_score': 0.5, 'tide_phase': 'unknown'}
            return self.tide_data

    def _get_hourly_scores(self, hour: int) -> Tuple[float, str, float]:
        """Get tide and hour scores for a specific hour of the day (V6).

        Args:
            hour: Hour of day (0-23)

        Returns:
            Tuple of (tide_score, tide_phase, hour_score)
        """
        # Hour score based on fishing patterns (alba/ocaso = best)
        HOUR_SCORES = {
            0: 0.2, 1: 0.2, 2: 0.2, 3: 0.3,      # Madrugada
            4: 0.5, 5: 0.7,                        # Pre-alba
            6: 0.95, 7: 0.9,                       # Alba (excelente)
            8: 0.75, 9: 0.65, 10: 0.55, 11: 0.45, # Mañana
            12: 0.35, 13: 0.3, 14: 0.35,          # Mediodía (bajo)
            15: 0.45, 16: 0.55, 17: 0.7,          # Tarde
            18: 0.9, 19: 0.85,                     # Atardecer (excelente)
            20: 0.6, 21: 0.5, 22: 0.4, 23: 0.3    # Noche
        }
        hour_score = HOUR_SCORES.get(hour, 0.5)

        # Get tide score for specific hour
        tide_score = 0.5  # Default neutral
        tide_phase = 'unknown'

        if TIDES_AVAILABLE and self.tide_fetcher:
            try:
                # Create datetime for the target hour
                target_dt = self.analysis_datetime.replace(hour=hour, minute=0, second=0)

                # Get center of coastline for reference
                center_lat = np.mean([p[0] for p in self.coastline_points])
                center_lon = np.mean([p[1] for p in self.coastline_points])

                # Get tide state for target hour
                tide_state = self.tide_fetcher.get_tidal_state(target_dt, center_lat, center_lon)

                if hasattr(tide_state, '_asdict'):
                    tide_dict = tide_state._asdict()
                    tide_score = tide_dict.get('fishing_score', 0.5)
                    tide_phase = tide_dict.get('phase', 'unknown')
                elif isinstance(tide_state, dict):
                    tide_score = tide_state.get('fishing_score', 0.5)
                    tide_phase = tide_state.get('phase', 'unknown')

            except Exception as e:
                print(f"[WARN] Error calculando marea para hora {hour}: {e}")

        return tide_score, tide_phase, hour_score

    def analyze_spots_all_hours(self) -> Dict[int, List[Dict]]:
        """Pre-calculate scores for all 24 hours (V6).

        Returns:
            Dict mapping hour (0-23) to list of spots with scores
        """
        all_hours_data = {}

        print("[INFO] Pre-calculando scores para 24 horas...")
        for hour in range(24):
            # Reset spots to original state
            for spot in self.sampled_spots:
                spot.pop('score', None)

            # Calculate scores for this hour
            spots_for_hour = self.analyze_spots(target_hour=hour)

            # Deep copy the results
            all_hours_data[hour] = [
                {
                    'lat': s['lat'],
                    'lon': s['lon'],
                    'score': s['score'],
                    'tide_phase': s.get('tide_phase', 'unknown'),
                    'tide_score': s.get('tide_score', 0.5),
                    'hour_score': s.get('hour_score', 0.5),
                    'substrate': s.get('substrate', 'unknown'),
                    'species': s.get('species', [])
                }
                for s in spots_for_hour
            ]

        # Find best hour for each spot
        for spot_idx in range(len(self.sampled_spots)):
            best_hour = max(range(24), key=lambda h: all_hours_data[h][spot_idx]['score'])
            best_score = all_hours_data[best_hour][spot_idx]['score']

            # Store in original spot data
            if spot_idx < len(self.sampled_spots):
                self.sampled_spots[spot_idx]['best_hour'] = best_hour
                self.sampled_spots[spot_idx]['best_score'] = best_score

        print(f"[OK] Scores calculados para 24 horas, {len(self.sampled_spots)} spots")
        return all_hours_data

    def _fetch_physics_data(self, lat: float, lon: float) -> Dict:
        """Fetch SSS and SLA data for scoring (V4)."""
        if not PHYSICS_AVAILABLE:
            return {'sss_score': 0.5, 'sla_score': 0.5}

        try:
            if not self.physics_fetcher:
                self.physics_fetcher = CopernicusPhysicsFetcher()

            # Use analysis date (can be current or specified)
            today = self.analysis_datetime.strftime('%Y-%m-%d')

            # Get SSS (salinity)
            sss_value = self.physics_fetcher.get_sss_for_location(today, lat, lon)
            if sss_value is not None:
                self.sss_score = self.physics_fetcher.calculate_sss_score(sss_value)
            else:
                self.sss_score = 0.5  # Neutral

            # Get SLA (sea level anomaly)
            sla_value = self.physics_fetcher.get_sla_for_location(today, lat, lon)
            if sla_value is not None:
                self.sla_score = self.physics_fetcher.calculate_sla_score(sla_value)
            else:
                self.sla_score = 0.5  # Neutral

            return {
                'sss_value': sss_value,
                'sss_score': self.sss_score,
                'sla_value': sla_value,
                'sla_score': self.sla_score
            }

        except Exception as e:
            print(f"[WARN] Error obteniendo SSS/SLA: {e}")
            self.sss_score = 0.5
            self.sla_score = 0.5
            return {'sss_score': 0.5, 'sla_score': 0.5}

    def _fetch_chla_data(self, lat: float, lon: float) -> Dict:
        """Fetch Chlorophyll-a data for scoring (V7)."""
        if not CHLA_AVAILABLE:
            return {'chla_value': None, 'chla_score': 0.5}

        try:
            if not self.chla_fetcher:
                self.chla_fetcher = ChlorophyllFetcher()

            # Use analysis date
            today = self.analysis_datetime.strftime('%Y-%m-%d')

            # Get Chl-a value
            self.chla_value = self.chla_fetcher.get_value_for_location(today, lat, lon)

            if self.chla_value is not None:
                self.chla_score = self.chla_fetcher.calculate_score(self.chla_value)
            else:
                self.chla_score = 0.5  # Neutral

            return {
                'chla_value': self.chla_value,
                'chla_score': self.chla_score
            }

        except Exception as e:
            print(f"[WARN] Error obteniendo Clorofila-a: {e}")
            self.chla_score = 0.5
            self.chla_value = None
            return {'chla_value': None, 'chla_score': 0.5}

    def _fetch_sst_historical_data(self, lat: float, lon: float) -> Dict:
        """Fetch SST Historical data for scoring (V7)."""
        if not SST_HISTORICAL_AVAILABLE:
            return {'sst': None, 'anomaly': 0.0, 'score': 0.5}

        try:
            if not self.sst_provider:
                self.sst_provider = SSTHistoricalProvider()

            # Use analysis date
            today = self.analysis_datetime.strftime('%Y-%m-%d')

            # Get SST with anomaly
            result = self.sst_provider.get_sst_with_anomaly(today, lat, lon)

            self.sst_historical_score = result.score
            self.sst_anomaly = result.anomaly

            return {
                'sst': result.sst,
                'anomaly': result.anomaly,
                'monthly_mean': result.monthly_mean,
                'trend_7d': result.trend_7d,
                'score': result.score
            }

        except Exception as e:
            print(f"[WARN] Error obteniendo SST historico: {e}")
            self.sst_historical_score = 0.5
            self.sst_anomaly = 0.0
            return {'sst': None, 'anomaly': 0.0, 'score': 0.5}

    def _generate_dynamic_hotspots(self) -> None:
        """Generate dynamic hotspots from GFW data (V7)."""
        if not GFW_HOTSPOTS_AVAILABLE:
            self.dynamic_hotspots = []
            return

        try:
            if not self.gfw_generator:
                self.gfw_generator = GFWHotspotGenerator()

            # Generate hotspots
            self.dynamic_hotspots = self.gfw_generator.generate_hotspots(
                min_fishing_hours=5.0,
                eps_km=2.0,
                min_samples=5
            )

            print(f"[OK] Generados {len(self.dynamic_hotspots)} hotspots dinamicos GFW")

        except Exception as e:
            print(f"[WARN] Error generando hotspots GFW: {e}")
            self.dynamic_hotspots = []

    def _get_gfw_bonus(self, lat: float, lon: float) -> float:
        """Calculate GFW hotspot proximity bonus (V7)."""
        if not self.dynamic_hotspots or not self.gfw_generator:
            return 0.0

        try:
            return self.gfw_generator.calculate_proximity_bonus(
                lat, lon,
                hotspots=self.dynamic_hotspots,
                max_distance_km=10.0,
                max_bonus=10.0
            )
        except:
            return 0.0

    def create_map(self, output_path: str = "output/analysis_map.html") -> str:
        """Create visualization map."""
        if not self.coastline_points:
            raise ValueError("No coastline data loaded")

        center = (
            np.mean([p[0] for p in self.coastline_points]),
            np.mean([p[1] for p in self.coastline_points])
        )

        self.map_view.create_map(center=center, zoom=10)
        # Pass segments to draw each coastline section separately
        segments = getattr(self, 'coastline_segments', None)
        self.map_view.add_coastline(self.coastline_points, segments=segments)
        self.map_view.add_fish_zones(self.fish_zones)
        self.map_view.add_flow_lines(self.flow_lines, self.current_vectors)
        self.map_view.add_marine_points(self.marine_points)
        self.map_view.add_fishing_spots(self.sampled_spots)
        self.map_view.add_legend()

        # Add timeline if data available
        if TIMELINE_AVAILABLE:
            try:
                print("[INFO] Generando datos de linea de tiempo...")
                timeline_data = generate_timeline_data()
                self.map_view.add_timeline(timeline_data)
                print(f"[OK] Timeline con {len(timeline_data.get('monthly_stats', []))} meses de datos")
            except Exception as e:
                print(f"[WARN] No se pudo agregar timeline: {e}")

        # Add multi-day predictions for interactive date switching
        try:
            print("[INFO] Generando predicciones para 7 dias...")
            multiday_data = self.generate_multiday_predictions(days=7)
            self.map_view.add_multiday_spots(multiday_data)
            print(f"[OK] {len(multiday_data)} dias de predicciones generados")
        except Exception as e:
            print(f"[WARN] No se pudo agregar panel multi-dia: {e}")

        # Add hourly predictions for 7 days (for dynamic date selector)
        if HOURLY_PREDICTIONS_AVAILABLE:
            try:
                print("[INFO] Generando predicciones horarias para 7 dias...")
                hourly_gen = HourlyPredictionGenerator()
                analysis_date_str = self.analysis_datetime.strftime('%Y-%m-%d')

                # Generate for center location
                center_lat = np.mean([p[0] for p in self.coastline_points])
                center_lon = np.mean([p[1] for p in self.coastline_points])

                hourly_multiday = hourly_gen.generate_multiday(
                    start_date=analysis_date_str,
                    num_days=7,
                    lat=center_lat,
                    lon=center_lon
                )

                # Add to map for dynamic date switching
                self.map_view.add_multiday_hourly_data(hourly_multiday)

                # Also add hourly panel for first day
                first_date = analysis_date_str
                if first_date in hourly_multiday.get('days', {}):
                    first_day_data = hourly_multiday['days'][first_date]
                    hourly_panel_data = {
                        'date': first_date,
                        'location_name': hourly_multiday['location']['name'],
                        'predictions': first_day_data['predictions'],
                        'tide_extremes': first_day_data['tide_extremes'],
                        'best_hours': first_day_data['best_hours']
                    }
                    self.map_view.add_hourly_panel(hourly_panel_data)

                print(f"[OK] Predicciones horarias para 7 dias embebidas")
            except Exception as e:
                print(f"[WARN] No se pudo agregar predicciones horarias: {e}")

        # Add unified hourly scoring data (V6) for spot-level hour selection
        if hasattr(self, 'hourly_spots_data') and self.hourly_spots_data:
            try:
                self.map_view.add_hourly_spots_data(self.hourly_spots_data)
                print(f"[OK] Datos de scoring unificado por hora embebidos")
            except Exception as e:
                print(f"[WARN] No se pudo agregar scoring unificado: {e}")

        # Add user location marker if proximity search is active
        if self.user_location:
            self.map_view.add_user_location(
                self.user_location['lat'],
                self.user_location['lon'],
                self.user_location.get('radius_km', 10.0)
            )

        self.map_view.finalize()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        self.map_view.save(output_path)

        return output_path

    def run_full_analysis(
        self,
        coastline_path: str,
        output_path: str = "output/analysis_map.html",
        target_date: datetime = None,
        user_location: Dict = None
    ) -> Dict:
        """Run complete analysis pipeline.

        Args:
            coastline_path: Path to coastline GeoJSON file
            output_path: Output HTML map path
            target_date: Optional datetime for analysis. If None, uses current datetime.
            user_location: Optional dict with 'lat', 'lon', 'radius_km' for proximity search.
        """
        # Store user location for proximity search
        self.user_location = user_location

        # Use provided date or current datetime
        self.analysis_datetime = target_date if target_date else datetime.now()
        analysis_date_str = self.analysis_datetime.strftime('%Y-%m-%d')

        print("=" * 60)
        print("   ANALISIS DE PESCA CON ML")
        print("=" * 60)
        print(f"Fecha: {self.analysis_datetime.strftime('%d/%m/%Y %H:%M')}\n")

        # 1. Load coastline
        print("[1/8] Cargando linea costera...")
        n_points = self.load_coastline(coastline_path)
        print(f"      {n_points} puntos cargados")

        # 2. Sample spots with focus on key areas
        print("\n[2/8] Muestreando puntos de pesca...")
        # Base sampling along entire coast - high density for full coverage
        # 281km coast / 300m spacing = ~937 spots, limited to 600
        spots = self.sample_fishing_spots(spacing_m=300, max_spots=600)
        print(f"      {len(spots)} spots muestreados")

        # 3. Generate fish zones
        print("\n[3/8] Generando zonas de peces...")
        zones = self.generate_fish_zones()
        print(f"      {len(zones)} zonas generadas")

        # 4. Predict anchovy migration
        print("\n[4/8] Prediciendo zonas de anchoveta...")
        anchovy_zones = self.predict_anchovy_migration(
            target_date=analysis_date_str,
            target_hour=17
        )
        print(f"      {len(anchovy_zones)} zonas de anchoveta")

        # 5. Fetch marine data
        print("\n[5/8] Obteniendo datos marinos...")
        n_vectors = self.fetch_marine_data()

        # 6. Get conditions (before ML for temporal features)
        print("\n[6/8] Obteniendo condiciones meteorologicas y solunares...")
        conditions = self.get_conditions()

        # 7. ML prediction (uses 32 advanced features)
        print("\n[7/8] Ejecutando prediccion ML con 32 features...")
        pca_results = self.run_ml_prediction()

        # 8. Analyze spots (current hour)
        print("\n[8/8] Analizando spots...")
        results = self.analyze_spots()

        # 8b. Pre-calculate scores for all 24 hours (V6 - unified scoring)
        print("\n[8b] Pre-calculando scores para 24 horas (scoring unificado)...")
        self.hourly_spots_data = self.analyze_spots_all_hours()

        weather = conditions.get('weather', {})
        solunar = conditions.get('solunar', {})
        print(f"      Temp: {weather.get('temperature', 'N/A')}C")
        print(f"      Luna: {solunar.get('moon_phase', 'N/A')}")

        # Create map
        print("\nGenerando mapa...")
        map_path = self.create_map(output_path)

        # Filter results to Peru only for recommendations
        peru_results = [s for s in results if s['lat'] >= PERU_SOUTH_LIMIT]

        # Print results based on proximity search
        if self.user_location:
            nearby_spots = self._filter_spots_by_proximity(peru_results)
            self._print_proximity_results(peru_results[:5], nearby_spots[:5])
        else:
            self._print_results(peru_results[:5])

        print(f"\n{'=' * 60}")
        print(f"MAPA GUARDADO: {map_path}")
        print("=" * 60)

        return {
            'spots': results,
            'zones': zones,
            'pca': pca_results,
            'conditions': conditions,
            'map_path': map_path,
            'user_location': self.user_location
        }

    def generate_multiday_predictions(self, days: int = 7) -> List[Dict]:
        """
        Generate predictions for multiple days for interactive date switching.

        Args:
            days: Number of days to generate (default: 7)

        Returns:
            List of daily prediction summaries
        """
        base_date = self.analysis_datetime
        predictions = []

        # Reference location for tide/physics calculations
        ref_lat = -17.7  # Approximate center of fishing area
        ref_lon = -71.3

        for day_offset in range(days):
            target_dt = base_date + timedelta(days=day_offset)
            target_date_str = target_dt.strftime('%Y-%m-%d')
            day_name = ['Lun', 'Mar', 'Mie', 'Jue', 'Vie', 'Sab', 'Dom'][target_dt.weekday()]

            # Calculate tide score for this day
            tide_score = 0.5
            tide_phase = 'unknown'
            if TIDES_AVAILABLE and self.tide_fetcher:
                try:
                    # Get tide at 6 AM (typical fishing hour)
                    morning_dt = target_dt.replace(hour=6, minute=0)
                    tide_state = self.tide_fetcher.get_tidal_state(morning_dt, ref_lat, ref_lon)
                    if hasattr(tide_state, '_asdict'):
                        tide_dict = tide_state._asdict()
                        tide_score = tide_dict.get('fishing_score', 0.5)
                        tide_phase = tide_dict.get('phase', 'unknown')
                except Exception:
                    pass

            # Calculate SSS/SLA scores for this day
            sss_score = 0.5
            sla_score = 0.5
            if PHYSICS_AVAILABLE and self.physics_fetcher:
                try:
                    sss_val = self.physics_fetcher.get_sss_for_location(target_date_str, ref_lat, ref_lon)
                    if sss_val:
                        sss_score = self.physics_fetcher.calculate_sss_score(sss_val)

                    sla_val = self.physics_fetcher.get_sla_for_location(target_date_str, ref_lat, ref_lon)
                    if sla_val:
                        sla_score = self.physics_fetcher.calculate_sla_score(sla_val)
                except Exception:
                    pass

            # Calculate composite score
            # Base score from current spots (averaged)
            base_score = 35  # Default
            if self.sampled_spots:
                scores = [s.get('score', 35) for s in self.sampled_spots if 'score' in s]
                if scores:
                    base_score = sum(scores) / len(scores)

            # Apply daily environmental bonuses
            tide_bonus = (tide_score - 0.5) * 30  # ±15 points
            sss_bonus = (sss_score - 0.5) * 20   # ±10 points
            sla_bonus = (sla_score - 0.5) * 16   # ±8 points

            daily_score = base_score + tide_bonus + sss_bonus + sla_bonus
            daily_score = max(10, min(95, daily_score))  # Clamp to valid range

            # Get top 5 spots for this day (recalculated with daily bonuses)
            top_spots = []
            for spot in self.sampled_spots[:5]:
                spot_base = spot.get('score', 35)
                spot_score = spot_base + tide_bonus + sss_bonus + sla_bonus
                spot_score = max(10, min(95, spot_score))
                top_spots.append({
                    'lat': float(spot.get('lat', 0)),
                    'lon': float(spot.get('lon', 0)),
                    'score': float(round(spot_score, 1)),
                    'species': spot.get('species', [])
                })

            # Translate tide phase to Spanish
            tide_phase_es = {
                'flooding': 'Marea entrante',
                'ebbing': 'Marea saliente',
                'slack_high': 'Pleamar',
                'slack_low': 'Bajamar',
                'unknown': 'Desconocido'
            }.get(tide_phase, tide_phase)

            predictions.append({
                'date': target_date_str,
                'day_name': day_name,
                'day_offset': day_offset,
                'avg_score': float(round(daily_score, 1)),
                'tide_score': float(round(tide_score, 2)),
                'tide_phase': tide_phase_es,
                'sss_score': float(round(sss_score, 2)),
                'sla_score': float(round(sla_score, 2)),
                'top_spots': top_spots
            })

        return predictions

    # === Private helpers ===

    def _remove_duplicates(self, points: List[Tuple], threshold_m: float = 50) -> List[Tuple]:
        if not points:
            return []
        result = [points[0]]
        for lat, lon in points[1:]:
            if self._distance_m(lat, lon, result[-1][0], result[-1][1]) > threshold_m:
                result.append((lat, lon))
        return result

    def _perpendicular_to_sea(self, idx: int) -> float:
        """
        Calculate bearing perpendicular to coast pointing towards the sea.

        In Peru's south coast (Tacna-Ilo), the Pacific Ocean is always to the WEST.
        So the perpendicular direction should have a westward component (lon decreases).
        """
        if len(self.coastline_points) < 2:
            return 270  # Default: West

        idx = min(idx, len(self.coastline_points) - 1)
        if idx == 0:
            p1, p2 = self.coastline_points[0], self.coastline_points[1]
        elif idx >= len(self.coastline_points) - 1:
            p1, p2 = self.coastline_points[-2], self.coastline_points[-1]
        else:
            p1, p2 = self.coastline_points[idx - 1], self.coastline_points[idx + 1]

        coast_bearing = self._bearing(p1[0], p1[1], p2[0], p2[1])
        perp1 = (coast_bearing + 90) % 360
        perp2 = (coast_bearing - 90) % 360

        lat, lon = self.coastline_points[idx]

        # Test both perpendicular directions - choose the one going towards sea (WEST)
        for perp in [perp1, perp2]:
            # Convert bearing to radians for trig
            # Bearing: 0=N, 90=E, 180=S, 270=W
            rad = np.radians(90 - perp)  # Convert to math angle (0=E, 90=N)

            # Calculate test point 100m in this direction
            dx = 100 * np.cos(rad)  # East-West component (+ = East)

            # If dx is negative, we're going West (towards the ocean)
            if dx < 0:
                return perp

        # Fallback: return the one closer to 270 (West)
        diff1 = min(abs(perp1 - 270), 360 - abs(perp1 - 270))
        diff2 = min(abs(perp2 - 270), 360 - abs(perp2 - 270))
        return perp1 if diff1 < diff2 else perp2

    def _perpendicular_to_sea_segment(self, segment: List[Tuple[float, float]], idx: int) -> float:
        """
        Calculate bearing perpendicular to coast using only points from the same segment.

        This prevents incorrect bearings when coastline has multiple disconnected segments.

        Args:
            segment: List of (lat, lon) points for a single coastline segment
            idx: Index within the segment

        Returns:
            Bearing in degrees pointing towards the sea (westward)
        """
        if len(segment) < 2:
            return 270  # Default: West

        idx = min(idx, len(segment) - 1)
        if idx == 0:
            p1, p2 = segment[0], segment[1]
        elif idx >= len(segment) - 1:
            p1, p2 = segment[-2], segment[-1]
        else:
            p1, p2 = segment[idx - 1], segment[idx + 1]

        coast_bearing = self._bearing(p1[0], p1[1], p2[0], p2[1])
        perp1 = (coast_bearing + 90) % 360
        perp2 = (coast_bearing - 90) % 360

        # Test both perpendicular directions - choose the one going towards sea (WEST)
        for perp in [perp1, perp2]:
            rad = np.radians(90 - perp)
            dx = 100 * np.cos(rad)
            if dx < 0:  # Going West = towards ocean
                return perp

        # Fallback: return the one closer to 270 (West)
        diff1 = min(abs(perp1 - 270), 360 - abs(perp1 - 270))
        diff2 = min(abs(perp2 - 270), 360 - abs(perp2 - 270))
        return perp1 if diff1 < diff2 else perp2

    def _distance_m(self, lat1, lon1, lat2, lon2) -> float:
        dlat = (lat2 - lat1) * 111000
        dlon = (lon2 - lon1) * 111000 * np.cos(np.radians((lat1 + lat2) / 2))
        return np.sqrt(dlat**2 + dlon**2)

    def _bearing(self, lat1, lon1, lat2, lon2) -> float:
        dlat, dlon = lat2 - lat1, lon2 - lon1
        return np.degrees(np.arctan2(dlon, dlat)) % 360

    def _get_species(self, lat: float) -> List[Dict]:
        rocky = [(-17.7, -17.65), (-17.82, -17.78)]
        sandy = [(-18.15, -18.05), (-17.93, -17.88)]

        substrate = "mixto"
        for (lat_min, lat_max) in rocky:
            if lat_min <= lat <= lat_max:
                substrate = "roca"
                break
        for (lat_min, lat_max) in sandy:
            if lat_min <= lat <= lat_max:
                substrate = "arena"
                break

        return [
            {'name': s, 'lure': self.SPECIES_LURES.get(s, '')}
            for s in self.SPECIES_BY_SUBSTRATE.get(substrate, [])[:3]
        ]

    def _get_ml_boost(self, lat: float, lon: float) -> float:
        """Get ML score boost for a location."""
        if not self.ml_predictions:
            return 0

        best_boost = 0
        for pred in self.ml_predictions:
            dist = self._distance_m(lat, lon, pred.lat, pred.lon)
            if dist < 5000:  # Within 5km
                proximity = 1 - (dist / 5000)
                boost = (pred.score / 100) * 15 * proximity * pred.confidence
                best_boost = max(best_boost, boost)

        return best_boost

    def _filter_spots_by_proximity(self, spots: List[Dict]) -> List[Dict]:
        """Filter and sort spots by distance to user location."""
        if not self.user_location:
            return spots

        user_lat = self.user_location['lat']
        user_lon = self.user_location['lon']
        radius_km = self.user_location.get('radius_km', 10.0)
        radius_m = radius_km * 1000

        nearby = []
        for spot in spots:
            dist = self._distance_m(user_lat, user_lon, spot['lat'], spot['lon'])
            if dist <= radius_m:
                spot_copy = spot.copy()
                spot_copy['distance_to_user'] = dist
                nearby.append(spot_copy)

        # Sort by score (best first), then by distance
        nearby.sort(key=lambda x: (-x['score'], x['distance_to_user']))
        return nearby

    def _print_results(self, top_spots: List[Dict]):
        # Spots should already be filtered to Peru before calling this method
        print(f"\n{'=' * 60}")
        print("TOP 5 MEJORES PUNTOS DE PESCA (PERU)")
        print("=" * 60)

        if not top_spots:
            print("\nNo hay spots disponibles.")
            return

        for i, spot in enumerate(top_spots[:5]):
            emoji = "*" if i == 0 else "#"
            print(f"\n{emoji}{i+1} - Score: {spot['score']:.1f}/100")
            print(f"   Coords: {spot['lat']:.6f}, {spot['lon']:.6f}")
            print(f"   Dist. peces: {spot['distance_to_fish']:.0f}m")
            if spot.get('species'):
                names = [s['name'] for s in spot['species']]
                print(f"   Especies: {', '.join(names)}")

    def _print_proximity_results(self, global_spots: List[Dict], nearby_spots: List[Dict]):
        """Print both global and nearby best spots."""
        user_lat = self.user_location['lat']
        user_lon = self.user_location['lon']
        radius_km = self.user_location.get('radius_km', 10.0)

        # Nearby spots
        print(f"\n{'=' * 60}")
        print(f"MEJORES SPOTS CERCANOS (dentro de {radius_km}km)")
        print(f"Tu ubicacion: ({user_lat:.4f}, {user_lon:.4f})")
        print("=" * 60)

        if nearby_spots:
            for i, spot in enumerate(nearby_spots):
                emoji = "*" if i == 0 else "#"
                dist_km = spot.get('distance_to_user', 0) / 1000
                print(f"\n{emoji}{i+1} - Score: {spot['score']:.1f}/100 | Distancia: {dist_km:.1f}km")
                print(f"   Coords: {spot['lat']:.6f}, {spot['lon']:.6f}")
                if spot.get('species'):
                    names = [s['name'] for s in spot['species']]
                    print(f"   Especies: {', '.join(names)}")
        else:
            print(f"\nNo hay spots dentro de {radius_km}km de tu ubicacion.")
            print("Intenta aumentar el radio con --radius")

        # Global spots
        print(f"\n{'-' * 60}")
        print("MEJORES SPOTS GLOBALES (toda la costa)")
        print("-" * 60)

        for i, spot in enumerate(global_spots):
            dist_to_user = self._distance_m(user_lat, user_lon, spot['lat'], spot['lon']) / 1000
            emoji = "*" if i == 0 else "#"
            print(f"\n{emoji}{i+1} - Score: {spot['score']:.1f}/100 | A {dist_to_user:.1f}km de ti")
            print(f"   Coords: {spot['lat']:.6f}, {spot['lon']:.6f}")

    # === Fish Movement Prediction ===

    def predict_fish_movement(self, hours_ahead: int = 24) -> List[Dict]:
        """
        Predict where fish schools will move based on current patterns.

        Uses Lagrangian advection with current vectors to predict
        future positions of detected fish zones.
        """
        if not HISTORICAL_AVAILABLE:
            print("[WARN] Fish movement prediction requires historical_fetcher module")
            return []

        if not self.current_vectors or not self.fish_zones:
            print("[WARN] Need current vectors and fish zones for movement prediction")
            return []

        print(f"[INFO] Predicting fish movement for next {hours_ahead} hours...")

        # Initialize movement predictor
        self.movement_predictor = FishMovementPredictor()

        # Build current field from vectors
        u_data = {}
        v_data = {}
        for vec in self.current_vectors:
            key = (round(vec.lat, 2), round(vec.lon, 2))
            rad = np.radians(vec.direction)
            u_data[key] = vec.speed * np.sin(rad)
            v_data[key] = vec.speed * np.cos(rad)

        self.movement_predictor.current_field = {
            k: (u_data[k], v_data[k]) for k in u_data if k in v_data
        }

        # Predict movement for each fish zone
        current_hotspots = [(z['lat'], z['lon']) for z in self.fish_zones]
        self.future_hotspots = self.movement_predictor.get_future_hotspots(
            current_hotspots, hours=hours_ahead
        )

        print(f"[OK] Predicted {len(self.future_hotspots)} future positions")

        # Add to fish zones for visualization
        for fh in self.future_hotspots:
            self.fish_zones.append({
                'lat': fh['predicted_lat'],
                'lon': fh['predicted_lon'],
                'intensity': 0.5 * fh['confidence'],
                'is_prediction': True,
                'hours_ahead': hours_ahead,
                'confidence': fh['confidence']
            })

        return self.future_hotspots

    # === Supervised Learning with Historical Data ===

    def train_with_historical_data(
        self,
        start_date: str = "2022-01-01",
        end_date: str = "2025-12-31",
        force_download: bool = False
    ) -> Dict:
        """
        Train ML model using historical data with actual fishing activity.

        This enables supervised learning with real fishing data from GFW.

        Args:
            start_date: Start of historical period
            end_date: End of historical period
            force_download: Force re-download even if cached

        Returns:
            Training statistics
        """
        if not HISTORICAL_AVAILABLE:
            print("[WARN] Historical data module not available")
            print("       Install: pip install copernicusmarine earthaccess erddapy")
            return {}

        print("=" * 60)
        print("TRAINING WITH HISTORICAL DATA (SUPERVISED)")
        print("=" * 60)

        # Initialize historical fetcher
        self.historical_fetcher = HistoricalDataFetcher()

        # Build training dataset
        X_train, y_train = self.historical_fetcher.build_training_dataset(
            start_date, end_date
        )

        if X_train.shape[0] < 10:
            print("[ERROR] Insufficient training data")
            return {'error': 'Insufficient data'}

        # Train predictor in supervised mode
        print("[INFO] Training ML model with actual fishing data...")

        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import GradientBoostingRegressor

        # Create supervised model
        supervised_model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )

        # Cross-validation
        cv_scores = cross_val_score(
            supervised_model, X_train, y_train,
            cv=5, scoring='neg_mean_squared_error'
        )

        rmse_scores = np.sqrt(-cv_scores)
        print(f"[OK] Cross-validation RMSE: {rmse_scores.mean():.3f} (+/- {rmse_scores.std():.3f})")

        # Train final model
        supervised_model.fit(X_train, y_train)

        # Update predictor with supervised model
        self.predictor.regressor = supervised_model
        self.predictor.is_fitted = True
        self.is_supervised_mode = True

        stats = {
            'training_samples': X_train.shape[0],
            'positive_samples': int((y_train > 0).sum()),
            'cv_rmse_mean': float(rmse_scores.mean()),
            'cv_rmse_std': float(rmse_scores.std()),
            'date_range': f"{start_date} to {end_date}",
            'mode': 'supervised'
        }

        print(f"\n[OK] Model trained with {stats['training_samples']} samples")
        print(f"[OK] Positive fishing events: {stats['positive_samples']}")

        return stats

    def download_historical_data(
        self,
        years: int = 4,
        sources: List[str] = ['noaa', 'gfw']
    ) -> Dict:
        """
        Download historical data for specified number of years.

        Args:
            years: Number of years to download (default 4)
            sources: Data sources ('noaa', 'copernicus', 'gfw')

        Returns:
            Download statistics
        """
        if not HISTORICAL_AVAILABLE:
            print("[ERROR] Historical data module not available")
            return {}

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365 * years)).strftime("%Y-%m-%d")

        print(f"\n[INFO] Downloading {years} years of historical data...")
        print(f"       Period: {start_date} to {end_date}")
        print(f"       Sources: {', '.join(sources)}")

        self.historical_fetcher = HistoricalDataFetcher()

        results = {'sources': {}}

        if 'noaa' in sources:
            ds = self.historical_fetcher.fetch_noaa_erddap_sst(start_date, end_date)
            results['sources']['noaa_sst'] = ds is not None

        if 'copernicus' in sources:
            ds = self.historical_fetcher.fetch_copernicus_sst(start_date, end_date)
            results['sources']['copernicus_sst'] = ds is not None

            ds = self.historical_fetcher.fetch_copernicus_currents(start_date, end_date)
            results['sources']['copernicus_currents'] = ds is not None

            ds = self.historical_fetcher.fetch_copernicus_chlorophyll(start_date, end_date)
            results['sources']['copernicus_chlorophyll'] = ds is not None

        if 'gfw' in sources:
            events = self.historical_fetcher.fetch_gfw_fishing_activity(start_date, end_date)
            results['sources']['gfw_fishing'] = len(events) > 0
            results['gfw_events'] = len(events)

        results['date_range'] = f"{start_date} to {end_date}"
        results['years'] = years

        stats = self.historical_fetcher.get_statistics()
        results.update(stats)

        print(f"\n[OK] Download complete. Database: {stats.get('database_path', 'N/A')}")

        return results

    def run_full_analysis_supervised(
        self,
        coastline_path: str,
        output_path: str = "output/analysis_supervised.html",
        train_years: int = 4,
        target_hour: int = 17
    ) -> Dict:
        """
        Run complete analysis with supervised ML model.

        This first trains on historical data, then runs the prediction.

        Args:
            coastline_path: Path to coastline GeoJSON
            output_path: Output HTML path
            train_years: Years of historical data for training
            target_hour: Target hour for anchovy prediction (default 5 PM)
        """
        print("=" * 60)
        print("   ANALISIS DE PESCA CON ML SUPERVISADO")
        print("   + MODELO DE MIGRACION DE ANCHOVETA")
        print("=" * 60)
        print(f"Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
        print(f"Hora objetivo: {target_hour}:00\n")

        # 1. Load coastline
        print("[1/10] Cargando linea costera...")
        n_points = self.load_coastline(coastline_path)
        print(f"       {n_points} puntos cargados")

        # 2. Train with historical data
        print("\n[2/10] Entrenando modelo con datos historicos...")
        train_stats = self.train_with_historical_data(
            start_date=(datetime.now() - timedelta(days=365 * train_years)).strftime("%Y-%m-%d"),
            end_date=datetime.now().strftime("%Y-%m-%d")
        )

        # 3. Sample spots
        print("\n[3/10] Muestreando puntos de pesca...")
        spots = self.sample_fishing_spots()
        print(f"       {len(spots)} spots muestreados")

        # 4. Generate fish zones
        print("\n[4/10] Generando zonas de peces...")
        zones = self.generate_fish_zones()
        print(f"       {len(zones)} zonas base generadas")

        # 5. Predict anchovy migration (NEW)
        print("\n[5/10] Prediciendo migracion de anchoveta...")
        anchovy_zones = self.predict_anchovy_migration(
            target_date=datetime.now().strftime('%Y-%m-%d'),
            target_hour=target_hour
        )

        # 6. Fetch marine data
        print("\n[6/10] Obteniendo datos marinos...")
        self.fetch_marine_data()

        # 7. Predict fish movement
        print("\n[7/10] Prediciendo movimiento de peces (24h)...")
        future = self.predict_fish_movement(hours_ahead=24)
        print(f"       {len(future)} predicciones de movimiento")

        # 8. Get conditions
        print("\n[8/10] Obteniendo condiciones...")
        self.get_conditions()

        # 9. ML prediction (now supervised)
        print("\n[9/10] Ejecutando prediccion ML supervisada...")
        self.run_ml_prediction()

        # 10. Analyze and create map
        print("\n[10/10] Analizando y generando mapa...")
        results = self.analyze_spots()
        self.create_map(output_path)

        self._print_results(results[:5])

        # Print anchovy info
        if self.anchovy_zones:
            print(f"\n{'=' * 60}")
            print("ZONAS DE ANCHOVETA PREDICHAS")
            print("=" * 60)
            for i, z in enumerate(self.anchovy_zones[:5]):
                print(f"  #{i+1} Score:{z['score']:.0f} | {z['lat']:.3f}, {z['lon']:.3f} | SST:{z['avg_sst']:.1f}C | Hist:{z['historical_hours']:.0f}h")

        print(f"\n{'=' * 60}")
        print(f"MAPA GUARDADO: {output_path}")
        print(f"MODO: {'SUPERVISADO' if self.is_supervised_mode else 'NO SUPERVISADO'}")
        print(f"ZONAS ANCHOVETA: {len(self.anchovy_zones)}")
        print("=" * 60)

        return {
            'spots': results,
            'zones': zones,
            'anchovy_zones': self.anchovy_zones,
            'future_hotspots': future,
            'training_stats': train_stats,
            'map_path': output_path
        }
