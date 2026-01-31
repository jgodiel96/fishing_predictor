"""
Analysis Controller - Orchestrates data fetching, ML prediction, and visualization.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

# Import centralized data configuration
try:
    from data.data_config import DataConfig, LEGACY_DB
except ImportError:
    # Fallback if data_config not available
    DataConfig = None
    LEGACY_DB = Path("data/real_only/real_data_100.db")

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

# Optional: Historical data for supervised learning
try:
    from data.fetchers.historical_fetcher import HistoricalDataFetcher, FishMovementPredictor
    HISTORICAL_AVAILABLE = True
except ImportError:
    HISTORICAL_AVAILABLE = False

# Timeline data availability
try:
    _db_path = LEGACY_DB if DataConfig is None else DataConfig.LEGACY_DB
    TIMELINE_AVAILABLE = _db_path.exists()
except:
    TIMELINE_AVAILABLE = False


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

        # Historical data for supervised learning
        self.historical_fetcher: Optional['HistoricalDataFetcher'] = None
        self.is_supervised_mode: bool = False

        # Results
        self.ml_predictions: List = []
        self.pca_analysis: Dict = {}
        self.conditions: Dict = {}

    def load_coastline(self, geojson_path: str) -> int:
        """Load coastline from GeoJSON file."""
        with open(geojson_path, 'r') as f:
            data = json.load(f)

        self.coastline_points = []
        for feature in data.get('features', []):
            geom = feature.get('geometry', {})
            coords = []

            if geom.get('type') == 'LineString':
                coords = geom.get('coordinates', [])
            elif geom.get('type') == 'MultiLineString':
                for line in geom.get('coordinates', []):
                    coords.extend(line)

            for coord in coords:
                self.coastline_points.append((coord[1], coord[0]))  # lat, lon

        self.coastline_points.sort(key=lambda x: x[0])
        self.coastline_points = self._remove_duplicates(self.coastline_points)

        return len(self.coastline_points)

    def sample_fishing_spots(self, num_spots: int = 35) -> List[Dict]:
        """Sample spots along the coastline."""
        if len(self.coastline_points) <= num_spots:
            indices = range(len(self.coastline_points))
        else:
            indices = np.linspace(0, len(self.coastline_points) - 1, num_spots, dtype=int)

        self.sampled_spots = []
        for i, idx in enumerate(indices):
            lat, lon = self.coastline_points[idx]
            bearing = self._perpendicular_to_sea(idx)

            self.sampled_spots.append({
                'id': i + 1,
                'lat': lat,
                'lon': lon,
                'bearing_to_sea': bearing,
                'score': 0,
                'distance_to_fish': 0,
                'direction_to_fish': 0,
                'species': self._get_species(lat)
            })

        return self.sampled_spots

    def fetch_marine_data(self) -> int:
        """Fetch marine data and generate flow lines following coast contour."""
        print("[INFO] Generando muestreo paralelo a la costa...")

        coast_step = max(1, len(self.coastline_points) // 25)
        coast_samples = self.coastline_points[::coast_step]

        # Generate offshore points perpendicular to coast
        offshore_km = [3, 8, 15, 25]
        sample_points = []

        for i, (lat, lon) in enumerate(coast_samples):
            bearing = self._perpendicular_to_sea(i * coast_step)
            for dist in offshore_km:
                rad = np.radians(bearing)
                new_lat = lat + dist / 111.0 * np.cos(rad)
                new_lon = lon + dist / 111.0 * np.sin(rad) / np.cos(np.radians(lat))
                sample_points.append((new_lat, new_lon))

        print(f"[INFO] Muestreando {len(sample_points)} puntos...")

        self.marine_fetcher = MarineDataFetcher()
        self.current_vectors = self.marine_fetcher.fetch_current_vectors(sample_points)
        self.flow_lines = self.marine_fetcher.get_flow_lines(num_steps=5, step_km=4.0)
        self.marine_points = self.marine_fetcher.sampled_points

        print(f"[OK] {len(self.current_vectors)} vectores, {len(self.flow_lines)} lineas de flujo")
        return len(self.current_vectors)

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

    def analyze_spots(self) -> List[Dict]:
        """Analyze and score fishing spots with tide integration (V4)."""
        if not self.fish_zones:
            self.generate_fish_zones()

        # Get tide score (0-1) - fallback to 0.5 if not available
        tide_score = self.tide_data.get('tide_score', 0.5) if self.tide_data else 0.5
        tide_phase = self.tide_data.get('tide_phase', 'unknown') if self.tide_data else 'unknown'

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

            # Apply tide bonus (V4): up to +15 points for excellent tide conditions
            tide_bonus = (tide_score - 0.5) * 30  # Range: -15 to +15

            spot['score'] = min(100, max(0, best_score + ml_boost + tide_bonus))
            spot['distance_to_fish'] = best_dist
            spot['direction_to_fish'] = best_dir
            spot['tide_phase'] = tide_phase
            spot['tide_score'] = tide_score

        self.sampled_spots.sort(key=lambda s: s['score'], reverse=True)
        return self.sampled_spots

    def get_conditions(self) -> Dict:
        """Get weather, solunar, and tide conditions."""
        if not self.coastline_points:
            return {}

        center_lat = np.mean([p[0] for p in self.coastline_points])
        center_lon = np.mean([p[1] for p in self.coastline_points])

        self.conditions = get_fishing_conditions(center_lat, center_lon)

        # Add tide data (V4)
        self._fetch_tide_data(center_lat, center_lon)
        if self.tide_data:
            self.conditions['tide'] = self.tide_data

        return self.conditions

    def _fetch_tide_data(self, lat: float, lon: float) -> Dict:
        """Fetch tide data for current time and location."""
        if not TIDES_AVAILABLE:
            self.tide_data = {}
            return {}

        try:
            if not self.tide_fetcher:
                self.tide_fetcher = TideFetcher()

            now = datetime.now()
            today = now.strftime('%Y-%m-%d')

            # Get tide state for current time
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

    def create_map(self, output_path: str = "output/analysis_map.html") -> str:
        """Create visualization map."""
        if not self.coastline_points:
            raise ValueError("No coastline data loaded")

        center = (
            np.mean([p[0] for p in self.coastline_points]),
            np.mean([p[1] for p in self.coastline_points])
        )

        self.map_view.create_map(center=center, zoom=10)
        self.map_view.add_coastline(self.coastline_points)
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

        self.map_view.finalize()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        self.map_view.save(output_path)

        return output_path

    def run_full_analysis(self, coastline_path: str, output_path: str = "output/analysis_map.html") -> Dict:
        """Run complete analysis pipeline."""
        print("=" * 60)
        print("   ANALISIS DE PESCA CON ML")
        print("=" * 60)
        print(f"Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M')}\n")

        # 1. Load coastline
        print("[1/8] Cargando linea costera...")
        n_points = self.load_coastline(coastline_path)
        print(f"      {n_points} puntos cargados")

        # 2. Sample spots
        print("\n[2/8] Muestreando puntos de pesca...")
        spots = self.sample_fishing_spots()
        print(f"      {len(spots)} spots muestreados")

        # 3. Generate fish zones
        print("\n[3/8] Generando zonas de peces...")
        zones = self.generate_fish_zones()
        print(f"      {len(zones)} zonas generadas")

        # 4. Predict anchovy migration
        print("\n[4/8] Prediciendo zonas de anchoveta...")
        anchovy_zones = self.predict_anchovy_migration(
            target_date=datetime.now().strftime('%Y-%m-%d'),
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

        # 8. Analyze spots
        print("\n[8/8] Analizando spots...")
        results = self.analyze_spots()

        weather = conditions.get('weather', {})
        solunar = conditions.get('solunar', {})
        print(f"      Temp: {weather.get('temperature', 'N/A')}C")
        print(f"      Luna: {solunar.get('moon_phase', 'N/A')}")

        # Create map
        print("\nGenerando mapa...")
        map_path = self.create_map(output_path)

        # Print results
        self._print_results(results[:5])

        print(f"\n{'=' * 60}")
        print(f"MAPA GUARDADO: {map_path}")
        print("=" * 60)

        return {
            'spots': results,
            'zones': zones,
            'pca': pca_results,
            'conditions': conditions,
            'map_path': map_path
        }

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
        if len(self.coastline_points) < 2:
            return 270

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
        for perp in [perp1, perp2]:
            rad = np.radians(perp)
            test_lon = lon + 100 / (111000 * np.cos(np.radians(lat))) * np.sin(rad)
            if test_lon < lon:
                return perp

        return perp1

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

    def _print_results(self, top_spots: List[Dict]):
        print(f"\n{'=' * 60}")
        print("TOP 5 MEJORES PUNTOS DE PESCA")
        print("=" * 60)

        for i, spot in enumerate(top_spots):
            emoji = "*" if i == 0 else "#"
            print(f"\n{emoji}{i+1} - Score: {spot['score']:.1f}/100")
            print(f"   Coords: {spot['lat']:.6f}, {spot['lon']:.6f}")
            print(f"   Dist. peces: {spot['distance_to_fish']:.0f}m")
            if spot.get('species'):
                names = [s['name'] for s in spot['species']]
                print(f"   Especies: {', '.join(names)}")

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
