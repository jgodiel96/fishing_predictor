"""
Obtención de datos marinos reales para predicción de pesca.
Usa Open-Meteo Marine API para SST y condiciones del mar.

Basado en investigación de:
- pyBOA (Belkin-O'Reilly Algorithm) para detección de frentes térmicos
- Copernicus Marine Service para datos oceanográficos
- NOAA OISST para SST histórico
- Investigación de Humboldt Current System (15% de pesca global)
"""

import requests
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import json
from pathlib import Path

# Import domain constants
from domain import HOTSPOTS, SPECIES_BY_NAME, THRESHOLDS, ENDPOINTS


@dataclass
class MarinePoint:
    """Punto con datos marinos."""
    lat: float
    lon: float
    sst: float  # Sea Surface Temperature (°C)
    wave_height: float  # metros
    wave_period: float  # segundos
    current_speed: float  # m/s
    current_direction: float  # grados
    timestamp: str = ""  # Fecha/hora de la medición


@dataclass
class CurrentVector:
    """Vector de corriente oceánica para visualización de flujo."""
    lat: float
    lon: float
    u: float  # Componente este-oeste (m/s)
    v: float  # Componente norte-sur (m/s)
    speed: float  # Velocidad total (m/s)
    direction: float  # Dirección hacia donde fluye (grados)


@dataclass
class SSTHistory:
    """Historial de SST para un punto."""
    lat: float
    lon: float
    dates: List[str] = field(default_factory=list)
    temperatures: List[float] = field(default_factory=list)
    trend: float = 0.0  # Tendencia: positivo=calentando, negativo=enfriando


@dataclass
class ThermalFront:
    """Frente térmico detectado."""
    lat: float
    lon: float
    gradient: float  # °C/km
    direction: float  # dirección del gradiente
    intensity: float  # 0-1, normalizado


class MarineDataFetcher:
    """
    Obtiene datos marinos de múltiples fuentes.

    Prioridad de datos:
    1. Copernicus Marine Service (datos históricos de alta calidad)
    2. Open-Meteo Marine API (datos en tiempo real)

    Capacidades:
    - SST en tiempo real y histórico
    - Corrientes oceánicas (velocidad y dirección) desde Copernicus
    - Altura y período de olas desde Copernicus
    """

    BASE_URL = ENDPOINTS.openmeteo_marine

    def __init__(self, cache_dir: str = None, use_copernicus: bool = True):
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent / "data" / "cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Almacenar vectores de corriente para visualización
        self.current_vectors: List[CurrentVector] = []
        # Almacenar historial SST
        self.sst_history: List[SSTHistory] = []
        # Puntos marinos muestreados (para visualización)
        self.sampled_points: List[MarinePoint] = []

        # Proveedor de datos de Copernicus
        self.use_copernicus = use_copernicus
        self._copernicus_provider = None
        if use_copernicus:
            try:
                from core.copernicus_data_provider import CopernicusDataProvider
                self._copernicus_provider = CopernicusDataProvider()
            except ImportError:
                self._copernicus_provider = None

    def fetch_from_copernicus(self, date: str) -> List[MarinePoint]:
        """
        Obtiene datos marinos desde Copernicus Marine Service.

        Args:
            date: Fecha en formato YYYY-MM-DD

        Returns:
            Lista de MarinePoint con datos reales de Copernicus
        """
        if self._copernicus_provider is None:
            print("[WARN] Copernicus provider no disponible")
            return []

        print(f"[INFO] Cargando datos de Copernicus para {date}...")
        ocean_points = self._copernicus_provider.get_data_for_date(date)

        if not ocean_points:
            print("[WARN] No hay datos de Copernicus para esta fecha")
            return []

        # Convertir a MarinePoint
        points = []
        for op in ocean_points:
            if op.sst is None:
                continue

            point = MarinePoint(
                lat=op.lat,
                lon=op.lon,
                sst=op.sst,
                wave_height=op.wave_height if op.wave_height else 1.0,
                wave_period=op.wave_period if op.wave_period else 8.0,
                current_speed=op.current_speed if op.current_speed else 0.1,
                current_direction=op.current_direction if op.current_direction else 180.0,
                timestamp=date
            )
            points.append(point)
            self.sampled_points.append(point)

            # Crear vector de corriente para visualización
            if op.uo is not None and op.vo is not None:
                self.current_vectors.append(CurrentVector(
                    lat=op.lat,
                    lon=op.lon,
                    u=op.uo,
                    v=op.vo,
                    speed=op.current_speed or np.sqrt(op.uo**2 + op.vo**2),
                    direction=op.current_direction or np.degrees(np.arctan2(op.vo, op.uo))
                ))

        print(f"[OK] {len(points)} puntos cargados desde Copernicus")
        print(f"     - SST: {sum(1 for p in points if p.sst)} puntos")
        print(f"     - Corrientes: {len(self.current_vectors)} vectores")
        print(f"     - Olas: {sum(1 for op in ocean_points if op.wave_height)} puntos")

        return points

    def fetch_grid(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        resolution: float = 0.1,  # grados (~11km)
        date: str = None  # Fecha para datos de Copernicus
    ) -> List[MarinePoint]:
        """
        Obtiene datos marinos en una grilla.

        Args:
            lat_min, lat_max: rango de latitud
            lon_min, lon_max: rango de longitud
            resolution: resolución en grados
            date: Fecha para usar datos de Copernicus (YYYY-MM-DD)

        Returns:
            Lista de MarinePoint con datos
        """
        # Si hay fecha y Copernicus disponible, usar datos reales
        if date and self._copernicus_provider:
            copernicus_points = self.fetch_from_copernicus(date)
            if copernicus_points:
                # Filtrar por región
                filtered = [
                    p for p in copernicus_points
                    if lat_min <= p.lat <= lat_max and lon_min <= p.lon <= lon_max
                ]
                if filtered:
                    return filtered

        # Fallback a Open-Meteo API
        points = []

        # Generar grilla
        lats = np.arange(lat_min, lat_max, resolution)
        lons = np.arange(lon_min, lon_max, resolution)

        print(f"[INFO] Obteniendo datos marinos para {len(lats) * len(lons)} puntos...")

        for lat in lats:
            for lon in lons:
                data = self._fetch_point(lat, lon)
                if data:
                    points.append(data)

        print(f"[OK] {len(points)} puntos con datos marinos")
        return points

    def fetch_points(self, coordinates: List[Tuple[float, float]]) -> List[MarinePoint]:
        """
        Obtiene datos marinos para una lista de coordenadas.
        """
        points = []

        # Agrupar en batches para eficiencia
        batch_size = 10
        total = len(coordinates)

        for i in range(0, total, batch_size):
            batch = coordinates[i:i+batch_size]

            for lat, lon in batch:
                data = self._fetch_point(lat, lon)
                if data:
                    points.append(data)

            if (i + batch_size) % 50 == 0:
                print(f"[INFO] Procesados {min(i + batch_size, total)}/{total} puntos")

        return points

    def _fetch_point(self, lat: float, lon: float) -> Optional[MarinePoint]:
        """
        Obtiene datos marinos para un punto.
        """
        # Verificar cache
        cache_key = f"marine_{lat:.2f}_{lon:.2f}_{datetime.now().strftime('%Y%m%d%H')}"
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            with open(cache_file, 'r') as f:
                data = json.load(f)
                return MarinePoint(**data)

        params = {
            "latitude": lat,
            "longitude": lon,
            "current": [
                "sea_surface_temperature",
                "wave_height",
                "wave_period",
                "ocean_current_velocity",
                "ocean_current_direction"
            ],
            "timezone": "America/Lima"
        }

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                current = data.get("current", {})

                point = MarinePoint(
                    lat=lat,
                    lon=lon,
                    sst=current.get("sea_surface_temperature", 17.0),
                    wave_height=current.get("wave_height", 1.0),
                    wave_period=current.get("wave_period", 8.0),
                    current_speed=current.get("ocean_current_velocity", 0.1),
                    current_direction=current.get("ocean_current_direction", 180)
                )

                # Guardar en cache
                with open(cache_file, 'w') as f:
                    json.dump({
                        "lat": point.lat,
                        "lon": point.lon,
                        "sst": point.sst,
                        "wave_height": point.wave_height,
                        "wave_period": point.wave_period,
                        "current_speed": point.current_speed,
                        "current_direction": point.current_direction
                    }, f)

                return point

        except Exception as e:
            # Silenciosamente retornar None si falla
            pass

        return None

    def fetch_sst_history(
        self,
        lat: float,
        lon: float,
        days: int = 7
    ) -> Optional[SSTHistory]:
        """
        Obtiene historial de SST para un punto (últimos N días).

        Args:
            lat, lon: coordenadas
            days: número de días hacia atrás

        Returns:
            SSTHistory con fechas y temperaturas
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "sea_surface_temperature",
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "timezone": "America/Lima"
        }

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=15)

            if response.status_code == 200:
                data = response.json()
                hourly = data.get("hourly", {})

                times = hourly.get("time", [])
                temps = hourly.get("sea_surface_temperature", [])

                if times and temps:
                    # Filtrar valores None
                    valid_data = [(t, temp) for t, temp in zip(times, temps) if temp is not None]

                    if valid_data:
                        dates = [d[0] for d in valid_data]
                        temperatures = [d[1] for d in valid_data]

                        # Calcular tendencia (pendiente de regresión lineal simple)
                        if len(temperatures) > 1:
                            x = np.arange(len(temperatures))
                            trend = np.polyfit(x, temperatures, 1)[0]  # Pendiente
                        else:
                            trend = 0.0

                        history = SSTHistory(
                            lat=lat,
                            lon=lon,
                            dates=dates,
                            temperatures=temperatures,
                            trend=trend
                        )
                        self.sst_history.append(history)
                        return history

        except Exception as e:
            print(f"[WARN] Error obteniendo historial SST: {e}")

        return None

    def fetch_current_vectors(
        self,
        coordinates: List[Tuple[float, float]]
    ) -> List[CurrentVector]:
        """Obtiene vectores de corriente para visualización de flujo."""
        self.current_vectors = []

        for lat, lon in coordinates:
            if not (point := self._fetch_point(lat, lon)):
                continue

            self.sampled_points.append(point)

            # Validar datos de corriente
            if point.current_speed is None or point.current_direction is None:
                continue

            # Convertir a componentes U, V
            rad = np.radians(point.current_direction)
            self.current_vectors.append(CurrentVector(
                lat=lat,
                lon=lon,
                u=point.current_speed * np.sin(rad),
                v=point.current_speed * np.cos(rad),
                speed=point.current_speed,
                direction=point.current_direction
            ))

        return self.current_vectors

    def get_flow_lines(self, num_steps: int = 5, step_km: float = 2.0) -> List[List[Tuple[float, float]]]:
        """Genera líneas de flujo simulando trayectorias de partículas."""
        flow_lines = []
        step_deg = step_km / 111.0  # km a grados

        for vector in self.current_vectors:
            line = [(vector.lat, vector.lon)]
            lat, lon, direction = vector.lat, vector.lon, vector.direction

            for _ in range(num_steps):
                rad = np.radians(direction)
                lat += step_deg * np.cos(rad)
                lon += step_deg * np.sin(rad) / np.cos(np.radians(lat))
                line.append((lat, lon))

            flow_lines.append(line)

        return flow_lines


class ThermalFrontDetector:
    """
    Detecta frentes térmicos a partir de datos SST.
    Los peces se concentran en frentes térmicos.
    """

    def __init__(self, gradient_threshold: float = 0.3):
        """
        Args:
            gradient_threshold: °C/km mínimo para considerar frente
        """
        self.gradient_threshold = gradient_threshold

    def detect_fronts(
        self,
        marine_points: List[MarinePoint],
        min_intensity: float = 0.3
    ) -> List[ThermalFront]:
        """
        Detecta frentes térmicos en los datos.

        Args:
            marine_points: datos marinos
            min_intensity: intensidad mínima para reportar

        Returns:
            Lista de frentes térmicos detectados
        """
        if len(marine_points) < 4:
            return []

        fronts = []

        # Crear matriz de SST
        lats = sorted(set(p.lat for p in marine_points))
        lons = sorted(set(p.lon for p in marine_points))

        # Mapear puntos
        sst_map = {}
        for p in marine_points:
            sst_map[(round(p.lat, 2), round(p.lon, 2))] = p.sst

        # Calcular gradientes
        for p in marine_points:
            gradient, direction = self._calculate_gradient(
                p.lat, p.lon, sst_map
            )

            if gradient >= self.gradient_threshold:
                # Normalizar intensidad
                intensity = min(1.0, gradient / 1.0)  # 1°C/km = intensidad máxima

                if intensity >= min_intensity:
                    fronts.append(ThermalFront(
                        lat=p.lat,
                        lon=p.lon,
                        gradient=gradient,
                        direction=direction,
                        intensity=intensity
                    ))

        # Ordenar por intensidad
        fronts.sort(key=lambda f: f.intensity, reverse=True)

        return fronts

    def _calculate_gradient(
        self,
        lat: float,
        lon: float,
        sst_map: Dict[Tuple[float, float], float],
        delta: float = 0.1  # ~11km
    ) -> Tuple[float, float]:
        """
        Calcula gradiente de SST en un punto.

        Returns:
            (gradiente en °C/km, dirección del gradiente)
        """
        key = (round(lat, 2), round(lon, 2))
        if key not in sst_map:
            return 0.0, 0.0

        center_sst = sst_map[key]

        # Buscar vecinos
        neighbors = [
            ((round(lat + delta, 2), round(lon, 2)), 0),      # Norte
            ((round(lat - delta, 2), round(lon, 2)), 180),    # Sur
            ((round(lat, 2), round(lon + delta, 2)), 90),     # Este
            ((round(lat, 2), round(lon - delta, 2)), 270),    # Oeste
        ]

        max_gradient = 0.0
        gradient_dir = 0.0

        for neighbor_key, direction in neighbors:
            if neighbor_key in sst_map:
                neighbor_sst = sst_map[neighbor_key]
                diff = abs(center_sst - neighbor_sst)

                # Convertir a °C/km (delta grados ≈ 11km)
                gradient = diff / (delta * 111)

                if gradient > max_gradient:
                    max_gradient = gradient
                    # La dirección apunta hacia agua más fría
                    if neighbor_sst < center_sst:
                        gradient_dir = direction
                    else:
                        gradient_dir = (direction + 180) % 360

        return max_gradient, gradient_dir


class FishZonePredictor:
    """
    Predice zonas de actividad de peces basado en datos reales.
    Uses centralized HOTSPOTS and SPECIES from domain.py
    """

    def __init__(self):
        self.marine_fetcher = MarineDataFetcher()
        self.front_detector = ThermalFrontDetector()

    def predict_zones(
        self,
        coastline_points: List[Tuple[float, float]],
        num_zones: int = 6
    ) -> List[Dict]:
        """
        Predice zonas de peces basado en datos reales.

        Args:
            coastline_points: puntos de la costa (lat, lon)
            num_zones: número de zonas a retornar

        Returns:
            Lista de zonas con predicción
        """
        print("[INFO] Obteniendo datos marinos reales...")

        # Siempre incluir hotspots históricos primero
        zones = self._get_historical_zones()

        # Intentar obtener datos marinos adicionales
        try:
            lats = [p[0] for p in coastline_points]
            lons = [p[1] for p in coastline_points]

            # Área marina (hacia el oeste de la costa)
            lat_min, lat_max = min(lats) - 0.1, max(lats) + 0.1
            lon_min = min(lons) - 0.3  # 30km hacia el mar
            lon_max = min(lons) + 0.05  # Cerca de la costa

            # Generar puntos de muestreo en el mar
            sample_points = []
            for lat in np.arange(lat_min, lat_max, 0.2):
                for lon in np.arange(lon_min, lon_max, 0.15):
                    sample_points.append((lat, lon))

            print(f"[INFO] Muestreando {len(sample_points)} puntos marinos...")

            # Obtener datos marinos
            marine_data = self.marine_fetcher.fetch_points(sample_points)

            if marine_data:
                print(f"[OK] {len(marine_data)} puntos con datos SST")

                # Actualizar zonas históricas con SST real
                for zone in zones:
                    nearby = [p for p in marine_data
                              if abs(p.lat - zone['lat']) < 0.2 and abs(p.lon - zone['lon']) < 0.2]
                    if nearby:
                        zone['sst'] = np.mean([p.sst for p in nearby])
                        zone['sst_source'] = 'real'

                # Detectar frentes térmicos y agregar
                fronts = self.front_detector.detect_fronts(marine_data, min_intensity=0.2)
                print(f"[OK] {len(fronts)} frentes térmicos detectados")

                for i, front in enumerate(fronts[:3]):
                    zones.append({
                        'id': len(zones) + 1,
                        'lat': front.lat,
                        'lon': front.lon,
                        'radius': 200 + front.intensity * 200,
                        'intensity': front.intensity,
                        'movement_direction': front.direction,
                        'cause': 'thermal_front',
                        'sst_gradient': front.gradient,
                        'sst_source': 'real'
                    })

        except Exception as e:
            print(f"[WARN] Error obteniendo datos marinos: {e}")

        # Ordenar por intensidad y retornar
        zones.sort(key=lambda z: z['intensity'], reverse=True)
        print(f"[OK] {len(zones)} zonas de peces identificadas")

        return zones[:num_zones]

    def _get_historical_zones(self) -> List[Dict]:
        """
        Retorna zonas basadas en patrones históricos REALES conocidos.

        Datos basados en:
        - Reportes de IMARPE (Instituto del Mar del Perú)
        - Conocimiento de pescadores locales documentado
        - Cartas náuticas de zonas de pesca

        Uses centralized HOTSPOTS from domain.py
        """
        zones = []

        for i, hotspot in enumerate(HOTSPOTS):
            # Mover punto hacia el mar (~2km)
            zone_lon = hotspot.lon - 0.02

            # Intensidad base con bonus histórico
            base_intensity = 0.6 * hotspot.bonus_factor

            zones.append({
                'id': i + 1,
                'lat': hotspot.lat,
                'lon': zone_lon,
                'radius': 250,
                'intensity': min(1.0, base_intensity),
                'movement_direction': 90 + np.random.normal(0, 15),  # Hacia la costa
                'cause': f'historical_imarpe ({hotspot.name})',
                'sst': THRESHOLDS.sst_optimal_center,
                'sst_source': 'climatology_imarpe'
            })

        return zones

    def _calculate_sst_score(self, sst: float) -> float:
        """
        Calcula score basado en SST usando THRESHOLDS de domain.py.
        """
        optimal_min = THRESHOLDS.sst_optimal_min
        optimal_max = THRESHOLDS.sst_optimal_max
        center = THRESHOLDS.sst_optimal_center

        if optimal_min <= sst <= optimal_max:
            distance = abs(sst - center)
            half_range = (optimal_max - optimal_min) / 2
            return 1.0 - (distance / half_range) * 0.5
        elif sst < optimal_min:
            return max(0.2, 1.0 - (optimal_min - sst) / 5)
        else:
            return max(0.2, 1.0 - (sst - optimal_max) / 5)

    def _fallback_prediction(
        self,
        coastline: List[Tuple[float, float]],
        num_zones: int
    ) -> List[Dict]:
        """
        Predicción de respaldo usando patrones históricos REALES de IMARPE.

        Uses centralized HOTSPOTS from domain.py
        """
        print("[INFO] Usando predicción de respaldo (patrones históricos IMARPE)")

        zones = []

        for i, hotspot in enumerate(HOTSPOTS[:num_zones]):
            zones.append({
                'id': i + 1,
                'lat': hotspot.lat,
                'lon': hotspot.lon - 0.02,
                'radius': 200,
                'intensity': 0.5 + hotspot.bonus_factor * 0.3,
                'movement_direction': 90 + np.random.normal(0, 30),
                'cause': f'historical_imarpe ({hotspot.name})',
                'sst': THRESHOLDS.sst_optimal_center,
                'sst_source': 'climatology_imarpe'
            })

        return zones
