"""
Optimizador de puntos de pesca basado en modelo geométrico.

El modelo considera:
1. Línea de orilla como curva (límite mar/playa)
2. Zonas de actividad de peces en el agua
3. Vectores de distancia desde orilla a zonas de peces
4. Dirección de movimiento de peces
5. Optimización para encontrar el mejor punto de pesca
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from datetime import datetime
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import cdist
import json


@dataclass
class ShorePoint:
    """Punto en la línea de orilla."""
    id: int
    lat: float
    lon: float

    # Calculados
    distance_to_fish_zone: float = 0.0  # Distancia a zona de peces más cercana
    fish_zone_direction: float = 0.0     # Dirección hacia zona de peces (grados)
    score: float = 0.0                   # Score de pesca (0-100)

    # Datos oceanográficos del transecto
    sst_gradient: float = 0.0
    chlorophyll_level: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "lat": self.lat,
            "lon": self.lon,
            "distance_to_fish_zone_m": round(self.distance_to_fish_zone, 1),
            "fish_zone_direction": round(self.fish_zone_direction, 1),
            "score": round(self.score, 1),
            "sst_gradient": round(self.sst_gradient, 3),
            "chlorophyll": round(self.chlorophyll_level, 2)
        }


@dataclass
class FishZone:
    """Zona de actividad de peces detectada."""
    id: int
    center_lat: float
    center_lon: float
    radius_m: float  # Radio aproximado de la zona

    # Características
    intensity: float = 0.0      # Intensidad de actividad (0-1)
    movement_direction: float = 0.0  # Dirección de movimiento (grados)
    movement_speed: float = 0.0      # Velocidad relativa

    # Causa de la zona
    cause: str = ""  # "thermal_front", "chlorophyll_hotspot", "current_convergence"

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "center": (self.center_lat, self.center_lon),
            "radius_m": self.radius_m,
            "intensity": round(self.intensity, 2),
            "movement_direction": round(self.movement_direction, 1),
            "cause": self.cause
        }


class CoastlineCurve:
    """
    Modelo de línea de orilla como curva suave.
    Usa interpolación spline para crear una curva continua.
    """

    def __init__(self):
        self.reference_points: List[Tuple[float, float]] = []
        self.curve_points: List[Tuple[float, float]] = []
        self.tck = None  # Parámetros del spline

    def set_reference_points(self, points: List[Tuple[float, float]]):
        """
        Define puntos de referencia de la orilla.

        Args:
            points: Lista de (lat, lon) que definen la orilla
        """
        self.reference_points = points

        if len(points) >= 4:
            # Usar spline para suavizar
            lats = [p[0] for p in points]
            lons = [p[1] for p in points]

            try:
                # Crear spline paramétrico
                self.tck, _ = splprep([lats, lons], s=0.0001, k=min(3, len(points)-1))
            except Exception:
                self.tck = None
        else:
            self.tck = None

    def generate_curve(self, num_points: int = 50) -> List[Tuple[float, float]]:
        """
        Genera puntos a lo largo de la curva de orilla.

        Args:
            num_points: Número de puntos a generar

        Returns:
            Lista de (lat, lon) a lo largo de la curva
        """
        if self.tck is not None:
            # Evaluar spline
            u_new = np.linspace(0, 1, num_points)
            lats, lons = splev(u_new, self.tck)
            self.curve_points = list(zip(lats, lons))
        elif len(self.reference_points) >= 2:
            # Interpolación lineal simple
            self.curve_points = self._linear_interpolate(num_points)
        else:
            self.curve_points = self.reference_points.copy()

        return self.curve_points

    def _linear_interpolate(self, num_points: int) -> List[Tuple[float, float]]:
        """Interpolación lineal entre puntos de referencia."""
        if len(self.reference_points) < 2:
            return self.reference_points

        total_segments = len(self.reference_points) - 1
        points_per_segment = max(1, num_points // total_segments)

        result = []
        for i in range(total_segments):
            p1 = self.reference_points[i]
            p2 = self.reference_points[i + 1]

            for j in range(points_per_segment):
                t = j / points_per_segment
                lat = p1[0] + t * (p2[0] - p1[0])
                lon = p1[1] + t * (p2[1] - p1[1])
                result.append((lat, lon))

        result.append(self.reference_points[-1])
        return result

    def get_point_at_distance(self, start_idx: int, distance_m: float, direction: float) -> Tuple[float, float]:
        """
        Calcula un punto a cierta distancia y dirección desde un punto de la curva.

        Args:
            start_idx: Índice del punto de origen en la curva
            distance_m: Distancia en metros
            direction: Dirección en grados (0=N, 90=E, 180=S, 270=W)

        Returns:
            (lat, lon) del punto calculado
        """
        if start_idx >= len(self.curve_points):
            start_idx = len(self.curve_points) - 1

        lat, lon = self.curve_points[start_idx]

        # Convertir dirección a radianes
        dir_rad = np.radians(direction)

        # Calcular desplazamiento
        dlat = distance_m / 111000 * np.cos(dir_rad)
        dlon = distance_m / (111000 * np.cos(np.radians(lat))) * np.sin(dir_rad)

        return (lat + dlat, lon + dlon)

    def calculate_perpendicular_direction(self, idx: int) -> float:
        """
        Calcula la dirección perpendicular a la costa (hacia el mar) en un punto.

        Args:
            idx: Índice del punto en la curva

        Returns:
            Dirección en grados hacia el mar
        """
        if len(self.curve_points) < 2:
            return 270.0  # Default: oeste

        # Usar puntos vecinos para calcular tangente
        if idx == 0:
            p1, p2 = self.curve_points[0], self.curve_points[1]
        elif idx >= len(self.curve_points) - 1:
            p1, p2 = self.curve_points[-2], self.curve_points[-1]
        else:
            p1, p2 = self.curve_points[idx - 1], self.curve_points[idx + 1]

        # Dirección de la costa
        dlat = p2[0] - p1[0]
        dlon = p2[1] - p1[1]
        coast_direction = np.degrees(np.arctan2(dlon, dlat)) % 360

        # Perpendicular hacia el mar (costa peruana: mar al oeste)
        perp1 = (coast_direction + 90) % 360
        perp2 = (coast_direction - 90) % 360

        # Elegir la que apunta más hacia el oeste (mar)
        if 180 <= perp1 <= 360 or perp1 < 90:
            return perp1 if 180 <= perp1 <= 315 else perp2
        return perp2 if 180 <= perp2 <= 315 else perp1


class FishZoneDetector:
    """
    Detecta zonas de actividad de peces basándose en:
    - Frentes térmicos (gradientes de SST)
    - Concentraciones de clorofila (alimento)
    - Convergencia de corrientes
    """

    def __init__(self):
        self.zones: List[FishZone] = []

    def detect_from_sst_data(
        self,
        sst_grid: np.ndarray,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        gradient_threshold: float = 0.3
    ) -> List[FishZone]:
        """
        Detecta zonas de peces basadas en gradientes de SST.

        Los peces pelágicos se concentran en frentes térmicos
        donde hay cambios bruscos de temperatura.
        """
        if sst_grid is None or sst_grid.size == 0:
            return []

        # Calcular gradiente
        grad_y, grad_x = np.gradient(sst_grid)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Encontrar máximos locales del gradiente
        zones = []
        zone_id = 0

        rows, cols = sst_grid.shape
        lat_step = (lat_range[1] - lat_range[0]) / rows
        lon_step = (lon_range[1] - lon_range[0]) / cols

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if gradient_magnitude[i, j] > gradient_threshold:
                    # Es un punto de gradiente alto
                    local_max = True
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if gradient_magnitude[i + di, j + dj] > gradient_magnitude[i, j]:
                                local_max = False
                                break

                    if local_max:
                        lat = lat_range[0] + i * lat_step
                        lon = lon_range[0] + j * lon_step

                        # Dirección del gradiente (peces se mueven perpendicular al frente)
                        direction = np.degrees(np.arctan2(grad_x[i, j], grad_y[i, j])) % 360

                        zone_id += 1
                        zones.append(FishZone(
                            id=zone_id,
                            center_lat=lat,
                            center_lon=lon,
                            radius_m=500,  # Radio estimado
                            intensity=min(1.0, gradient_magnitude[i, j] / 1.0),
                            movement_direction=direction,
                            cause="thermal_front"
                        ))

        self.zones.extend(zones)
        return zones

    def detect_from_chlorophyll(
        self,
        chl_grid: np.ndarray,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        threshold: float = 3.0
    ) -> List[FishZone]:
        """
        Detecta zonas de peces basadas en concentración de clorofila.

        Alta clorofila = mucho fitoplancton = alimento = peces.
        """
        if chl_grid is None or chl_grid.size == 0:
            return []

        zones = []
        zone_id = len(self.zones)

        rows, cols = chl_grid.shape
        lat_step = (lat_range[1] - lat_range[0]) / rows
        lon_step = (lon_range[1] - lon_range[0]) / cols

        for i in range(rows):
            for j in range(cols):
                if chl_grid[i, j] > threshold:
                    lat = lat_range[0] + i * lat_step
                    lon = lon_range[0] + j * lon_step

                    zone_id += 1
                    zones.append(FishZone(
                        id=zone_id,
                        center_lat=lat,
                        center_lon=lon,
                        radius_m=300,
                        intensity=min(1.0, chl_grid[i, j] / 10.0),
                        movement_direction=0,  # Sin dirección específica
                        cause="chlorophyll_hotspot"
                    ))

        self.zones.extend(zones)
        return zones

    def generate_simulated_zones(
        self,
        coastline: CoastlineCurve,
        num_zones: int = 5,
        min_distance_m: float = 200,
        max_distance_m: float = 800
    ) -> List[FishZone]:
        """
        Genera zonas de peces simuladas basadas en la costa.
        Útil cuando no hay datos reales disponibles.
        """
        if not coastline.curve_points:
            return []

        zones = []
        np.random.seed(int(datetime.now().hour))

        # Seleccionar puntos de costa aleatorios
        indices = np.random.choice(
            len(coastline.curve_points),
            min(num_zones, len(coastline.curve_points)),
            replace=False
        )

        for i, idx in enumerate(indices):
            # Dirección hacia el mar
            sea_dir = coastline.calculate_perpendicular_direction(idx)

            # Distancia aleatoria
            distance = np.random.uniform(min_distance_m, max_distance_m)

            # Calcular posición de la zona
            zone_lat, zone_lon = coastline.get_point_at_distance(idx, distance, sea_dir)

            # Dirección de movimiento (tendencia hacia la costa)
            # Agregar variación aleatoria
            movement_dir = (sea_dir + 180 + np.random.normal(0, 30)) % 360

            zones.append(FishZone(
                id=i + 1,
                center_lat=zone_lat,
                center_lon=zone_lon,
                radius_m=np.random.uniform(150, 400),
                intensity=np.random.uniform(0.4, 0.9),
                movement_direction=movement_dir,
                movement_speed=np.random.uniform(0.3, 0.8),
                cause="simulated_activity"
            ))

        self.zones = zones
        return zones


class FishingOptimizer:
    """
    Optimizador principal que encuentra el mejor punto de pesca.

    Modelo:
    1. Define la línea de orilla como curva
    2. Detecta/genera zonas de actividad de peces
    3. Calcula vectores desde cada punto de orilla a zonas de peces
    4. Considera dirección de movimiento de peces
    5. Recomienda el punto óptimo
    """

    def __init__(self):
        self.coastline = CoastlineCurve()
        self.detector = FishZoneDetector()
        self.shore_points: List[ShorePoint] = []
        self.fish_zones: List[FishZone] = []

    def set_coastline(self, reference_points: List[Tuple[float, float]], num_points: int = 30):
        """
        Define la línea de orilla.

        Args:
            reference_points: Puntos de referencia (lat, lon)
            num_points: Número de puntos a generar en la curva
        """
        self.coastline.set_reference_points(reference_points)
        curve_points = self.coastline.generate_curve(num_points)

        # Crear ShorePoints
        self.shore_points = []
        for i, (lat, lon) in enumerate(curve_points):
            self.shore_points.append(ShorePoint(
                id=i + 1,
                lat=lat,
                lon=lon
            ))

        print(f"[OK] Línea de orilla definida: {len(self.shore_points)} puntos")

    def detect_fish_zones(
        self,
        sst_data: np.ndarray = None,
        chl_data: np.ndarray = None,
        lat_range: Tuple[float, float] = None,
        lon_range: Tuple[float, float] = None,
        use_simulation: bool = True
    ):
        """
        Detecta zonas de actividad de peces.
        """
        self.fish_zones = []

        if sst_data is not None and lat_range and lon_range:
            zones = self.detector.detect_from_sst_data(
                sst_data, lat_range, lon_range
            )
            self.fish_zones.extend(zones)

        if chl_data is not None and lat_range and lon_range:
            zones = self.detector.detect_from_chlorophyll(
                chl_data, lat_range, lon_range
            )
            self.fish_zones.extend(zones)

        # Si no hay datos o pocas zonas, simular
        if use_simulation and len(self.fish_zones) < 3:
            zones = self.detector.generate_simulated_zones(
                self.coastline,
                num_zones=5,
                min_distance_m=150,
                max_distance_m=600
            )
            self.fish_zones.extend(zones)

        print(f"[OK] Zonas de peces detectadas: {len(self.fish_zones)}")

    def _calculate_distance(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float
    ) -> float:
        """Calcula distancia en metros entre dos puntos."""
        dlat = (lat2 - lat1) * 111000
        dlon = (lon2 - lon1) * 111000 * np.cos(np.radians((lat1 + lat2) / 2))
        return np.sqrt(dlat**2 + dlon**2)

    def _calculate_bearing(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float
    ) -> float:
        """Calcula dirección de p1 a p2 en grados."""
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        return np.degrees(np.arctan2(dlon, dlat)) % 360

    def analyze(self) -> List[ShorePoint]:
        """
        Realiza el análisis completo.

        Para cada punto de orilla:
        1. Calcula distancia a cada zona de peces
        2. Considera dirección de movimiento de peces
        3. Calcula score basado en proximidad y movimiento

        Returns:
            Lista de ShorePoints ordenados por score (mejor primero)
        """
        if not self.shore_points:
            raise ValueError("Primero define la línea de orilla con set_coastline()")

        if not self.fish_zones:
            self.detect_fish_zones()

        for shore_point in self.shore_points:
            best_score = 0
            best_distance = float('inf')
            best_direction = 0

            for zone in self.fish_zones:
                # Distancia desde orilla a centro de zona
                distance = self._calculate_distance(
                    shore_point.lat, shore_point.lon,
                    zone.center_lat, zone.center_lon
                )

                # Dirección desde orilla hacia zona
                direction_to_zone = self._calculate_bearing(
                    shore_point.lat, shore_point.lon,
                    zone.center_lat, zone.center_lon
                )

                # Factor de movimiento: ¿los peces se mueven hacia este punto?
                # Si la dirección de movimiento apunta hacia la orilla, mejor
                movement_factor = 1.0
                if zone.movement_direction:
                    # Diferencia angular entre movimiento y dirección inversa
                    ideal_movement = (direction_to_zone + 180) % 360
                    angle_diff = abs(zone.movement_direction - ideal_movement)
                    if angle_diff > 180:
                        angle_diff = 360 - angle_diff
                    # Bonus si peces se mueven hacia la orilla
                    movement_factor = 1.0 + (1.0 - angle_diff / 180) * 0.5

                # Score para esta zona
                # Menor distancia + mayor intensidad + movimiento favorable = mejor
                distance_score = max(0, 100 - distance / 10)  # 0-100 basado en distancia
                intensity_score = zone.intensity * 50
                zone_score = (distance_score * 0.5 + intensity_score * 0.3) * movement_factor

                if zone_score > best_score:
                    best_score = zone_score
                    best_distance = distance
                    best_direction = direction_to_zone

            shore_point.distance_to_fish_zone = best_distance
            shore_point.fish_zone_direction = best_direction
            shore_point.score = min(100, best_score)

        # Ordenar por score
        self.shore_points.sort(key=lambda p: p.score, reverse=True)

        return self.shore_points

    def get_recommendation(self) -> Dict:
        """
        Obtiene recomendación del mejor punto de pesca.
        """
        if not self.shore_points:
            self.analyze()

        best = self.shore_points[0]

        return {
            "best_spot": {
                "id": best.id,
                "latitude": best.lat,
                "longitude": best.lon,
                "score": round(best.score, 1),
                "distance_to_fish_m": round(best.distance_to_fish_zone, 1),
                "direction_to_fish": round(best.fish_zone_direction, 1)
            },
            "top_5_spots": [p.to_dict() for p in self.shore_points[:5]],
            "fish_zones": [z.to_dict() for z in self.fish_zones],
            "total_shore_points": len(self.shore_points),
            "analysis_time": datetime.now().isoformat()
        }

    def to_geojson(self) -> Dict:
        """
        Exporta todo el análisis como GeoJSON para visualización.
        """
        features = []

        # 1. Línea de orilla
        if self.coastline.curve_points:
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[p[1], p[0]] for p in self.coastline.curve_points]
                },
                "properties": {
                    "type": "coastline",
                    "name": "Línea de Orilla"
                }
            })

        # 2. Puntos de pesca (orilla)
        for point in self.shore_points:
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [point.lon, point.lat]
                },
                "properties": {
                    "type": "shore_point",
                    "id": point.id,
                    "score": point.score,
                    "distance_to_fish_m": point.distance_to_fish_zone,
                    "direction_to_fish": point.fish_zone_direction
                }
            })

        # 3. Zonas de peces
        for zone in self.fish_zones:
            # Centro de la zona
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [zone.center_lon, zone.center_lat]
                },
                "properties": {
                    "type": "fish_zone",
                    "id": zone.id,
                    "intensity": zone.intensity,
                    "movement_direction": zone.movement_direction,
                    "cause": zone.cause,
                    "radius_m": zone.radius_m
                }
            })

        # 4. Vectores de mejor punto a zonas
        if self.shore_points:
            best = self.shore_points[0]
            for zone in self.fish_zones:
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [
                            [best.lon, best.lat],
                            [zone.center_lon, zone.center_lat]
                        ]
                    },
                    "properties": {
                        "type": "vector_to_fish",
                        "from_point": best.id,
                        "to_zone": zone.id
                    }
                })

        return {
            "type": "FeatureCollection",
            "features": features
        }
