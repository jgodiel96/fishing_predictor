"""
Procesador de línea costera real basado en datos GeoJSON.
Coloca puntos EXACTAMENTE sobre la línea de costa.
"""

import json
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class CoastPoint:
    """Punto en la línea de costa."""
    lat: float
    lon: float
    index: int
    bearing_to_sea: float  # Dirección perpendicular hacia el mar
    bearing_along_coast: float  # Dirección a lo largo de la costa


@dataclass
class CoastSegment:
    """Segmento de costa entre dos puntos."""
    start: CoastPoint
    end: CoastPoint
    length_m: float
    name: str = ""


class RealCoastline:
    """
    Modelo de línea costera basado en datos reales (GeoJSON).
    Garantiza que todos los puntos estén EXACTAMENTE en la costa.
    """

    def __init__(self, geojson_path: str = None):
        """
        Inicializa con datos de coastline.

        Args:
            geojson_path: ruta al archivo GeoJSON de coastline
        """
        self.points: List[CoastPoint] = []
        self.segments: List[CoastSegment] = []

        if geojson_path:
            self.load_from_geojson(geojson_path)

    def load_from_geojson(self, path: str):
        """Carga línea costera desde archivo GeoJSON."""
        with open(path, 'r') as f:
            data = json.load(f)

        all_coords = []

        for feature in data.get("features", []):
            geom = feature.get("geometry", {})
            if geom.get("type") == "LineString":
                coords = geom.get("coordinates", [])
                all_coords.extend(coords)
            elif geom.get("type") == "MultiLineString":
                for line in geom.get("coordinates", []):
                    all_coords.extend(line)

        # Eliminar duplicados manteniendo orden
        seen = set()
        unique_coords = []
        for coord in all_coords:
            key = (round(coord[0], 6), round(coord[1], 6))
            if key not in seen:
                seen.add(key)
                unique_coords.append(coord)

        # Ordenar de sur a norte (por latitud)
        unique_coords.sort(key=lambda c: c[1])

        # Crear puntos con bearings calculados
        self._create_points(unique_coords)

        print(f"[OK] Cargados {len(self.points)} puntos de costa")

    def _create_points(self, coords: List[List[float]]):
        """Crea puntos de costa con bearings calculados."""
        self.points = []

        for i, (lon, lat) in enumerate(coords):
            # Calcular bearing a lo largo de la costa
            if i == 0:
                # Primer punto: usar siguiente
                next_lon, next_lat = coords[i + 1] if len(coords) > 1 else (lon, lat)
                bearing_coast = self._calculate_bearing(lat, lon, next_lat, next_lon)
            elif i == len(coords) - 1:
                # Último punto: usar anterior
                prev_lon, prev_lat = coords[i - 1]
                bearing_coast = self._calculate_bearing(prev_lat, prev_lon, lat, lon)
            else:
                # Punto intermedio: promedio entre anterior y siguiente
                prev_lon, prev_lat = coords[i - 1]
                next_lon, next_lat = coords[i + 1]
                b1 = self._calculate_bearing(prev_lat, prev_lon, lat, lon)
                b2 = self._calculate_bearing(lat, lon, next_lat, next_lon)
                bearing_coast = self._average_bearing(b1, b2)

            # Bearing hacia el mar (perpendicular a la costa)
            # En la costa oeste de Sudamérica, el mar está hacia el oeste
            # Perpendicular = coast_bearing + 90 o - 90
            perp1 = (bearing_coast + 90) % 360
            perp2 = (bearing_coast - 90) % 360

            # Elegir la dirección que apunta más hacia el oeste (mar)
            # El mar está aproximadamente entre 200-300 grados (suroeste a noroeste)
            if 180 <= perp1 <= 315 or perp1 <= 45:
                bearing_sea = perp1
            else:
                bearing_sea = perp2

            # Verificación adicional: el mar tiene longitudes más negativas
            # Así que elegimos la dirección con componente oeste (lon decrece)
            test_point = self._point_at_bearing(lat, lon, bearing_sea, 1000)
            if test_point[1] > lon:  # Si la longitud aumenta, es hacia tierra
                bearing_sea = (bearing_sea + 180) % 360

            self.points.append(CoastPoint(
                lat=lat,
                lon=lon,
                index=i,
                bearing_to_sea=bearing_sea,
                bearing_along_coast=bearing_coast
            ))

    def _calculate_bearing(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float
    ) -> float:
        """Calcula bearing entre dos puntos."""
        lat1_r = np.radians(lat1)
        lat2_r = np.radians(lat2)
        dlon = np.radians(lon2 - lon1)

        x = np.sin(dlon) * np.cos(lat2_r)
        y = np.cos(lat1_r) * np.sin(lat2_r) - np.sin(lat1_r) * np.cos(lat2_r) * np.cos(dlon)

        bearing = np.degrees(np.arctan2(x, y))
        return bearing % 360

    def _average_bearing(self, b1: float, b2: float) -> float:
        """Promedio circular de dos bearings."""
        x = np.cos(np.radians(b1)) + np.cos(np.radians(b2))
        y = np.sin(np.radians(b1)) + np.sin(np.radians(b2))
        return np.degrees(np.arctan2(y, x)) % 360

    def _point_at_bearing(
        self,
        lat: float, lon: float,
        bearing: float,
        distance_m: float
    ) -> Tuple[float, float]:
        """Calcula punto a cierta distancia y bearing."""
        bearing_rad = np.radians(bearing)
        dlat = distance_m / 111000 * np.cos(bearing_rad)
        dlon = distance_m / (111000 * np.cos(np.radians(lat))) * np.sin(bearing_rad)
        return (lat + dlat, lon + dlon)

    def _distance_m(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float
    ) -> float:
        """Distancia aproximada en metros entre dos puntos."""
        dlat = (lat2 - lat1) * 111000
        dlon = (lon2 - lon1) * 111000 * np.cos(np.radians((lat1 + lat2) / 2))
        return np.sqrt(dlat**2 + dlon**2)

    def get_point_on_coast(
        self,
        lat: float,
        lon: float
    ) -> CoastPoint:
        """
        Encuentra el punto más cercano EN la línea de costa.

        Args:
            lat, lon: coordenadas de referencia

        Returns:
            Punto en la costa más cercano
        """
        if not self.points:
            raise ValueError("No hay puntos de costa cargados")

        min_dist = float('inf')
        nearest = None

        for point in self.points:
            dist = self._distance_m(lat, lon, point.lat, point.lon)
            if dist < min_dist:
                min_dist = dist
                nearest = point

        return nearest

    def get_points_in_range(
        self,
        lat_min: float, lat_max: float,
        lon_min: float = None, lon_max: float = None
    ) -> List[CoastPoint]:
        """
        Obtiene puntos de costa en un rango de coordenadas.
        """
        result = []
        for point in self.points:
            if lat_min <= point.lat <= lat_max:
                if lon_min is None or (lon_min <= point.lon <= lon_max):
                    result.append(point)
        return result

    def create_transect_from_point(
        self,
        coast_point: CoastPoint,
        distance_m: float = 500,
        num_points: int = 5
    ) -> List[Tuple[float, float, float]]:
        """
        Crea un transecto perpendicular desde un punto de costa hacia el mar.

        Args:
            coast_point: punto de origen en la costa
            distance_m: distancia total del transecto
            num_points: número de puntos en el transecto

        Returns:
            Lista de (lat, lon, distancia_desde_costa)
        """
        transect = [(coast_point.lat, coast_point.lon, 0.0)]

        for i in range(1, num_points):
            dist = (i / (num_points - 1)) * distance_m
            new_point = self._point_at_bearing(
                coast_point.lat,
                coast_point.lon,
                coast_point.bearing_to_sea,
                dist
            )
            transect.append((new_point[0], new_point[1], dist))

        return transect

    def sample_coast(
        self,
        num_points: int = 20,
        lat_range: Tuple[float, float] = None
    ) -> List[CoastPoint]:
        """
        Muestrea puntos equidistantes a lo largo de la costa.

        Args:
            num_points: número de puntos a muestrear
            lat_range: (lat_min, lat_max) opcional

        Returns:
            Lista de puntos muestreados
        """
        if lat_range:
            points = self.get_points_in_range(lat_range[0], lat_range[1])
        else:
            points = self.points

        if len(points) <= num_points:
            return points

        # Muestrear equidistantemente
        indices = np.linspace(0, len(points) - 1, num_points, dtype=int)
        return [points[i] for i in indices]

    def to_geojson(self) -> Dict:
        """Exporta la costa como GeoJSON."""
        coords = [[p.lon, p.lat] for p in self.points]

        return {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords
                },
                "properties": {
                    "name": "Línea de Costa",
                    "num_points": len(self.points)
                }
            }]
        }

    def get_fishing_spots(
        self,
        num_spots: int = 15,
        spacing_km: float = 5.0
    ) -> List[Dict]:
        """
        Genera spots de pesca a lo largo de la costa.

        Args:
            num_spots: número de spots
            spacing_km: espaciado mínimo entre spots en km

        Returns:
            Lista de diccionarios con información de cada spot
        """
        if not self.points:
            return []

        spots = []
        sampled = self.sample_coast(num_spots)

        for i, point in enumerate(sampled):
            spots.append({
                "id": i + 1,
                "name": f"Spot {i + 1}",
                "latitude": point.lat,
                "longitude": point.lon,
                "bearing_to_sea": point.bearing_to_sea,
                "bearing_along_coast": point.bearing_along_coast,
            })

        return spots


def load_coastline(data_manager=None) -> RealCoastline:
    """
    Carga la línea costera real de OSM.

    Returns:
        Instancia de RealCoastline
    """
    # Preferir el archivo OSM real si existe
    osm_path = Path(__file__).parent.parent / "data" / "cache" / "coastline_real_osm.geojson"

    if osm_path.exists():
        return RealCoastline(str(osm_path))

    # Fallback: usar DataManager para descargar
    if data_manager is None:
        from data.data_manager import DataManager
        data_manager = DataManager()

    coastline_path = data_manager.download_coastline()
    return RealCoastline(str(coastline_path))
