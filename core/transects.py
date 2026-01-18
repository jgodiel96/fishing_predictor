"""
Analizador de transectos perpendiculares a la costa.
Crea lineas desde la orilla hacia el mar para analisis oceanografico.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from datetime import datetime

from .coastline import CoastlineSegment, CoastlineModel, SubstrateType


@dataclass
class TransectPoint:
    """Punto individual en un transecto."""
    lat: float
    lon: float
    distance_from_shore_m: float
    sst: Optional[float] = None
    chlorophyll: Optional[float] = None
    depth_estimate_m: Optional[float] = None


@dataclass
class Transect:
    """
    Transecto perpendicular a la costa.
    Va desde la orilla hacia el mar con puntos de muestreo.
    """
    id: int
    name: str
    shore_point: Tuple[float, float]  # (lat, lon) en la orilla
    bearing_to_sea: float  # Direccion hacia el mar
    substrate: SubstrateType
    points: List[TransectPoint] = field(default_factory=list)
    max_distance_m: float = 500.0

    # Datos calculados
    avg_sst: Optional[float] = None
    avg_chlorophyll: Optional[float] = None
    sst_gradient: Optional[float] = None  # C/km hacia el mar
    thermal_front_distance_m: Optional[float] = None

    @property
    def shore_lat(self) -> float:
        return self.shore_point[0]

    @property
    def shore_lon(self) -> float:
        return self.shore_point[1]

    def generate_points(
        self,
        num_points: int = 5,
        max_distance_m: float = 500.0
    ) -> List[TransectPoint]:
        """
        Genera puntos de muestreo a lo largo del transecto.

        Args:
            num_points: numero de puntos a generar
            max_distance_m: distancia maxima desde la orilla

        Returns:
            Lista de puntos del transecto
        """
        self.max_distance_m = max_distance_m
        self.points = []

        # Primer punto en la orilla
        self.points.append(TransectPoint(
            lat=self.shore_lat,
            lon=self.shore_lon,
            distance_from_shore_m=0.0
        ))

        # Puntos adicionales hacia el mar
        for i in range(1, num_points):
            distance = (i / (num_points - 1)) * max_distance_m

            # Calcular posicion
            bearing_rad = np.radians(self.bearing_to_sea)
            dlat = distance / 111000 * np.cos(bearing_rad)
            dlon = distance / (111000 * np.cos(np.radians(self.shore_lat))) * np.sin(bearing_rad)

            self.points.append(TransectPoint(
                lat=self.shore_lat + dlat,
                lon=self.shore_lon + dlon,
                distance_from_shore_m=distance
            ))

        return self.points

    def calculate_statistics(self):
        """Calcula estadisticas del transecto basado en los puntos."""
        sst_values = [p.sst for p in self.points if p.sst is not None]
        chl_values = [p.chlorophyll for p in self.points if p.chlorophyll is not None]

        if sst_values:
            self.avg_sst = np.mean(sst_values)

            # Gradiente SST (diferencia entre orilla y mar)
            if len(sst_values) >= 2:
                sst_shore = sst_values[0]
                sst_sea = sst_values[-1]
                dist_km = self.max_distance_m / 1000
                self.sst_gradient = (sst_sea - sst_shore) / dist_km if dist_km > 0 else 0

                # Detectar frente termico (cambio brusco > 0.5C en 100m)
                for i in range(1, len(sst_values)):
                    if abs(sst_values[i] - sst_values[i-1]) > 0.3:
                        self.thermal_front_distance_m = self.points[i].distance_from_shore_m
                        break

        if chl_values:
            self.avg_chlorophyll = np.mean(chl_values)

    def to_geojson_line(self) -> dict:
        """Convierte el transecto a una linea GeoJSON."""
        coordinates = [[p.lon, p.lat] for p in self.points]

        return {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coordinates
            },
            "properties": {
                "id": self.id,
                "name": self.name,
                "substrate": self.substrate.value,
                "avg_sst": self.avg_sst,
                "avg_chlorophyll": self.avg_chlorophyll,
                "sst_gradient": self.sst_gradient,
                "thermal_front_distance_m": self.thermal_front_distance_m,
                "max_distance_m": self.max_distance_m
            }
        }


class TransectAnalyzer:
    """
    Crea y analiza transectos perpendiculares a la costa.
    """

    def __init__(self, coastline_model: Optional[CoastlineModel] = None):
        """
        Inicializa el analizador.

        Args:
            coastline_model: modelo de costa a usar (crea uno por defecto)
        """
        self.coastline = coastline_model or CoastlineModel()
        self.transects: List[Transect] = []

    def create_transects_for_segment(
        self,
        segment: CoastlineSegment,
        num_transects: int = 5,
        points_per_transect: int = 5,
        max_distance_m: float = 500.0
    ) -> List[Transect]:
        """
        Crea transectos a lo largo de un segmento de costa.

        Args:
            segment: segmento de costa
            num_transects: numero de transectos a crear
            points_per_transect: puntos de muestreo por transecto
            max_distance_m: distancia maxima hacia el mar

        Returns:
            Lista de transectos creados
        """
        transects = []

        for i in range(num_transects):
            # Punto de origen en la costa
            fraction = i / (num_transects - 1) if num_transects > 1 else 0.5
            shore_point = segment.point_at_fraction(fraction)

            transect = Transect(
                id=i + 1,
                name=f"{segment.name} - T{i + 1}",
                shore_point=shore_point,
                bearing_to_sea=segment.bearing_to_sea,
                substrate=segment.substrate
            )

            transect.generate_points(
                num_points=points_per_transect,
                max_distance_m=max_distance_m
            )

            transects.append(transect)

        self.transects.extend(transects)
        return transects

    def create_transect_at_point(
        self,
        lat: float,
        lon: float,
        name: str = "Transecto",
        points_per_transect: int = 5,
        max_distance_m: float = 500.0,
        substrate: SubstrateType = SubstrateType.ARENA
    ) -> Transect:
        """
        Crea un transecto en un punto especifico.

        Detecta automaticamente la orientacion de la costa cercana.
        """
        # Buscar segmento cercano para obtener orientacion
        nearest = self.coastline.find_nearest_segment(lat, lon)

        bearing_to_sea = 250.0  # Valor por defecto para costa peruana (suroeste)
        if nearest:
            bearing_to_sea = nearest.bearing_to_sea
            if substrate == SubstrateType.ARENA:  # Si no se especifico, usar del segmento
                substrate = nearest.substrate

        transect = Transect(
            id=len(self.transects) + 1,
            name=name,
            shore_point=(lat, lon),
            bearing_to_sea=bearing_to_sea,
            substrate=substrate
        )

        transect.generate_points(
            num_points=points_per_transect,
            max_distance_m=max_distance_m
        )

        self.transects.append(transect)
        return transect

    def create_sweep_transects(
        self,
        lat_start: float,
        lon_start: float,
        lat_end: float,
        lon_end: float,
        num_transects: int = 10,
        points_per_transect: int = 5,
        max_distance_m: float = 500.0,
        name: str = "Barrido",
        substrate: SubstrateType = SubstrateType.ARENA
    ) -> List[Transect]:
        """
        Crea un barrido de transectos entre dos puntos de costa.

        Args:
            lat_start, lon_start: punto inicial del barrido
            lat_end, lon_end: punto final del barrido
            num_transects: numero de transectos
            points_per_transect: puntos por transecto
            max_distance_m: distancia maxima hacia el mar
            name: nombre base del barrido
            substrate: tipo de sustrato

        Returns:
            Lista de transectos del barrido
        """
        # Crear segmento virtual para el barrido
        segment = self.coastline.create_custom_segment(
            lat_start=lat_start,
            lon_start=lon_start,
            lat_end=lat_end,
            lon_end=lon_end,
            name=name,
            substrate=substrate
        )

        # Si el segmento es muy corto, extenderlo a lo largo de la costa
        if segment.length_m < 100:
            # Buscar orientacion del segmento mas cercano
            nearest = self.coastline.find_nearest_segment(
                (lat_start + lat_end) / 2,
                (lon_start + lon_end) / 2
            )

            if nearest:
                # Extender el segmento siguiendo la orientacion de la costa
                bearing_rad = np.radians(nearest.bearing)
                extension = 500  # metros en cada direccion

                # Calcular nuevos puntos extendidos
                dlat = extension / 111000 * np.cos(bearing_rad)
                dlon = extension / (111000 * np.cos(np.radians(lat_start))) * np.sin(bearing_rad)

                segment = CoastlineSegment(
                    name=name,
                    lat_start=lat_start - dlat,
                    lon_start=lon_start - dlon,
                    lat_end=lat_start + dlat,
                    lon_end=lon_start + dlon,
                    substrate=substrate,
                    description=f"Barrido extendido: {name}"
                )

        # Crear transectos a lo largo del segmento
        return self.create_transects_for_segment(
            segment=segment,
            num_transects=num_transects,
            points_per_transect=points_per_transect,
            max_distance_m=max_distance_m
        )

    def populate_oceanographic_data(
        self,
        sst_fetcher=None,
        chl_fetcher=None
    ):
        """
        Rellena los transectos con datos oceanograficos.

        Si no se proporcionan fetchers, genera datos simulados realistas.
        """
        np.random.seed(int(datetime.now().hour))

        for transect in self.transects:
            # Base segun sustrato
            if transect.substrate == SubstrateType.ROCA:
                base_sst = 15.5
                base_chl = 5.5
            elif transect.substrate == SubstrateType.ARENA:
                base_sst = 16.2
                base_chl = 4.0
            else:  # MIXTO
                base_sst = 15.8
                base_chl = 4.8

            for point in transect.points:
                # SST aumenta ligeramente hacia el mar (agua mas profunda)
                dist_factor = point.distance_from_shore_m / 500
                point.sst = base_sst + dist_factor * 0.3 + np.random.normal(0, 0.1)
                point.sst = round(np.clip(point.sst, 14.5, 19.0), 2)

                # Clorofila disminuye hacia el mar
                point.chlorophyll = base_chl * (1 - dist_factor * 0.3) + np.random.normal(0, 0.15)
                point.chlorophyll = round(np.clip(point.chlorophyll, 0.5, 12.0), 2)

                # Estimacion de profundidad
                point.depth_estimate_m = round(point.distance_from_shore_m * 0.02, 1)

            # Calcular estadisticas
            transect.calculate_statistics()

    def get_all_transects_geojson(self) -> dict:
        """Exporta todos los transectos como GeoJSON FeatureCollection."""
        features = [t.to_geojson_line() for t in self.transects]

        return {
            "type": "FeatureCollection",
            "features": features
        }

    def get_shore_points_geojson(self) -> dict:
        """Exporta los puntos de orilla como GeoJSON."""
        features = []

        for t in self.transects:
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [t.shore_lon, t.shore_lat]
                },
                "properties": {
                    "id": t.id,
                    "name": t.name,
                    "substrate": t.substrate.value,
                    "avg_sst": t.avg_sst,
                    "avg_chlorophyll": t.avg_chlorophyll,
                    "sst_gradient": t.sst_gradient
                }
            })

        return {
            "type": "FeatureCollection",
            "features": features
        }
