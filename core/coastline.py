"""
Modelo de linea costera para la costa sur de Peru.
Define la geometria real de la costa para posicionamiento correcto de puntos.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from enum import Enum


class SubstrateType(Enum):
    """Tipos de sustrato costero."""
    ROCA = "roca"
    ARENA = "arena"
    MIXTO = "mixto"


@dataclass
class CoastlineSegment:
    """
    Segmento de linea costera con orientacion definida.

    La orientacion (bearing) indica la direccion de la costa:
    - 0/360 = Norte
    - 90 = Este
    - 180 = Sur
    - 270 = Oeste

    El bearing_perpendicular apunta hacia el MAR (para transectos).
    """
    name: str
    lat_start: float
    lon_start: float
    lat_end: float
    lon_end: float
    substrate: SubstrateType
    description: str

    # Calculados automaticamente
    bearing: float = 0.0  # Direccion de la costa
    bearing_to_sea: float = 0.0  # Direccion perpendicular hacia el mar
    length_m: float = 0.0

    def __post_init__(self):
        """Calcular bearing y longitud automaticamente."""
        self._calculate_geometry()

    def _calculate_geometry(self):
        """Calcula la geometria del segmento."""
        dlat = self.lat_end - self.lat_start
        dlon = self.lon_end - self.lon_start

        # Bearing de la costa (direccion en la que corre)
        lat_avg = np.radians((self.lat_start + self.lat_end) / 2)
        dlon_adj = dlon * np.cos(lat_avg)

        self.bearing = np.degrees(np.arctan2(dlon_adj, dlat)) % 360

        # Perpendicular hacia el mar (en Peru, el mar esta al OESTE/SUROESTE)
        # La costa sur de Peru corre aproximadamente NW-SE
        # El mar esta hacia el oeste, asi que perpendicular + 90 grados (o - 90)
        # Determinamos cual apunta al mar basado en la longitud
        perp1 = (self.bearing + 90) % 360
        perp2 = (self.bearing - 90) % 360

        # En la costa peruana, el mar tiene longitudes MAS NEGATIVAS (mas al oeste)
        # Elegimos la direccion que va hacia el oeste
        if 180 < perp1 < 360 or perp1 < 90:
            self.bearing_to_sea = perp1 if 180 < perp1 < 360 else perp2
        else:
            self.bearing_to_sea = perp2 if 180 < perp2 < 360 else perp1

        # Para la costa sur de Peru, el mar esta generalmente hacia el suroeste (200-270 grados)
        # Verificamos y corregimos si es necesario
        if not (180 <= self.bearing_to_sea <= 315):
            self.bearing_to_sea = (self.bearing_to_sea + 180) % 360

        # Longitud en metros
        dist_lat_m = dlat * 111000
        dist_lon_m = dlon * 111000 * np.cos(lat_avg)
        self.length_m = np.sqrt(dist_lat_m**2 + dist_lon_m**2)

    @property
    def center(self) -> Tuple[float, float]:
        """Punto central del segmento."""
        return (
            (self.lat_start + self.lat_end) / 2,
            (self.lon_start + self.lon_end) / 2
        )

    def point_at_fraction(self, fraction: float) -> Tuple[float, float]:
        """Obtiene un punto a cierta fraccion (0-1) del segmento."""
        fraction = max(0, min(1, fraction))
        lat = self.lat_start + (self.lat_end - self.lat_start) * fraction
        lon = self.lon_start + (self.lon_end - self.lon_start) * fraction
        return (lat, lon)

    def point_towards_sea(
        self,
        from_point: Tuple[float, float],
        distance_m: float
    ) -> Tuple[float, float]:
        """
        Calcula un punto a cierta distancia hacia el mar desde un punto de la costa.

        Args:
            from_point: (lat, lon) punto de origen en la costa
            distance_m: distancia en metros hacia el mar

        Returns:
            (lat, lon) del punto en el mar
        """
        lat, lon = from_point

        # Convertir bearing a radianes
        bearing_rad = np.radians(self.bearing_to_sea)

        # Calcular desplazamiento
        dlat = distance_m / 111000 * np.cos(bearing_rad)
        dlon = distance_m / (111000 * np.cos(np.radians(lat))) * np.sin(bearing_rad)

        return (lat + dlat, lon + dlon)


class CoastlineModel:
    """
    Modelo completo de la linea costera de Tacna-Ilo.
    Define segmentos con orientacion correcta para analisis de transectos.
    """

    def __init__(self):
        self.segments: List[CoastlineSegment] = []
        self._build_coastline()

    def _build_coastline(self):
        """
        Construye el modelo de la costa sur de Peru (Tacna a Ilo).
        Los segmentos siguen la linea de costa real.
        """
        # Costa de Tacna a Ilo - ordenada de sur a norte
        # La costa peruana en esta zona corre aproximadamente de SE a NW

        self.segments = [
            # Zona Sur - Tacna
            CoastlineSegment(
                name="Boca del Rio",
                lat_start=-18.1250, lon_start=-70.8380,
                lat_end=-18.1160, lon_end=-70.8480,
                substrate=SubstrateType.ARENA,
                description="Desembocadura, playa arenosa extensa"
            ),
            CoastlineSegment(
                name="Playa Santa Rosa",
                lat_start=-18.0920, lon_start=-70.8620,
                lat_end=-18.0820, lon_end=-70.8740,
                substrate=SubstrateType.ARENA,
                description="Playa extensa, buena para corvina"
            ),
            CoastlineSegment(
                name="Los Palos",
                lat_start=-18.0570, lon_start=-70.8780,
                lat_end=-18.0470, lon_end=-70.8880,
                substrate=SubstrateType.MIXTO,
                description="Arena con rocas dispersas"
            ),
            CoastlineSegment(
                name="Vila Vila",
                lat_start=-18.0230, lon_start=-70.9070,
                lat_end=-18.0130, lon_end=-70.9170,
                substrate=SubstrateType.ROCA,
                description="Zona rocosa con buena estructura"
            ),
            CoastlineSegment(
                name="Punta Mesa",
                lat_start=-17.9930, lon_start=-70.9300,
                lat_end=-17.9830, lon_end=-70.9400,
                substrate=SubstrateType.ROCA,
                description="Punta rocosa con pozas"
            ),
            CoastlineSegment(
                name="Carlepe",
                lat_start=-17.9670, lon_start=-70.9430,
                lat_end=-17.9570, lon_end=-70.9530,
                substrate=SubstrateType.MIXTO,
                description="Rocas y arena alternadas"
            ),

            # Zona Ite
            CoastlineSegment(
                name="Playa Ite Norte",
                lat_start=-17.9370, lon_start=-70.9630,
                lat_end=-17.9270, lon_end=-70.9730,
                substrate=SubstrateType.ARENA,
                description="Playa amplia con surgencia activa"
            ),
            CoastlineSegment(
                name="Ite Centro",
                lat_start=-17.9070, lon_start=-70.9870,
                lat_end=-17.8970, lon_end=-70.9970,
                substrate=SubstrateType.MIXTO,
                description="Zona mixta muy productiva"
            ),
            CoastlineSegment(
                name="Ite Sur",
                lat_start=-17.8770, lon_start=-71.0130,
                lat_end=-17.8670, lon_end=-71.0230,
                substrate=SubstrateType.ROCA,
                description="Formaciones rocosas, cabrilla"
            ),

            # Zona Intermedia
            CoastlineSegment(
                name="Gentillar",
                lat_start=-17.8470, lon_start=-71.0430,
                lat_end=-17.8370, lon_end=-71.0530,
                substrate=SubstrateType.ROCA,
                description="Costa rocosa escarpada"
            ),
            CoastlineSegment(
                name="Punta Blanca",
                lat_start=-17.8170, lon_start=-71.0770,
                lat_end=-17.8070, lon_end=-71.0870,
                substrate=SubstrateType.ROCA,
                description="Punta rocosa, buena para robalo"
            ),
            CoastlineSegment(
                name="Pozo Redondo",
                lat_start=-17.7870, lon_start=-71.1170,
                lat_end=-17.7770, lon_end=-71.1270,
                substrate=SubstrateType.MIXTO,
                description="Pozas naturales entre rocas"
            ),
            CoastlineSegment(
                name="Fundicion",
                lat_start=-17.7620, lon_start=-71.1670,
                lat_end=-17.7520, lon_end=-71.1770,
                substrate=SubstrateType.ROCA,
                description="Rocas grandes, estructura compleja"
            ),
            CoastlineSegment(
                name="Playa Media Luna",
                lat_start=-17.7370, lon_start=-71.2170,
                lat_end=-17.7270, lon_end=-71.2270,
                substrate=SubstrateType.ARENA,
                description="Bahia arenosa en forma de media luna"
            ),

            # Zona Ilo
            CoastlineSegment(
                name="Punta Coles",
                lat_start=-17.7070, lon_start=-71.3270,
                lat_end=-17.6970, lon_end=-71.3370,
                substrate=SubstrateType.ROCA,
                description="Reserva con rocas y mucha vida marina"
            ),
            CoastlineSegment(
                name="Pocoma",
                lat_start=-17.6870, lon_start=-71.2900,
                lat_end=-17.6770, lon_end=-71.3000,
                substrate=SubstrateType.ROCA,
                description="Acantilados rocosos"
            ),
            CoastlineSegment(
                name="Ilo - Pozo Lizas",
                lat_start=-17.6470, lon_start=-71.3350,
                lat_end=-17.6370, lon_end=-71.3450,
                substrate=SubstrateType.ROCA,
                description="Pozas entre rocas, ideal spinning"
            ),
            CoastlineSegment(
                name="Ilo - Puerto",
                lat_start=-17.6370, lon_start=-71.3400,
                lat_end=-17.6270, lon_end=-71.3500,
                substrate=SubstrateType.MIXTO,
                description="Muelle y rocas adyacentes"
            ),
        ]

    def find_nearest_segment(
        self,
        lat: float,
        lon: float
    ) -> Optional[CoastlineSegment]:
        """
        Encuentra el segmento de costa mas cercano a un punto.

        Args:
            lat: latitud del punto
            lon: longitud del punto

        Returns:
            Segmento mas cercano o None si no hay segmentos
        """
        if not self.segments:
            return None

        min_dist = float('inf')
        nearest = None

        for segment in self.segments:
            center_lat, center_lon = segment.center

            # Distancia aproximada en metros
            dlat = (lat - center_lat) * 111000
            dlon = (lon - center_lon) * 111000 * np.cos(np.radians(lat))
            dist = np.sqrt(dlat**2 + dlon**2)

            if dist < min_dist:
                min_dist = dist
                nearest = segment

        return nearest

    def get_segment_by_name(self, name: str) -> Optional[CoastlineSegment]:
        """Obtiene un segmento por su nombre."""
        for segment in self.segments:
            if segment.name.lower() == name.lower():
                return segment
        return None

    def create_custom_segment(
        self,
        lat_start: float,
        lon_start: float,
        lat_end: float,
        lon_end: float,
        name: str = "Segmento Personalizado",
        substrate: SubstrateType = SubstrateType.ARENA
    ) -> CoastlineSegment:
        """
        Crea un segmento personalizado para un tramo especifico.

        Detecta automaticamente la orientacion basandose en segmentos cercanos
        o en la geometria general de la costa.
        """
        # Crear segmento inicial
        segment = CoastlineSegment(
            name=name,
            lat_start=lat_start,
            lon_start=lon_start,
            lat_end=lat_end,
            lon_end=lon_end,
            substrate=substrate,
            description=f"Segmento personalizado: {name}"
        )

        # Si el segmento es muy pequeno, buscar orientacion de segmento cercano
        if segment.length_m < 50:
            nearest = self.find_nearest_segment(
                (lat_start + lat_end) / 2,
                (lon_start + lon_end) / 2
            )
            if nearest:
                # Usar la orientacion del segmento cercano
                segment.bearing = nearest.bearing
                segment.bearing_to_sea = nearest.bearing_to_sea

        return segment

    def get_all_fishing_points(self) -> List[Dict]:
        """
        Obtiene todos los puntos de pesca como lista de diccionarios.
        Cada punto esta en el centro del segmento de costa.
        """
        points = []
        for segment in self.segments:
            lat, lon = segment.center
            points.append({
                "name": segment.name,
                "latitude": lat,
                "longitude": lon,
                "substrate": segment.substrate.value,
                "description": segment.description,
                "bearing_coast": segment.bearing,
                "bearing_to_sea": segment.bearing_to_sea,
                "segment_length_m": segment.length_m
            })
        return points
