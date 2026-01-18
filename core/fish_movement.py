"""
Predictor de movimiento de peces basado en gradientes termicos y clorofila.
Genera vectores de direccion para visualizacion con flechas.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from enum import Enum

from .transects import Transect, TransectAnalyzer
from .coastline import SubstrateType


class MovementTrend(Enum):
    """Tendencia de movimiento de peces."""
    HACIA_COSTA = "hacia_costa"  # Peces moviendose hacia la orilla
    HACIA_MAR = "hacia_mar"      # Peces moviendose hacia aguas profundas
    PARALELO_COSTA = "paralelo_costa"  # Moviendose a lo largo de la costa
    ESTACIONARIO = "estacionario"  # Sin movimiento claro


@dataclass
class MovementVector:
    """
    Vector de movimiento predicho para peces en una ubicacion.
    """
    lat: float
    lon: float
    direction_deg: float  # Direccion del movimiento (0-360)
    magnitude: float  # Intensidad del movimiento (0-1)
    trend: MovementTrend
    confidence: float  # Confianza de la prediccion (0-1)

    # Factores que influyen
    thermal_factor: float = 0.0  # Influencia del gradiente termico
    food_factor: float = 0.0  # Influencia de la clorofila
    substrate_factor: float = 0.0  # Influencia del sustrato

    # Especies recomendadas
    target_species: List[str] = None

    def __post_init__(self):
        if self.target_species is None:
            self.target_species = []

    @property
    def direction_rad(self) -> float:
        """Direccion en radianes."""
        return np.radians(self.direction_deg)

    @property
    def dx(self) -> float:
        """Componente X del vector (para visualizacion)."""
        return self.magnitude * np.sin(self.direction_rad)

    @property
    def dy(self) -> float:
        """Componente Y del vector (para visualizacion)."""
        return self.magnitude * np.cos(self.direction_rad)

    def get_arrow_endpoint(self, scale_m: float = 100) -> Tuple[float, float]:
        """
        Calcula el punto final de la flecha para visualizacion.

        Args:
            scale_m: escala en metros para la longitud de la flecha

        Returns:
            (lat, lon) del punto final
        """
        # Longitud proporcional a la magnitud
        length = scale_m * self.magnitude

        # Calcular desplazamiento
        dlat = length / 111000 * np.cos(self.direction_rad)
        dlon = length / (111000 * np.cos(np.radians(self.lat))) * np.sin(self.direction_rad)

        return (self.lat + dlat, self.lon + dlon)

    def to_geojson(self, scale_m: float = 100) -> dict:
        """Convierte a GeoJSON LineString (flecha)."""
        end_lat, end_lon = self.get_arrow_endpoint(scale_m)

        return {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [self.lon, self.lat],
                    [end_lon, end_lat]
                ]
            },
            "properties": {
                "direction_deg": round(self.direction_deg, 1),
                "magnitude": round(self.magnitude, 2),
                "trend": self.trend.value,
                "confidence": round(self.confidence, 2),
                "thermal_factor": round(self.thermal_factor, 2),
                "food_factor": round(self.food_factor, 2),
                "target_species": self.target_species
            }
        }


class FishMovementPredictor:
    """
    Predice la direccion de movimiento de peces basandose en:
    - Gradientes termicos (SST)
    - Concentracion de clorofila (alimento)
    - Tipo de sustrato
    - Hora del dia
    """

    # Temperaturas optimas por tipo de pez (costa peruana)
    OPTIMAL_TEMPS = {
        "corvina": (15.5, 17.5),
        "robalo": (16.0, 18.0),
        "cabrilla": (15.0, 17.0),
        "lenguado": (15.5, 17.5),
        "pejerrey": (14.5, 16.5),
        "pintadilla": (15.0, 17.0),
    }

    # Preferencias de sustrato
    SUBSTRATE_PREFS = {
        "corvina": [SubstrateType.ARENA, SubstrateType.MIXTO],
        "robalo": [SubstrateType.ROCA, SubstrateType.MIXTO],
        "cabrilla": [SubstrateType.ROCA],
        "lenguado": [SubstrateType.ARENA],
        "pejerrey": [SubstrateType.ARENA, SubstrateType.MIXTO],
        "pintadilla": [SubstrateType.ROCA, SubstrateType.MIXTO],
    }

    def __init__(self):
        self.vectors: List[MovementVector] = []

    def predict_movement(
        self,
        transects: List[Transect],
        hour: int = 12,
        include_species: bool = True
    ) -> List[MovementVector]:
        """
        Predice movimiento de peces basado en transectos analizados.

        Args:
            transects: lista de transectos con datos oceanograficos
            hour: hora del dia (0-23)
            include_species: incluir recomendaciones de especies

        Returns:
            Lista de vectores de movimiento
        """
        self.vectors = []

        for transect in transects:
            vector = self._analyze_transect(transect, hour)

            if include_species:
                vector.target_species = self._recommend_species(transect, hour)

            self.vectors.append(vector)

        return self.vectors

    def _analyze_transect(
        self,
        transect: Transect,
        hour: int
    ) -> MovementVector:
        """Analiza un transecto y predice movimiento."""

        # Punto de referencia (orilla)
        lat, lon = transect.shore_point

        # Analizar gradiente termico
        thermal_factor = 0.0
        thermal_direction = 0.0

        if transect.sst_gradient is not None:
            # Gradiente positivo = agua mas calida hacia el mar
            # Peces siguen agua mas calida en invierno, mas fria en verano
            thermal_factor = min(abs(transect.sst_gradient) / 2.0, 1.0)

            if transect.sst_gradient > 0.1:
                # Agua mas calida hacia el mar - peces van hacia el mar
                thermal_direction = transect.bearing_to_sea
            elif transect.sst_gradient < -0.1:
                # Agua mas fria hacia el mar - peces van hacia la costa
                thermal_direction = (transect.bearing_to_sea + 180) % 360
            else:
                # Sin gradiente claro - movimiento paralelo
                thermal_direction = (transect.bearing_to_sea + 90) % 360

        # Analizar clorofila (alimento)
        food_factor = 0.0
        food_direction = 0.0

        if transect.avg_chlorophyll is not None:
            # Alta clorofila cerca de la costa atrae peces
            shore_chl = transect.points[0].chlorophyll if transect.points else 0
            sea_chl = transect.points[-1].chlorophyll if transect.points else 0

            if shore_chl and sea_chl:
                if shore_chl > sea_chl * 1.2:
                    # Mas alimento en la costa
                    food_factor = min((shore_chl - sea_chl) / 3.0, 1.0)
                    food_direction = (transect.bearing_to_sea + 180) % 360
                elif sea_chl > shore_chl * 1.2:
                    # Mas alimento mar adentro
                    food_factor = min((sea_chl - shore_chl) / 3.0, 1.0)
                    food_direction = transect.bearing_to_sea

        # Factor sustrato
        substrate_factor = 0.3 if transect.substrate == SubstrateType.ROCA else 0.1

        # Factor hora (peces mas activos en horas doradas)
        if 5 <= hour <= 8 or 17 <= hour <= 20:
            activity_mult = 1.2  # Horas doradas
        elif 11 <= hour <= 14:
            activity_mult = 0.7  # Mediodia - menos activos
        else:
            activity_mult = 1.0

        # Combinar factores para direccion final
        # Ponderacion: termica 50%, alimento 35%, aleatorio 15%
        if thermal_factor > food_factor:
            primary_direction = thermal_direction
            primary_weight = thermal_factor
        else:
            primary_direction = food_direction
            primary_weight = food_factor

        # Agregar componente aleatorio para variabilidad natural
        random_component = np.random.normal(0, 15)  # +/- 15 grados
        final_direction = (primary_direction + random_component) % 360

        # Magnitud del movimiento
        magnitude = (
            thermal_factor * 0.5 +
            food_factor * 0.35 +
            substrate_factor * 0.15
        ) * activity_mult

        magnitude = min(1.0, max(0.1, magnitude))

        # Determinar tendencia
        bearing_to_sea = transect.bearing_to_sea
        diff_to_sea = abs(final_direction - bearing_to_sea)
        diff_to_coast = abs(final_direction - (bearing_to_sea + 180) % 360)

        if diff_to_sea < 45 or diff_to_sea > 315:
            trend = MovementTrend.HACIA_MAR
        elif diff_to_coast < 45 or diff_to_coast > 315:
            trend = MovementTrend.HACIA_COSTA
        elif magnitude < 0.2:
            trend = MovementTrend.ESTACIONARIO
        else:
            trend = MovementTrend.PARALELO_COSTA

        # Confianza basada en claridad de los gradientes
        confidence = min(1.0, (thermal_factor + food_factor) / 1.5 + 0.3)

        return MovementVector(
            lat=lat,
            lon=lon,
            direction_deg=final_direction,
            magnitude=magnitude,
            trend=trend,
            confidence=confidence,
            thermal_factor=thermal_factor,
            food_factor=food_factor,
            substrate_factor=substrate_factor
        )

    def _recommend_species(
        self,
        transect: Transect,
        hour: int
    ) -> List[str]:
        """Recomienda especies probables basado en condiciones."""
        species = []
        avg_sst = transect.avg_sst or 16.0

        for fish, (temp_min, temp_max) in self.OPTIMAL_TEMPS.items():
            # Verificar temperatura
            if temp_min <= avg_sst <= temp_max:
                # Verificar sustrato
                if transect.substrate in self.SUBSTRATE_PREFS.get(fish, []):
                    species.append(fish)
                elif transect.substrate == SubstrateType.MIXTO:
                    # Mixto es bueno para casi todos
                    species.append(fish)

        # Si no hay coincidencias, dar opciones generales
        if not species:
            if transect.substrate == SubstrateType.ROCA:
                species = ["cabrilla", "pintadilla"]
            elif transect.substrate == SubstrateType.ARENA:
                species = ["corvina", "lenguado"]
            else:
                species = ["corvina", "robalo"]

        return species[:3]  # Maximo 3 especies

    def get_dominant_trend(self) -> MovementTrend:
        """Obtiene la tendencia dominante de todos los vectores."""
        if not self.vectors:
            return MovementTrend.ESTACIONARIO

        trends = [v.trend for v in self.vectors]
        from collections import Counter
        most_common = Counter(trends).most_common(1)

        return most_common[0][0] if most_common else MovementTrend.ESTACIONARIO

    def get_average_direction(self) -> float:
        """Calcula la direccion promedio de movimiento."""
        if not self.vectors:
            return 0.0

        # Promedio circular
        sin_sum = sum(np.sin(np.radians(v.direction_deg)) for v in self.vectors)
        cos_sum = sum(np.cos(np.radians(v.direction_deg)) for v in self.vectors)

        return np.degrees(np.arctan2(sin_sum, cos_sum)) % 360

    def get_movement_geojson(self, scale_m: float = 150) -> dict:
        """Exporta todos los vectores como GeoJSON."""
        features = [v.to_geojson(scale_m) for v in self.vectors]

        return {
            "type": "FeatureCollection",
            "features": features
        }

    def get_summary(self) -> Dict:
        """Resumen del analisis de movimiento."""
        if not self.vectors:
            return {"error": "No hay vectores calculados"}

        return {
            "num_vectors": len(self.vectors),
            "dominant_trend": self.get_dominant_trend().value,
            "average_direction": round(self.get_average_direction(), 1),
            "average_magnitude": round(np.mean([v.magnitude for v in self.vectors]), 2),
            "average_confidence": round(np.mean([v.confidence for v in self.vectors]), 2),
            "trends_breakdown": {
                trend.value: sum(1 for v in self.vectors if v.trend == trend)
                for trend in MovementTrend
            }
        }
