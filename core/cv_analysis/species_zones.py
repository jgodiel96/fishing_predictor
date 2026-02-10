"""
Species Zone Generator for Fishing Predictor V8.

Generates colored fishing zones based on:
- Substrate type (rock/sand/mixed)
- Depth zones (shallow/moderate/deep)
- Species habitat preferences

Output: Polygons with species recommendations and colors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Tuple, Dict, Set
import colorsys

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from .substrate_classifier import SubstrateType

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURACION DE ESPECIES
# =============================================================================

class DepthZone(Enum):
    """Zonas de profundidad."""
    INTERTIDAL = "intertidal"   # 0-2m
    SHALLOW = "shallow"         # 2-10m
    MODERATE = "moderate"       # 10-30m
    DEEP = "deep"               # 30-100m
    VERY_DEEP = "very_deep"     # >100m


@dataclass
class SpeciesHabitat:
    """Preferencias de habitat de una especie."""
    name: str
    name_es: str
    preferred_substrates: Set[SubstrateType]
    depth_range: Tuple[float, float]  # (min, max) en metros
    optimal_depth: float
    color: Tuple[int, int, int]  # RGB
    icon: str = ""

    def get_affinity(self, substrate: SubstrateType, depth: float) -> float:
        """
        Calcula afinidad de la especie para habitat dado.

        Returns:
            Score 0-1
        """
        # Score por sustrato
        if substrate in self.preferred_substrates:
            substrate_score = 1.0
        elif substrate == SubstrateType.MIXED:
            substrate_score = 0.6
        else:
            substrate_score = 0.2

        # Score por profundidad
        abs_depth = abs(depth)
        min_d, max_d = self.depth_range

        if min_d <= abs_depth <= max_d:
            # Dentro del rango - calcular proximidad a optimo
            distance = abs(abs_depth - self.optimal_depth)
            range_size = max_d - min_d
            depth_score = 1.0 - (distance / range_size) * 0.5
        else:
            # Fuera del rango
            if abs_depth < min_d:
                depth_score = max(0, 1.0 - (min_d - abs_depth) / 10)
            else:
                depth_score = max(0, 1.0 - (abs_depth - max_d) / 20)

        return substrate_score * 0.5 + depth_score * 0.5


# Base de datos de especies
SPECIES_DATABASE: Dict[str, SpeciesHabitat] = {
    'corvina': SpeciesHabitat(
        name='Corvina',
        name_es='Corvina',
        preferred_substrates={SubstrateType.ROCK, SubstrateType.MIXED},
        depth_range=(3, 40),
        optimal_depth=15,
        color=(255, 165, 0),  # Naranja
        icon='🐟'
    ),
    'lenguado': SpeciesHabitat(
        name='Flounder',
        name_es='Lenguado',
        preferred_substrates={SubstrateType.SAND},
        depth_range=(2, 50),
        optimal_depth=20,
        color=(255, 215, 0),  # Dorado
        icon='🐠'
    ),
    'cabrilla': SpeciesHabitat(
        name='Rock Bass',
        name_es='Cabrilla',
        preferred_substrates={SubstrateType.ROCK},
        depth_range=(2, 30),
        optimal_depth=10,
        color=(255, 69, 0),  # Rojo-naranja
        icon='🐟'
    ),
    'chita': SpeciesHabitat(
        name='Chita',
        name_es='Chita',
        preferred_substrates={SubstrateType.ROCK, SubstrateType.MIXED},
        depth_range=(1, 20),
        optimal_depth=8,
        color=(220, 20, 60),  # Carmesi
        icon='🐟'
    ),
    'pejerrey': SpeciesHabitat(
        name='Silverside',
        name_es='Pejerrey',
        preferred_substrates={SubstrateType.SAND, SubstrateType.MIXED},
        depth_range=(0, 15),
        optimal_depth=5,
        color=(70, 130, 180),  # Azul acero
        icon='🐟'
    ),
    'lorna': SpeciesHabitat(
        name='Lorna Drum',
        name_es='Lorna',
        preferred_substrates={SubstrateType.SAND, SubstrateType.MIXED},
        depth_range=(3, 35),
        optimal_depth=15,
        color=(147, 112, 219),  # Purpura medio
        icon='🐟'
    ),
    'pintadilla': SpeciesHabitat(
        name='Painted Comber',
        name_es='Pintadilla',
        preferred_substrates={SubstrateType.ROCK},
        depth_range=(2, 25),
        optimal_depth=12,
        color=(255, 127, 80),  # Coral
        icon='🐟'
    ),
    'tramboyo': SpeciesHabitat(
        name='Blenny',
        name_es='Tramboyo',
        preferred_substrates={SubstrateType.ROCK},
        depth_range=(0, 15),
        optimal_depth=5,
        color=(100, 149, 237),  # Azul aciano
        icon='🐟'
    ),
    'tollo': SpeciesHabitat(
        name='Smoothhound Shark',
        name_es='Tollo',
        preferred_substrates={SubstrateType.SAND, SubstrateType.MIXED},
        depth_range=(10, 100),
        optimal_depth=40,
        color=(105, 105, 105),  # Gris oscuro
        icon='🦈'
    ),
}


# =============================================================================
# ZONA DE ESPECIES
# =============================================================================

@dataclass
class SpeciesZone:
    """Representa una zona de pesca para especies especificas."""
    zone_id: str
    polygon: List[Tuple[float, float]]  # [(lat, lon), ...]
    center: Tuple[float, float]
    substrate: SubstrateType
    depth_zone: DepthZone
    avg_depth: float
    species_scores: Dict[str, float]  # species -> affinity
    primary_species: str
    secondary_species: List[str]
    color: Tuple[int, int, int]
    area_km2: float = 0.0

    def to_geojson_feature(self) -> Dict:
        """Convierte a formato GeoJSON Feature."""
        # Crear coordenadas en formato GeoJSON [lon, lat]
        coords = [[lon, lat] for lat, lon in self.polygon]
        # Cerrar poligono
        if coords and coords[0] != coords[-1]:
            coords.append(coords[0])

        return {
            'type': 'Feature',
            'properties': {
                'zone_id': self.zone_id,
                'substrate': self.substrate.value,
                'depth_zone': self.depth_zone.value,
                'avg_depth': self.avg_depth,
                'primary_species': self.primary_species,
                'secondary_species': self.secondary_species,
                'species_scores': self.species_scores,
                'color': f'rgb({self.color[0]},{self.color[1]},{self.color[2]})',
                'area_km2': self.area_km2,
            },
            'geometry': {
                'type': 'Polygon',
                'coordinates': [coords]
            }
        }


# =============================================================================
# GENERADOR DE ZONAS
# =============================================================================

class SpeciesZoneGenerator:
    """
    Genera zonas de pesca por especie.

    Combina:
    - Mapa de sustrato (rock/sand/mixed)
    - Mapa de profundidad
    - Preferencias de habitat de especies

    Output: Poligonos coloreados con especies recomendadas.
    """

    __slots__ = ('species_db', 'min_zone_area', '_zones')

    def __init__(
        self,
        species_db: Optional[Dict[str, SpeciesHabitat]] = None,
        min_zone_area: float = 0.001  # km^2
    ):
        """
        Args:
            species_db: Base de datos de especies (usa default si None)
            min_zone_area: Area minima de zona en km^2
        """
        self.species_db = species_db or SPECIES_DATABASE
        self.min_zone_area = min_zone_area
        self._zones: List[SpeciesZone] = []

    def generate_zones(
        self,
        substrate_grid: np.ndarray,
        depth_grid: np.ndarray,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float
    ) -> List[SpeciesZone]:
        """
        Genera zonas de pesca desde grillas de sustrato y profundidad.

        Args:
            substrate_grid: Array 2D con tipos de sustrato (0=rock, 1=sand, 2=mixed)
            depth_grid: Array 2D con profundidad en metros
            lat_min, lat_max, lon_min, lon_max: Limites geograficos

        Returns:
            Lista de SpeciesZone
        """
        if not HAS_CV2:
            logger.error("OpenCV requerido para generar zonas")
            return []

        zones = []
        h, w = substrate_grid.shape

        # Calcular resolucion
        pixel_lat = (lat_max - lat_min) / h
        pixel_lon = (lon_max - lon_min) / w

        # Crear mapa combinado sustrato + profundidad
        combined = self._create_combined_map(substrate_grid, depth_grid)

        # Encontrar regiones contiguas
        unique_values = np.unique(combined)

        zone_id = 0
        for value in unique_values:
            if value == 255:  # Valor invalido
                continue

            # Crear mascara para este tipo
            mask = (combined == value).astype(np.uint8) * 255

            # Encontrar contornos
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                # Calcular area
                area_pixels = cv2.contourArea(contour)
                area_km2 = self._pixels_to_km2(area_pixels, pixel_lat, pixel_lon)

                if area_km2 < self.min_zone_area:
                    continue

                # Extraer valores de esta region
                mask_region = np.zeros_like(mask)
                cv2.drawContours(mask_region, [contour], -1, 255, -1)

                substrate_values = substrate_grid[mask_region > 0]
                depth_values = depth_grid[mask_region > 0]

                # Determinar sustrato dominante
                substrate = self._get_dominant_substrate(substrate_values)

                # Calcular profundidad promedio
                valid_depths = depth_values[~np.isnan(depth_values)]
                avg_depth = float(np.median(valid_depths)) if len(valid_depths) > 0 else 0

                # Determinar zona de profundidad
                depth_zone = self._get_depth_zone(avg_depth)

                # Calcular scores por especie
                species_scores = {}
                for species_id, habitat in self.species_db.items():
                    score = habitat.get_affinity(substrate, avg_depth)
                    species_scores[species_id] = round(score, 3)

                # Ordenar especies por score
                sorted_species = sorted(
                    species_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )

                if not sorted_species:
                    continue

                primary = sorted_species[0][0]
                secondary = [s[0] for s in sorted_species[1:4] if s[1] > 0.5]

                # Obtener color de la especie principal
                color = self.species_db[primary].color

                # Convertir contorno a coordenadas geograficas
                polygon = self._contour_to_coords(
                    contour, lat_min, lat_max, lon_min, lon_max, h, w
                )

                # Calcular centro
                M = cv2.moments(contour)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    center_lat = lat_max - (cy / h) * (lat_max - lat_min)
                    center_lon = lon_min + (cx / w) * (lon_max - lon_min)
                else:
                    center_lat = (lat_min + lat_max) / 2
                    center_lon = (lon_min + lon_max) / 2

                zone = SpeciesZone(
                    zone_id=f"zone_{zone_id:04d}",
                    polygon=polygon,
                    center=(center_lat, center_lon),
                    substrate=substrate,
                    depth_zone=depth_zone,
                    avg_depth=avg_depth,
                    species_scores=species_scores,
                    primary_species=primary,
                    secondary_species=secondary,
                    color=color,
                    area_km2=area_km2
                )

                zones.append(zone)
                zone_id += 1

        self._zones = zones
        logger.info(f"Generadas {len(zones)} zonas de especies")

        return zones

    def _create_combined_map(
        self,
        substrate: np.ndarray,
        depth: np.ndarray
    ) -> np.ndarray:
        """Crea mapa combinado de sustrato + zona de profundidad."""
        # Clasificar profundidad en zonas (0-4)
        depth_zones = np.zeros_like(depth, dtype=np.uint8)
        abs_depth = np.abs(depth)

        depth_zones[abs_depth <= 2] = 0    # Intertidal
        depth_zones[(abs_depth > 2) & (abs_depth <= 10)] = 1   # Shallow
        depth_zones[(abs_depth > 10) & (abs_depth <= 30)] = 2  # Moderate
        depth_zones[(abs_depth > 30) & (abs_depth <= 100)] = 3 # Deep
        depth_zones[abs_depth > 100] = 4   # Very deep
        depth_zones[np.isnan(depth)] = 255  # Invalid

        # Combinar: substrate * 5 + depth_zone
        # Esto da valores unicos para cada combinacion
        combined = substrate.astype(np.uint8) * 5 + depth_zones
        combined[depth_zones == 255] = 255

        return combined

    def _get_dominant_substrate(self, values: np.ndarray) -> SubstrateType:
        """Determina sustrato dominante en una region."""
        if len(values) == 0:
            return SubstrateType.UNKNOWN

        counts = np.bincount(values.astype(int), minlength=4)
        dominant = np.argmax(counts)

        type_map = {
            0: SubstrateType.ROCK,
            1: SubstrateType.SAND,
            2: SubstrateType.MIXED,
            3: SubstrateType.UNKNOWN
        }

        return type_map.get(dominant, SubstrateType.UNKNOWN)

    def _get_depth_zone(self, depth: float) -> DepthZone:
        """Clasifica profundidad en zona."""
        abs_d = abs(depth)

        if abs_d <= 2:
            return DepthZone.INTERTIDAL
        elif abs_d <= 10:
            return DepthZone.SHALLOW
        elif abs_d <= 30:
            return DepthZone.MODERATE
        elif abs_d <= 100:
            return DepthZone.DEEP
        else:
            return DepthZone.VERY_DEEP

    def _contour_to_coords(
        self,
        contour: np.ndarray,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        h: int,
        w: int
    ) -> List[Tuple[float, float]]:
        """Convierte contorno de pixeles a coordenadas."""
        coords = []

        # Simplificar contorno
        epsilon = 0.01 * cv2.arcLength(contour, True)
        simplified = cv2.approxPolyDP(contour, epsilon, True)

        for point in simplified:
            px, py = point[0]
            lat = lat_max - (py / h) * (lat_max - lat_min)
            lon = lon_min + (px / w) * (lon_max - lon_min)
            coords.append((lat, lon))

        return coords

    def _pixels_to_km2(
        self,
        pixels: float,
        pixel_lat: float,
        pixel_lon: float
    ) -> float:
        """Convierte area en pixeles a km^2."""
        # Aproximacion: 1 grado lat ≈ 111 km
        km_per_pixel_lat = pixel_lat * 111
        km_per_pixel_lon = pixel_lon * 111  # Simplificado

        return pixels * km_per_pixel_lat * km_per_pixel_lon

    def get_zones_geojson(self) -> Dict:
        """Retorna todas las zonas en formato GeoJSON."""
        features = [zone.to_geojson_feature() for zone in self._zones]

        return {
            'type': 'FeatureCollection',
            'features': features
        }

    def get_species_summary(self) -> Dict[str, Dict]:
        """Resumen de zonas por especie."""
        summary = {}

        for species_id in self.species_db:
            zones_for_species = [
                z for z in self._zones
                if z.primary_species == species_id or species_id in z.secondary_species
            ]

            total_area = sum(z.area_km2 for z in zones_for_species)
            primary_zones = [z for z in zones_for_species if z.primary_species == species_id]

            summary[species_id] = {
                'name': self.species_db[species_id].name_es,
                'total_zones': len(zones_for_species),
                'primary_zones': len(primary_zones),
                'total_area_km2': round(total_area, 3),
                'color': self.species_db[species_id].color,
            }

        return summary


# =============================================================================
# FUNCIONES DE CONVENIENCIA
# =============================================================================

def generate_species_zones(
    substrate_grid: np.ndarray,
    depth_grid: np.ndarray,
    bounds: Tuple[float, float, float, float]  # lat_min, lat_max, lon_min, lon_max
) -> List[SpeciesZone]:
    """
    Genera zonas de especies de forma simple.

    Args:
        substrate_grid: Mapa de sustrato
        depth_grid: Mapa de profundidad
        bounds: (lat_min, lat_max, lon_min, lon_max)

    Returns:
        Lista de SpeciesZone
    """
    generator = SpeciesZoneGenerator()
    return generator.generate_zones(
        substrate_grid, depth_grid,
        bounds[0], bounds[1], bounds[2], bounds[3]
    )


def get_species_at_point(
    lat: float,
    lon: float,
    zones: List[SpeciesZone]
) -> Optional[SpeciesZone]:
    """
    Encuentra zona de especies en un punto.

    Args:
        lat, lon: Coordenadas
        zones: Lista de zonas

    Returns:
        SpeciesZone si el punto esta en alguna zona
    """
    for zone in zones:
        if _point_in_polygon(lat, lon, zone.polygon):
            return zone
    return None


def _point_in_polygon(
    lat: float,
    lon: float,
    polygon: List[Tuple[float, float]]
) -> bool:
    """Verifica si un punto esta dentro de un poligono (ray casting)."""
    n = len(polygon)
    inside = False

    j = n - 1
    for i in range(n):
        lat_i, lon_i = polygon[i]
        lat_j, lon_j = polygon[j]

        if ((lon_i > lon) != (lon_j > lon)) and \
           (lat < (lat_j - lat_i) * (lon - lon_i) / (lon_j - lon_i) + lat_i):
            inside = not inside

        j = i

    return inside
