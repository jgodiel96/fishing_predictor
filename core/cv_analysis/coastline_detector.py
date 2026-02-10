"""
Coastline Detector using Computer Vision.

Detects precise water/land boundary from satellite imagery.
Uses color analysis (HSV) and edge detection for high precision.

For SAM integration, requires: pip install segment-anything torch
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import math

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from PIL import Image
    import requests
    from io import BytesIO
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURACION
# =============================================================================

@dataclass(frozen=True)
class TileConfig:
    """Configuracion para descarga de tiles."""
    zoom: int = 17  # ~1.2m/pixel
    tile_size: int = 256
    source: str = 'esri'  # 'esri', 'google', 'osm'


TILE_SOURCES = {
    'esri': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    'google': 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
}

# Rangos HSV para deteccion de agua de oceano
# H: 0-180 en OpenCV (azul ~100-130)
# S: Saturacion (agua tiene saturacion moderada)
# V: Valor/brillo
WATER_HSV_RANGES = {
    'deep_ocean': {
        'lower': np.array([95, 40, 30]),
        'upper': np.array([130, 255, 180])
    },
    'coastal_water': {
        'lower': np.array([85, 30, 40]),
        'upper': np.array([115, 200, 160])
    },
}


class CoastlineDetectorCV:
    """
    Detector de linea costera usando Computer Vision.

    Metodos:
    1. Descarga tiles satelitales de alta resolucion
    2. Detecta agua usando analisis HSV
    3. Extrae contorno como linea costera
    4. Refina y suaviza el resultado

    Atributos:
        config: Configuracion de tiles
        _tile_cache: Cache de tiles descargados
    """

    __slots__ = ('config', '_tile_cache', '_session')

    def __init__(self, config: Optional[TileConfig] = None):
        self.config = config or TileConfig()
        self._tile_cache: Dict[str, np.ndarray] = {}
        self._session = None

    def detect_coastline(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        simplify_tolerance: float = 0.00005
    ) -> List[Tuple[float, float]]:
        """
        Detecta linea costera en el area especificada.

        Args:
            lat_min, lat_max, lon_min, lon_max: Limites del area
            simplify_tolerance: Tolerancia para simplificacion (grados)

        Returns:
            Lista de puntos (lat, lon) de la linea costera
        """
        if not HAS_CV2:
            logger.error("OpenCV no instalado. Ejecuta: pip install opencv-python")
            return []

        logger.info(f"Detectando linea costera en area: ({lat_min:.4f}, {lon_min:.4f}) - ({lat_max:.4f}, {lon_max:.4f})")

        # 1. Descargar imagen del area
        image, geo_transform = self._download_area_image(
            lat_min, lat_max, lon_min, lon_max
        )

        if image is None:
            logger.error("No se pudo descargar imagen satelital")
            return []

        # 2. Detectar mascara de agua
        water_mask = self._detect_water_mask(image)

        # 3. Extraer contorno
        contours = self._extract_contours(water_mask)

        if not contours:
            logger.warning("No se detectaron contornos")
            return []

        # 4. Convertir a coordenadas geograficas
        coastline_points = self._pixels_to_coords(
            contours, geo_transform, image.shape
        )

        # 5. Simplificar si es necesario
        if simplify_tolerance > 0:
            coastline_points = self._simplify_line(coastline_points, simplify_tolerance)

        logger.info(f"Linea costera detectada: {len(coastline_points)} puntos")

        return coastline_points

    def _download_area_image(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float
    ) -> Tuple[Optional[np.ndarray], Dict]:
        """Descarga y une tiles para cubrir el area."""
        if not HAS_PIL:
            return None, {}

        zoom = self.config.zoom
        tile_size = self.config.tile_size

        # Calcular tiles necesarios
        x_min, y_max = self._latlon_to_tile(lat_min, lon_min, zoom)
        x_max, y_min = self._latlon_to_tile(lat_max, lon_max, zoom)

        # Asegurar orden correcto
        x_min, x_max = min(x_min, x_max), max(x_min, x_max)
        y_min, y_max = min(y_min, y_max), max(y_min, y_max)

        width = (x_max - x_min + 1) * tile_size
        height = (y_max - y_min + 1) * tile_size

        # Crear imagen compuesta
        composite = np.zeros((height, width, 3), dtype=np.uint8)

        tiles_downloaded = 0
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                tile = self._download_tile(x, y, zoom)
                if tile is not None:
                    px = (x - x_min) * tile_size
                    py = (y - y_min) * tile_size
                    composite[py:py+tile_size, px:px+tile_size] = tile
                    tiles_downloaded += 1

        logger.debug(f"Descargados {tiles_downloaded} tiles")

        # Calcular geo transform
        nw_lat, nw_lon = self._tile_to_latlon(x_min, y_min, zoom)
        se_lat, se_lon = self._tile_to_latlon(x_max + 1, y_max + 1, zoom)

        geo_transform = {
            'lat_min': se_lat,
            'lat_max': nw_lat,
            'lon_min': nw_lon,
            'lon_max': se_lon,
            'pixel_lat': (nw_lat - se_lat) / height,
            'pixel_lon': (se_lon - nw_lon) / width
        }

        return composite, geo_transform

    def _download_tile(self, x: int, y: int, z: int) -> Optional[np.ndarray]:
        """Descarga un tile individual."""
        cache_key = f"{z}_{x}_{y}"

        if cache_key in self._tile_cache:
            return self._tile_cache[cache_key]

        url_template = TILE_SOURCES.get(self.config.source, TILE_SOURCES['esri'])
        url = url_template.format(x=x, y=y, z=z)

        try:
            if self._session is None:
                self._session = requests.Session()

            response = self._session.get(url, timeout=10)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                tile = np.array(img.convert('RGB'))
                self._tile_cache[cache_key] = tile
                return tile
        except Exception as e:
            logger.debug(f"Error descargando tile {cache_key}: {e}")

        return None

    def _detect_water_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Detecta pixeles de agua usando multiples criterios:
        1. Analisis HSV para tonos azules/cyan
        2. Indice de agua NDWI simplificado
        3. Filtrar regiones que tocan el borde de la imagen (oceano)
        """
        h, w = image.shape[:2]

        # Convertir a HSV y extraer canales RGB
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        blue = image[:, :, 2].astype(np.float32)
        green = image[:, :, 1].astype(np.float32)
        red = image[:, :, 0].astype(np.float32)

        # 1. Mascara HSV amplia para agua (azul a cyan)
        # Rango mas amplio para capturar diferentes tonos de oceano
        water_hsv_lower = np.array([85, 20, 20])
        water_hsv_upper = np.array([135, 255, 200])
        hsv_mask = cv2.inRange(hsv, water_hsv_lower, water_hsv_upper)

        # 2. Indice simplificado: agua tiene mas azul que rojo
        # y generalmente valores bajos de brillo comparado con arena/ciudad
        blue_ratio = np.zeros_like(blue)
        denom = red + green + blue + 1
        blue_ratio = (blue - red) / denom

        # Agua: diferencia positiva entre azul y rojo
        water_index = (blue_ratio > 0.05).astype(np.uint8) * 255

        # 3. Combinar con OR para capturar mas agua
        combined = cv2.bitwise_or(hsv_mask, water_index)

        # 4. Filtrar por brillo - agua no es muy brillante
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        not_too_bright = (gray < 180).astype(np.uint8) * 255
        combined = cv2.bitwise_and(combined, not_too_bright)

        # 5. Operaciones morfologicas
        kernel = np.ones((7, 7), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

        # 6. Encontrar la region que toca el borde izquierdo (donde esta el oceano)
        contours, _ = cv2.findContours(
            combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            # Buscar contornos que toquen el borde izquierdo o inferior
            ocean_contours = []
            for cnt in contours:
                # Verificar si toca el borde
                x, y, cw, ch = cv2.boundingRect(cnt)
                touches_left = x <= 5
                touches_bottom = (y + ch) >= h - 5

                if touches_left or touches_bottom:
                    area = cv2.contourArea(cnt)
                    if area > (h * w * 0.01):  # Al menos 1% del area
                        ocean_contours.append(cnt)

            if ocean_contours:
                # Usar el mas grande que toca el borde
                largest = max(ocean_contours, key=cv2.contourArea)
                ocean_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(ocean_mask, [largest], -1, 255, -1)
                combined = ocean_mask
            elif contours:
                # Si ninguno toca el borde, usar el mas grande
                largest = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest) > (h * w * 0.05):
                    ocean_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.drawContours(ocean_mask, [largest], -1, 255, -1)
                    combined = ocean_mask

        # 7. Suavizar
        kernel_smooth = np.ones((11, 11), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_smooth)

        return combined

    def _extract_contours(self, mask: np.ndarray) -> List[np.ndarray]:
        """Extrae contornos de la mascara."""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        if not contours:
            return []

        # Filtrar contornos muy pequenos
        min_area = mask.shape[0] * mask.shape[1] * 0.001  # 0.1% del area
        filtered = [c for c in contours if cv2.contourArea(c) > min_area]

        # Ordenar por area (mayor primero)
        filtered.sort(key=cv2.contourArea, reverse=True)

        return filtered

    def _pixels_to_coords(
        self,
        contours: List[np.ndarray],
        geo_transform: Dict,
        image_shape: Tuple[int, int]
    ) -> List[Tuple[float, float]]:
        """Convierte pixeles a coordenadas geograficas."""
        height, width = image_shape[:2]
        points = []

        # Usar el contorno mas grande (costa principal)
        if contours:
            main_contour = contours[0]

            for point in main_contour:
                px, py = point[0]

                # Convertir pixel a lat/lon
                lon = geo_transform['lon_min'] + (px / width) * (geo_transform['lon_max'] - geo_transform['lon_min'])
                lat = geo_transform['lat_max'] - (py / height) * (geo_transform['lat_max'] - geo_transform['lat_min'])

                points.append((lat, lon))

        return points

    def _simplify_line(
        self,
        points: List[Tuple[float, float]],
        tolerance: float
    ) -> List[Tuple[float, float]]:
        """Simplifica la linea usando Douglas-Peucker."""
        if len(points) < 3:
            return points

        # Implementacion simple de Douglas-Peucker
        def perpendicular_distance(point, line_start, line_end):
            if line_start == line_end:
                return math.sqrt((point[0] - line_start[0])**2 + (point[1] - line_start[1])**2)

            n = abs((line_end[1] - line_start[1]) * point[0] -
                   (line_end[0] - line_start[0]) * point[1] +
                   line_end[0] * line_start[1] -
                   line_end[1] * line_start[0])
            d = math.sqrt((line_end[1] - line_start[1])**2 + (line_end[0] - line_start[0])**2)

            return n / d if d > 0 else 0

        def douglas_peucker(points, tolerance):
            if len(points) <= 2:
                return points

            # Encontrar punto mas lejano
            max_dist = 0
            max_idx = 0

            for i in range(1, len(points) - 1):
                dist = perpendicular_distance(points[i], points[0], points[-1])
                if dist > max_dist:
                    max_dist = dist
                    max_idx = i

            if max_dist > tolerance:
                left = douglas_peucker(points[:max_idx + 1], tolerance)
                right = douglas_peucker(points[max_idx:], tolerance)
                return left[:-1] + right
            else:
                return [points[0], points[-1]]

        return douglas_peucker(points, tolerance)

    @staticmethod
    def _latlon_to_tile(lat: float, lon: float, zoom: int) -> Tuple[int, int]:
        """Convierte lat/lon a coordenadas de tile."""
        n = 2 ** zoom
        x = int((lon + 180) / 360 * n)
        lat_rad = math.radians(lat)
        y = int((1 - math.asinh(math.tan(lat_rad)) / math.pi) / 2 * n)
        return x, y

    @staticmethod
    def _tile_to_latlon(x: int, y: int, zoom: int) -> Tuple[float, float]:
        """Convierte coordenadas de tile a lat/lon."""
        n = 2 ** zoom
        lon = x / n * 360 - 180
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
        lat = math.degrees(lat_rad)
        return lat, lon

    def clear_cache(self) -> None:
        """Limpia la cache de tiles."""
        self._tile_cache.clear()


# =============================================================================
# FUNCION DE CONVENIENCIA
# =============================================================================

def detect_coastline_cv(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    zoom: int = 17
) -> List[Tuple[float, float]]:
    """
    Funcion de conveniencia para detectar linea costera.

    Args:
        lat_min, lat_max, lon_min, lon_max: Limites del area
        zoom: Nivel de zoom (mayor = mas detalle)

    Returns:
        Lista de puntos (lat, lon)
    """
    detector = CoastlineDetectorCV(TileConfig(zoom=zoom))
    return detector.detect_coastline(lat_min, lat_max, lon_min, lon_max)
