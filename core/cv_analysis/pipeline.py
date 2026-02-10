"""
CV Analysis Pipeline for Fishing Predictor V8.

Integrates all CV components into a single pipeline:
1. Coastline detection (precise water/land boundary)
2. Substrate classification (rock/sand/mixed)
3. Bathymetry estimation (SDB + GEBCO fusion)
4. Species zone generation (colored polygons)

Usage:
    pipeline = CVAnalysisPipeline()
    result = pipeline.analyze_area(lat_min, lat_max, lon_min, lon_max)
"""

from __future__ import annotations

import logging
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from datetime import datetime

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from .coastline_detector import CoastlineDetectorCV, TileConfig
from .substrate_classifier import SubstrateClassifier, SubstrateType, SubstrateResult
from .bathymetry import (
    SatelliteBathymetry, GEBCOBathymetry, BathymetryFusion,
    BathymetryResult, get_depth_zones
)
from .species_zones import (
    SpeciesZoneGenerator, SpeciesZone, DepthZone,
    SPECIES_DATABASE
)

logger = logging.getLogger(__name__)


# =============================================================================
# RESULTADO DEL PIPELINE
# =============================================================================

@dataclass
class CVAnalysisResult:
    """Resultado completo del analisis CV."""
    # Metadatos
    timestamp: str
    bounds: Tuple[float, float, float, float]  # lat_min, lat_max, lon_min, lon_max
    processing_time_s: float

    # Resultados
    coastline: List[Tuple[float, float]]
    water_mask: Optional[np.ndarray]
    substrate_grid: Optional[np.ndarray]
    depth_grid: Optional[np.ndarray]
    species_zones: List[SpeciesZone]

    # Estadisticas
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convierte a diccionario serializable."""
        return {
            'timestamp': self.timestamp,
            'bounds': self.bounds,
            'processing_time_s': self.processing_time_s,
            'coastline_points': len(self.coastline),
            'species_zones_count': len(self.species_zones),
            'stats': self.stats,
        }

    def save_geojson(self, path: Path) -> None:
        """Guarda resultados en formato GeoJSON."""
        features = []

        # Coastline como LineString
        if self.coastline:
            coords = [[lon, lat] for lat, lon in self.coastline]
            features.append({
                'type': 'Feature',
                'properties': {'type': 'coastline'},
                'geometry': {
                    'type': 'LineString',
                    'coordinates': coords
                }
            })

        # Species zones como Polygons
        for zone in self.species_zones:
            features.append(zone.to_geojson_feature())

        geojson = {
            'type': 'FeatureCollection',
            'features': features,
            'properties': {
                'timestamp': self.timestamp,
                'bounds': self.bounds,
            }
        }

        with open(path, 'w') as f:
            json.dump(geojson, f, indent=2)

        logger.info(f"GeoJSON guardado: {path}")


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

class CVAnalysisPipeline:
    """
    Pipeline completo de analisis por Computer Vision.

    Ejecuta:
    1. Descarga imagen satelital
    2. Detecta linea costera precisa
    3. Clasifica sustrato (rock/sand/mixed)
    4. Estima batimetria (SDB + GEBCO)
    5. Genera zonas de especies

    Attributes:
        tile_config: Configuracion de tiles
        gebco_path: Ruta a archivo GEBCO
        grid_size: Tamano de celda para clasificacion
    """

    __slots__ = (
        'tile_config', 'gebco_path', 'grid_size',
        '_coastline_detector', '_substrate_classifier',
        '_bathymetry', '_zone_generator',
        '_last_image', '_last_geo_transform', '_last_water_mask'
    )

    def __init__(
        self,
        tile_config: Optional[TileConfig] = None,
        gebco_path: Optional[Path] = None,
        grid_size: int = 64
    ):
        """
        Args:
            tile_config: Configuracion de tiles satelitales
            gebco_path: Ruta al archivo GEBCO NetCDF
            grid_size: Tamano de celda para clasificacion
        """
        self.tile_config = tile_config or TileConfig(zoom=17)
        self.gebco_path = gebco_path
        self.grid_size = grid_size

        # Componentes
        self._coastline_detector = CoastlineDetectorCV(self.tile_config)
        self._substrate_classifier = SubstrateClassifier(window_size=grid_size)
        self._bathymetry = BathymetryFusion(gebco_path)
        self._zone_generator = SpeciesZoneGenerator()

        # Cache de ultimo analisis
        self._last_image: Optional[np.ndarray] = None
        self._last_geo_transform: Optional[Dict] = None
        self._last_water_mask: Optional[np.ndarray] = None

    def analyze_area(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        include_species_zones: bool = True
    ) -> CVAnalysisResult:
        """
        Analiza un area completa.

        Args:
            lat_min, lat_max, lon_min, lon_max: Limites del area
            include_species_zones: Si generar zonas de especies

        Returns:
            CVAnalysisResult con todos los resultados
        """
        import time
        start_time = time.time()

        logger.info(f"Iniciando analisis CV: ({lat_min:.4f}, {lon_min:.4f}) - ({lat_max:.4f}, {lon_max:.4f})")

        bounds = (lat_min, lat_max, lon_min, lon_max)
        stats = {}

        # 1. Descargar imagen satelital
        logger.info("Paso 1/5: Descargando imagen satelital...")
        image, geo_transform = self._download_image(lat_min, lat_max, lon_min, lon_max)

        if image is None:
            logger.error("No se pudo descargar imagen")
            return CVAnalysisResult(
                timestamp=datetime.now().isoformat(),
                bounds=bounds,
                processing_time_s=time.time() - start_time,
                coastline=[],
                water_mask=None,
                substrate_grid=None,
                depth_grid=None,
                species_zones=[],
                stats={'error': 'image_download_failed'}
            )

        self._last_image = image
        self._last_geo_transform = geo_transform
        stats['image_shape'] = image.shape

        # 2. Detectar linea costera
        logger.info("Paso 2/5: Detectando linea costera...")
        coastline = self._detect_coastline(image, geo_transform)
        stats['coastline_points'] = len(coastline)

        # 3. Crear mascara de agua
        logger.info("Paso 3/5: Creando mascara de agua...")
        water_mask = self._create_water_mask(image)
        self._last_water_mask = water_mask
        stats['water_percentage'] = float(np.sum(water_mask > 0) / water_mask.size * 100)

        # 4. Clasificar sustrato
        logger.info("Paso 4/5: Clasificando sustrato...")
        substrate_grid = self._classify_substrate(image, water_mask)
        stats['substrate_distribution'] = self._get_substrate_stats(substrate_grid)

        # 5. Estimar batimetria
        logger.info("Paso 5/5: Estimando batimetria...")
        depth_grid = self._estimate_bathymetry(
            image, water_mask, geo_transform, lat_min, lat_max, lon_min, lon_max
        )
        stats['depth_range'] = self._get_depth_stats(depth_grid)

        # 6. Generar zonas de especies (solo en agua)
        species_zones = []
        if include_species_zones and water_mask is not None and depth_grid is not None:
            logger.info("Generando zonas de especies...")

            # Generar zonas basadas en la mascara de agua y profundidad
            species_zones = self._generate_water_zones(
                water_mask, depth_grid, geo_transform,
                lat_min, lat_max, lon_min, lon_max
            )
            stats['species_zones'] = len(species_zones)
            if species_zones:
                stats['species_summary'] = self._zone_generator.get_species_summary()

        processing_time = time.time() - start_time
        logger.info(f"Analisis completado en {processing_time:.2f}s")

        return CVAnalysisResult(
            timestamp=datetime.now().isoformat(),
            bounds=bounds,
            processing_time_s=processing_time,
            coastline=coastline,
            water_mask=water_mask,
            substrate_grid=substrate_grid,
            depth_grid=depth_grid,
            species_zones=species_zones,
            stats=stats
        )

    def _download_image(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float
    ) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """Descarga imagen satelital del area."""
        return self._coastline_detector._download_area_image(
            lat_min, lat_max, lon_min, lon_max
        )

    def _detect_coastline(
        self,
        image: np.ndarray,
        geo_transform: Dict
    ) -> List[Tuple[float, float]]:
        """Detecta linea costera desde imagen."""
        # Usar mascara de agua para extraer contorno
        water_mask = self._coastline_detector._detect_water_mask(image)
        contours = self._coastline_detector._extract_contours(water_mask)

        if not contours:
            return []

        return self._coastline_detector._pixels_to_coords(
            contours, geo_transform, image.shape
        )

    def _create_water_mask(self, image: np.ndarray) -> np.ndarray:
        """Crea mascara de agua."""
        return self._coastline_detector._detect_water_mask(image)

    def _classify_substrate(
        self,
        image: np.ndarray,
        water_mask: np.ndarray
    ) -> np.ndarray:
        """
        Clasifica sustrato en grilla.

        Para zonas de pesca, analizamos el fondo visible en aguas poco profundas
        y la zona costera inmediata.
        """
        # Crear mascara de zona costera (dilatar agua para incluir orilla)
        kernel = np.ones((50, 50), np.uint8)
        coastal_zone = cv2.dilate(water_mask, kernel, iterations=2)

        # La zona de interes es donde hay agua O zona costera cercana
        # pero marcamos el agua como tipo especial
        return self._substrate_classifier.classify_grid(
            image, self.grid_size, water_mask
        )

    def _estimate_bathymetry(
        self,
        image: np.ndarray,
        water_mask: np.ndarray,
        geo_transform: Dict,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float
    ) -> np.ndarray:
        """Estima batimetria."""
        # Obtener grilla fusionada
        depth_grid, lats, lons = self._bathymetry.get_depth_grid(
            lat_min, lat_max, lon_min, lon_max,
            image, geo_transform,
            resolution=0.0005  # ~50m
        )

        # Si no hay GEBCO, usar solo SDB
        if depth_grid.size == 0:
            sdb = SatelliteBathymetry()
            depth_grid = sdb.estimate_depth(image, water_mask)

        return depth_grid

    def _generate_water_zones(
        self,
        water_mask: np.ndarray,
        depth_grid: np.ndarray,
        geo_transform: Dict,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float
    ) -> List[SpeciesZone]:
        """
        Genera zonas de especies DENTRO del agua.

        Divide el area de agua en zonas por profundidad y genera
        recomendaciones de especies para cada zona.
        """
        from .species_zones import SpeciesZone, DepthZone, SubstrateType, SPECIES_DATABASE

        zones = []
        h, w = water_mask.shape

        # Encontrar contornos del agua
        contours, _ = cv2.findContours(
            water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return zones

        # Usar el contorno mas grande (oceano principal)
        ocean_contour = max(contours, key=cv2.contourArea)

        # Crear zonas por bandas de profundidad desde la costa
        # Dilatar la mascara de agua en pasos para crear bandas
        depth_bands = [
            ('shallow', 2, 10, (70, 130, 180)),      # Azul - 0-50px de costa
            ('moderate', 10, 30, (100, 149, 237)),   # Azul claro - 50-150px
            ('deep', 30, 100, (65, 105, 225)),       # Azul royal - 150-300px
        ]

        pixel_lat = (lat_max - lat_min) / h
        pixel_lon = (lon_max - lon_min) / w

        # Crear zona para cada banda de profundidad
        for band_name, depth_min, depth_max, color in depth_bands:
            # Erosionar el agua para obtener bandas
            if band_name == 'shallow':
                erosion = 0
                dilation = 50
            elif band_name == 'moderate':
                erosion = 50
                dilation = 150
            else:
                erosion = 150
                dilation = 300

            # Crear mascara de la banda
            kernel = np.ones((5, 5), np.uint8)

            if erosion > 0:
                inner = cv2.erode(water_mask, kernel, iterations=erosion // 5)
            else:
                inner = np.zeros_like(water_mask)

            outer = cv2.erode(water_mask, kernel, iterations=dilation // 5)
            band_mask = cv2.subtract(outer, inner)

            # Verificar que la banda tiene area
            if np.sum(band_mask > 0) < 100:
                continue

            # Encontrar contorno de la banda
            band_contours, _ = cv2.findContours(
                band_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if not band_contours:
                continue

            # Usar contorno mas grande de la banda
            band_contour = max(band_contours, key=cv2.contourArea)
            area_pixels = cv2.contourArea(band_contour)
            area_km2 = area_pixels * pixel_lat * pixel_lon * 111 * 111

            if area_km2 < 0.001:
                continue

            # Simplificar contorno
            epsilon = 0.005 * cv2.arcLength(band_contour, True)
            simplified = cv2.approxPolyDP(band_contour, epsilon, True)

            # Convertir a coordenadas geograficas
            polygon = []
            for point in simplified:
                px, py = point[0]
                lon = lon_min + (px / w) * (lon_max - lon_min)
                lat = lat_max - (py / h) * (lat_max - lat_min)
                polygon.append((lat, lon))

            # Calcular centro
            M = cv2.moments(band_contour)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                center_lon = lon_min + (cx / w) * (lon_max - lon_min)
                center_lat = lat_max - (cy / h) * (lat_max - lat_min)
            else:
                center_lat = (lat_min + lat_max) / 2
                center_lon = (lon_min + lon_max) / 2

            # Determinar profundidad media de la banda
            avg_depth = -(depth_min + depth_max) / 2

            # Determinar sustrato (asumimos mixto para agua, rocky cerca de costa)
            if band_name == 'shallow':
                substrate = SubstrateType.MIXED
            else:
                substrate = SubstrateType.SAND

            # Calcular scores por especie
            species_scores = {}
            for species_id, habitat in SPECIES_DATABASE.items():
                score = habitat.get_affinity(substrate, avg_depth)
                species_scores[species_id] = round(score, 3)

            # Ordenar y obtener primarias
            sorted_species = sorted(species_scores.items(), key=lambda x: x[1], reverse=True)
            primary = sorted_species[0][0] if sorted_species else 'unknown'
            secondary = [s[0] for s in sorted_species[1:4] if s[1] > 0.5]

            # Obtener color de especie primaria
            species_color = SPECIES_DATABASE[primary].color if primary in SPECIES_DATABASE else color

            # Determinar zona de profundidad
            if depth_max <= 10:
                depth_zone = DepthZone.SHALLOW
            elif depth_max <= 30:
                depth_zone = DepthZone.MODERATE
            else:
                depth_zone = DepthZone.DEEP

            zone = SpeciesZone(
                zone_id=f"water_{band_name}",
                polygon=polygon,
                center=(center_lat, center_lon),
                substrate=substrate,
                depth_zone=depth_zone,
                avg_depth=avg_depth,
                species_scores=species_scores,
                primary_species=primary,
                secondary_species=secondary,
                color=species_color,
                area_km2=area_km2
            )
            zones.append(zone)

        self._zone_generator._zones = zones
        return zones

    def _align_grids(
        self,
        substrate: np.ndarray,
        depth: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Alinea grids a la misma dimension."""
        if not HAS_CV2:
            return substrate, depth

        # Usar la dimension menor
        target_h = min(substrate.shape[0], depth.shape[0])
        target_w = min(substrate.shape[1], depth.shape[1])

        if substrate.shape != (target_h, target_w):
            substrate = cv2.resize(
                substrate, (target_w, target_h),
                interpolation=cv2.INTER_NEAREST
            )

        if depth.shape != (target_h, target_w):
            depth = cv2.resize(
                depth.astype(np.float32), (target_w, target_h),
                interpolation=cv2.INTER_LINEAR
            )

        return substrate, depth

    def _get_substrate_stats(self, grid: np.ndarray) -> Dict[str, float]:
        """Calcula estadisticas de sustrato."""
        total = grid.size
        return {
            'rock_pct': float(np.sum(grid == 0) / total * 100),
            'sand_pct': float(np.sum(grid == 1) / total * 100),
            'mixed_pct': float(np.sum(grid == 2) / total * 100),
            'water_pct': float(np.sum(grid == 3) / total * 100),
        }

    def _get_depth_stats(self, grid: np.ndarray) -> Dict[str, float]:
        """Calcula estadisticas de profundidad."""
        valid = grid[~np.isnan(grid)]
        if len(valid) == 0:
            return {'min': 0, 'max': 0, 'mean': 0}

        return {
            'min': float(np.min(valid)),
            'max': float(np.max(valid)),
            'mean': float(np.mean(valid)),
        }

    def get_depth_at_point(
        self,
        lat: float,
        lon: float
    ) -> BathymetryResult:
        """
        Obtiene profundidad en un punto especifico.

        Args:
            lat, lon: Coordenadas

        Returns:
            BathymetryResult
        """
        pixel_x, pixel_y = None, None

        if self._last_image is not None and self._last_geo_transform is not None:
            gt = self._last_geo_transform
            pixel_x = int((lon - gt['lon_min']) / gt['pixel_lon'])
            pixel_y = int((gt['lat_max'] - lat) / gt['pixel_lat'])

            # Validar limites
            h, w = self._last_image.shape[:2]
            if not (0 <= pixel_x < w and 0 <= pixel_y < h):
                pixel_x, pixel_y = None, None

        return self._bathymetry.get_depth(
            lat, lon,
            self._last_image, pixel_x, pixel_y
        )

    def get_substrate_at_point(
        self,
        lat: float,
        lon: float
    ) -> SubstrateResult:
        """
        Obtiene tipo de sustrato en un punto.

        Args:
            lat, lon: Coordenadas

        Returns:
            SubstrateResult
        """
        if self._last_image is None or self._last_geo_transform is None:
            return SubstrateResult(
                substrate_type=SubstrateType.UNKNOWN,
                confidence=0.0,
                rock_probability=0.5,
                sand_probability=0.5,
                texture_variance=0.0,
                mean_brightness=0.0
            )

        gt = self._last_geo_transform
        pixel_x = int((lon - gt['lon_min']) / gt['pixel_lon'])
        pixel_y = int((gt['lat_max'] - lat) / gt['pixel_lat'])

        h, w = self._last_image.shape[:2]

        # Extraer region
        half = self.grid_size // 2
        y1 = max(0, pixel_y - half)
        y2 = min(h, pixel_y + half)
        x1 = max(0, pixel_x - half)
        x2 = min(w, pixel_x + half)

        region = self._last_image[y1:y2, x1:x2]

        # Mascara de agua para la region
        water_region = None
        if self._last_water_mask is not None:
            water_region = self._last_water_mask[y1:y2, x1:x2]
            # Invertir para analizar tierra
            water_region = cv2.bitwise_not(water_region) if HAS_CV2 else None

        return self._substrate_classifier.classify_region(region, water_region)

    def clear_cache(self) -> None:
        """Limpia caches."""
        self._coastline_detector.clear_cache()
        self._last_image = None
        self._last_geo_transform = None
        self._last_water_mask = None


# =============================================================================
# FUNCION DE CONVENIENCIA
# =============================================================================

def analyze_fishing_area(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    output_path: Optional[Path] = None
) -> CVAnalysisResult:
    """
    Analiza area de pesca completa.

    Args:
        lat_min, lat_max, lon_min, lon_max: Limites del area
        output_path: Ruta para guardar GeoJSON (opcional)

    Returns:
        CVAnalysisResult
    """
    pipeline = CVAnalysisPipeline()
    result = pipeline.analyze_area(lat_min, lat_max, lon_min, lon_max)

    if output_path:
        result.save_geojson(output_path)

    return result
