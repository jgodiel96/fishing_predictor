"""
Bathymetry Module for Fishing Predictor V8.

Provides depth estimation from multiple sources:
1. Satellite-Derived Bathymetry (SDB) - Stumpf algorithm for shallow water
2. GEBCO - Global bathymetry data for deep water
3. Fusion - Combines both sources intelligently

SDB uses Blue/Green band ratio: depth = m0 + m1 * ln(n * blue / green)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Union
import math

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from scipy.interpolate import RegularGridInterpolator
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import netCDF4 as nc
    HAS_NETCDF = True
except ImportError:
    HAS_NETCDF = False

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURACION
# =============================================================================

@dataclass(frozen=True)
class SDBConfig:
    """Configuracion para Satellite-Derived Bathymetry."""
    # Coeficientes Stumpf (calibrables por region)
    m0: float = 0.0       # Offset
    m1: float = 52.073    # Escala (tipico para agua clara)
    n: float = 1000.0     # Factor de normalizacion

    # Limites
    max_depth: float = 25.0   # SDB confiable hasta ~25m
    min_ratio: float = 0.5    # Ratio B/G minimo valido
    max_ratio: float = 2.0    # Ratio B/G maximo valido


@dataclass
class BathymetryResult:
    """Resultado de estimacion batimetrica."""
    depth: float              # Profundidad en metros (negativo = bajo agua)
    source: str               # 'sdb', 'gebco', 'fusion'
    confidence: float         # 0-1
    sdb_depth: Optional[float] = None
    gebco_depth: Optional[float] = None

    def __post_init__(self):
        self.depth = np.float32(self.depth)
        self.confidence = np.float32(self.confidence)


# =============================================================================
# SATELLITE-DERIVED BATHYMETRY (SDB)
# =============================================================================

class SatelliteBathymetry:
    """
    Estimacion de profundidad desde imagenes satelitales.

    Usa el algoritmo de Stumpf (2003):
    depth = m0 + m1 * ln(n * blue / green)

    Efectivo para aguas claras hasta ~25m.
    """

    __slots__ = ('config', '_calibrated')

    def __init__(self, config: Optional[SDBConfig] = None):
        self.config = config or SDBConfig()
        self._calibrated = False

    def estimate_depth(
        self,
        image: np.ndarray,
        water_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Estima profundidad para cada pixel de la imagen.

        Args:
            image: Imagen RGB (numpy array)
            water_mask: Mascara de agua (255=agua)

        Returns:
            Array 2D con profundidad en metros (NaN donde no aplica)
        """
        if not HAS_CV2:
            logger.error("OpenCV requerido para SDB")
            return np.full(image.shape[:2], np.nan, dtype=np.float32)

        # Extraer bandas Blue (indice 2) y Green (indice 1) de RGB
        # Nota: En imagen RGB, R=0, G=1, B=2
        blue = image[:, :, 2].astype(np.float32)
        green = image[:, :, 1].astype(np.float32)

        # Evitar division por cero
        green = np.where(green < 1, 1, green)
        blue = np.where(blue < 1, 1, blue)

        # Calcular ratio Blue/Green
        ratio = blue / green

        # Aplicar formula Stumpf
        # depth = m0 + m1 * ln(n * blue / green)
        with np.errstate(divide='ignore', invalid='ignore'):
            depth = self.config.m0 + self.config.m1 * np.log(self.config.n * ratio)

        # Aplicar limites
        depth = np.clip(depth, 0, self.config.max_depth)

        # Invertir signo (profundidad es negativa bajo agua)
        depth = -depth

        # Marcar pixeles invalidos
        invalid = (
            (ratio < self.config.min_ratio) |
            (ratio > self.config.max_ratio)
        )
        depth[invalid] = np.nan

        # Aplicar mascara de agua
        if water_mask is not None:
            depth[water_mask == 0] = np.nan

        return depth.astype(np.float32)

    def estimate_depth_at_point(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        window_size: int = 5
    ) -> Tuple[float, float]:
        """
        Estima profundidad en un punto especifico.

        Args:
            image: Imagen RGB
            x, y: Coordenadas del pixel
            window_size: Tamano de ventana para promediar

        Returns:
            (depth, confidence) tupla
        """
        depth_map = self.estimate_depth(image)

        # Extraer ventana
        half = window_size // 2
        y1 = max(0, y - half)
        y2 = min(depth_map.shape[0], y + half + 1)
        x1 = max(0, x - half)
        x2 = min(depth_map.shape[1], x + half + 1)

        window = depth_map[y1:y2, x1:x2]
        valid = window[~np.isnan(window)]

        if len(valid) == 0:
            return np.nan, 0.0

        depth = float(np.median(valid))
        confidence = len(valid) / window.size

        return depth, confidence

    def calibrate(
        self,
        image: np.ndarray,
        known_depths: List[Tuple[int, int, float]]
    ) -> None:
        """
        Calibra coeficientes usando puntos de profundidad conocida.

        Args:
            image: Imagen RGB
            known_depths: Lista de (x, y, depth) tuplas
        """
        if len(known_depths) < 3:
            logger.warning("Se necesitan al menos 3 puntos para calibrar")
            return

        blue = image[:, :, 2].astype(np.float32)
        green = image[:, :, 1].astype(np.float32)

        # Recolectar datos
        X = []  # ln(n * B/G)
        Y = []  # depth

        for x, y, depth in known_depths:
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                b, g = blue[y, x], green[y, x]
                if g > 0:
                    ratio = self.config.n * b / g
                    if ratio > 0:
                        X.append(math.log(ratio))
                        Y.append(abs(depth))  # Usar valor absoluto

        if len(X) < 3:
            return

        # Regresion lineal: Y = m0 + m1 * X
        X = np.array(X)
        Y = np.array(Y)

        # Resolver sistema
        A = np.vstack([np.ones(len(X)), X]).T
        m0, m1 = np.linalg.lstsq(A, Y, rcond=None)[0]

        # Actualizar config (crear nueva instancia porque es frozen)
        self.config = SDBConfig(
            m0=m0,
            m1=m1,
            n=self.config.n,
            max_depth=self.config.max_depth,
            min_ratio=self.config.min_ratio,
            max_ratio=self.config.max_ratio
        )
        self._calibrated = True

        logger.info(f"SDB calibrado: m0={m0:.3f}, m1={m1:.3f}")


# =============================================================================
# GEBCO BATHYMETRY
# =============================================================================

class GEBCOBathymetry:
    """
    Batimetria desde GEBCO (General Bathymetric Chart of the Oceans).

    GEBCO proporciona datos globales de alta resolucion (~450m).
    Requiere descargar el archivo NetCDF de GEBCO.
    """

    __slots__ = ('_data', '_lats', '_lons', '_interpolator', '_kdtree')

    def __init__(self, gebco_file: Optional[Union[str, Path]] = None):
        """
        Args:
            gebco_file: Ruta al archivo GEBCO NetCDF
        """
        self._data: Optional[np.ndarray] = None
        self._lats: Optional[np.ndarray] = None
        self._lons: Optional[np.ndarray] = None
        self._interpolator = None
        self._kdtree = None

        if gebco_file:
            self.load(gebco_file)

    def load(self, gebco_file: Union[str, Path]) -> bool:
        """
        Carga datos GEBCO desde archivo NetCDF.

        Args:
            gebco_file: Ruta al archivo GEBCO

        Returns:
            True si carga exitosa
        """
        if not HAS_NETCDF:
            logger.error("netCDF4 requerido: pip install netCDF4")
            return False

        path = Path(gebco_file)
        if not path.exists():
            logger.error(f"Archivo GEBCO no encontrado: {path}")
            return False

        try:
            with nc.Dataset(path, 'r') as ds:
                self._lats = ds.variables['lat'][:].astype(np.float32)
                self._lons = ds.variables['lon'][:].astype(np.float32)
                self._data = ds.variables['elevation'][:].astype(np.float32)

            logger.info(f"GEBCO cargado: {self._data.shape}")
            self._create_interpolator()
            return True

        except Exception as e:
            logger.error(f"Error cargando GEBCO: {e}")
            return False

    def load_subset(
        self,
        gebco_file: Union[str, Path],
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float
    ) -> bool:
        """
        Carga solo un subconjunto de GEBCO para ahorrar memoria.

        Args:
            gebco_file: Ruta al archivo GEBCO
            lat_min, lat_max, lon_min, lon_max: Limites del area
        """
        if not HAS_NETCDF:
            return False

        path = Path(gebco_file)
        if not path.exists():
            return False

        try:
            with nc.Dataset(path, 'r') as ds:
                lats = ds.variables['lat'][:]
                lons = ds.variables['lon'][:]

                # Encontrar indices
                lat_idx = np.where((lats >= lat_min) & (lats <= lat_max))[0]
                lon_idx = np.where((lons >= lon_min) & (lons <= lon_max))[0]

                if len(lat_idx) == 0 or len(lon_idx) == 0:
                    return False

                # Extraer subconjunto
                self._lats = lats[lat_idx].astype(np.float32)
                self._lons = lons[lon_idx].astype(np.float32)
                self._data = ds.variables['elevation'][
                    lat_idx[0]:lat_idx[-1]+1,
                    lon_idx[0]:lon_idx[-1]+1
                ].astype(np.float32)

            logger.info(f"GEBCO subset cargado: {self._data.shape}")
            self._create_interpolator()
            return True

        except Exception as e:
            logger.error(f"Error cargando GEBCO subset: {e}")
            return False

    def _create_interpolator(self) -> None:
        """Crea interpolador para consultas rapidas."""
        if not HAS_SCIPY or self._data is None:
            return

        self._interpolator = RegularGridInterpolator(
            (self._lats, self._lons),
            self._data,
            method='linear',
            bounds_error=False,
            fill_value=np.nan
        )

    def get_depth(self, lat: float, lon: float) -> Tuple[float, float]:
        """
        Obtiene profundidad en una ubicacion.

        Args:
            lat, lon: Coordenadas

        Returns:
            (depth, confidence) tupla
            depth es negativo bajo el agua, positivo sobre tierra
        """
        if self._interpolator is None:
            return np.nan, 0.0

        try:
            depth = float(self._interpolator([[lat, lon]])[0])
            confidence = 0.9 if not np.isnan(depth) else 0.0
            return depth, confidence
        except Exception:
            return np.nan, 0.0

    def get_depth_grid(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        resolution: float = 0.001  # ~100m
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Obtiene grilla de profundidad para un area.

        Args:
            lat_min, lat_max, lon_min, lon_max: Limites
            resolution: Resolucion en grados

        Returns:
            (depths, lats, lons) arrays
        """
        if self._interpolator is None:
            return np.array([]), np.array([]), np.array([])

        lats = np.arange(lat_min, lat_max, resolution)
        lons = np.arange(lon_min, lon_max, resolution)

        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
        points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])

        depths = self._interpolator(points).reshape(lat_grid.shape)

        return depths.astype(np.float32), lats.astype(np.float32), lons.astype(np.float32)


# =============================================================================
# FUSION DE BATIMETRIA
# =============================================================================

class BathymetryFusion:
    """
    Fusiona multiples fuentes de batimetria.

    - Usa SDB para aguas poco profundas (<25m) cuando hay imagen
    - Usa GEBCO para aguas profundas
    - Combina ambos en la zona de transicion
    """

    __slots__ = ('sdb', 'gebco', 'transition_depth')

    def __init__(
        self,
        gebco_file: Optional[Union[str, Path]] = None,
        transition_depth: float = 20.0
    ):
        """
        Args:
            gebco_file: Ruta a archivo GEBCO
            transition_depth: Profundidad donde combinar SDB y GEBCO
        """
        self.sdb = SatelliteBathymetry()
        self.gebco = GEBCOBathymetry(gebco_file)
        self.transition_depth = transition_depth

    def get_depth(
        self,
        lat: float,
        lon: float,
        image: Optional[np.ndarray] = None,
        pixel_x: Optional[int] = None,
        pixel_y: Optional[int] = None
    ) -> BathymetryResult:
        """
        Obtiene profundidad fusionando fuentes.

        Args:
            lat, lon: Coordenadas
            image: Imagen satelital opcional para SDB
            pixel_x, pixel_y: Posicion en imagen

        Returns:
            BathymetryResult con profundidad y metadatos
        """
        sdb_depth = None
        sdb_conf = 0.0
        gebco_depth = None
        gebco_conf = 0.0

        # Intentar SDB si hay imagen
        if image is not None and pixel_x is not None and pixel_y is not None:
            sdb_depth, sdb_conf = self.sdb.estimate_depth_at_point(
                image, pixel_x, pixel_y
            )

        # Obtener GEBCO
        gebco_depth, gebco_conf = self.gebco.get_depth(lat, lon)

        # Fusionar
        if not np.isnan(sdb_depth) and not np.isnan(gebco_depth):
            # Ambas fuentes disponibles - ponderar por profundidad
            abs_depth = abs(sdb_depth)

            if abs_depth < self.transition_depth:
                # Zona poco profunda - preferir SDB
                weight_sdb = 1.0 - (abs_depth / self.transition_depth) * 0.5
                weight_gebco = 1.0 - weight_sdb
            else:
                # Zona profunda - preferir GEBCO
                weight_sdb = 0.0
                weight_gebco = 1.0

            depth = sdb_depth * weight_sdb + gebco_depth * weight_gebco
            confidence = sdb_conf * weight_sdb + gebco_conf * weight_gebco
            source = 'fusion'

        elif not np.isnan(sdb_depth):
            depth = sdb_depth
            confidence = sdb_conf
            source = 'sdb'

        elif not np.isnan(gebco_depth):
            depth = gebco_depth
            confidence = gebco_conf
            source = 'gebco'

        else:
            depth = np.nan
            confidence = 0.0
            source = 'none'

        return BathymetryResult(
            depth=depth,
            source=source,
            confidence=confidence,
            sdb_depth=sdb_depth,
            gebco_depth=gebco_depth
        )

    def get_depth_grid(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        image: Optional[np.ndarray] = None,
        geo_transform: Optional[Dict] = None,
        resolution: float = 0.001
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Genera grilla de profundidad fusionada.

        Args:
            lat_min, lat_max, lon_min, lon_max: Limites
            image: Imagen satelital para SDB
            geo_transform: Transformacion geo de la imagen
            resolution: Resolucion en grados

        Returns:
            (depths, lats, lons) arrays
        """
        # Obtener grilla GEBCO
        gebco_depths, lats, lons = self.gebco.get_depth_grid(
            lat_min, lat_max, lon_min, lon_max, resolution
        )

        if image is None or geo_transform is None:
            return gebco_depths, lats, lons

        # Calcular SDB
        sdb_map = self.sdb.estimate_depth(image)

        # Crear grilla fusionada
        fused = np.copy(gebco_depths)

        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                # Convertir lat/lon a pixel
                px = int((lon - geo_transform['lon_min']) / geo_transform['pixel_lon'])
                py = int((geo_transform['lat_max'] - lat) / geo_transform['pixel_lat'])

                if 0 <= px < sdb_map.shape[1] and 0 <= py < sdb_map.shape[0]:
                    sdb_val = sdb_map[py, px]

                    if not np.isnan(sdb_val):
                        abs_depth = abs(sdb_val)
                        if abs_depth < self.transition_depth:
                            # Zona poco profunda - usar SDB
                            weight = 1.0 - (abs_depth / self.transition_depth) * 0.5
                            gebco_val = gebco_depths[i, j]
                            if not np.isnan(gebco_val):
                                fused[i, j] = sdb_val * weight + gebco_val * (1 - weight)
                            else:
                                fused[i, j] = sdb_val

        return fused, lats, lons


# =============================================================================
# FUNCIONES DE CONVENIENCIA
# =============================================================================

def estimate_depth_from_image(
    image: np.ndarray,
    water_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Estima profundidad desde imagen satelital.

    Args:
        image: Imagen RGB
        water_mask: Mascara de agua

    Returns:
        Array de profundidades (metros, negativo=bajo agua)
    """
    sdb = SatelliteBathymetry()
    return sdb.estimate_depth(image, water_mask)


def get_depth_zones(depths: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Clasifica profundidades en zonas.

    Args:
        depths: Array de profundidades

    Returns:
        Dict con mascaras por zona
    """
    return {
        'intertidal': (depths > -2) & (depths <= 0),      # 0-2m
        'shallow': (depths > -10) & (depths <= -2),       # 2-10m
        'moderate': (depths > -30) & (depths <= -10),     # 10-30m
        'deep': (depths > -100) & (depths <= -30),        # 30-100m
        'very_deep': depths <= -100                        # >100m
    }
