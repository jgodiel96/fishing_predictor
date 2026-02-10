"""
Substrate Classifier using Computer Vision.

Classifies seafloor substrate from satellite imagery:
- Rocky zones (dark, irregular texture)
- Sandy zones (light, smooth texture)
- Mixed zones (combination)

Uses texture analysis (GLCM) and color features.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Tuple, Dict
import math

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

logger = logging.getLogger(__name__)


class SubstrateType(Enum):
    """Tipos de sustrato marino."""
    ROCK = "rock"
    SAND = "sand"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass
class SubstrateResult:
    """Resultado de clasificacion de sustrato."""
    substrate_type: SubstrateType
    confidence: float  # 0-1
    rock_probability: float
    sand_probability: float
    texture_variance: float
    mean_brightness: float

    def __post_init__(self):
        # Ensure float32 for memory efficiency
        self.confidence = np.float32(self.confidence)
        self.rock_probability = np.float32(self.rock_probability)
        self.sand_probability = np.float32(self.sand_probability)


# =============================================================================
# PARAMETROS DE CLASIFICACION
# =============================================================================

# Umbrales de brillo (0-255) para clasificacion inicial
BRIGHTNESS_THRESHOLDS = {
    'sand_min': 140,      # Arena es clara
    'rock_max': 100,      # Roca es oscura
}

# Umbrales de textura (varianza local)
TEXTURE_THRESHOLDS = {
    'smooth_max': 300,    # Arena tiene textura suave
    'rough_min': 800,     # Roca tiene textura rugosa
}

# Rangos HSV para deteccion de agua (para mascara)
WATER_HSV = {
    'lower': np.array([90, 20, 20]),
    'upper': np.array([130, 255, 200])
}

# Rangos HSV para arena
SAND_HSV = {
    'lower': np.array([15, 10, 120]),
    'upper': np.array([35, 80, 255])
}


class SubstrateClassifier:
    """
    Clasificador de sustrato marino.

    Analiza imagenes satelitales para determinar tipo de fondo:
    - Rocoso: Oscuro, textura irregular, alto contraste
    - Arenoso: Claro, textura suave, bajo contraste
    - Mixto: Combinacion de ambos

    Attributes:
        window_size: Tamano de ventana para analisis local
        _cache: Cache de resultados
    """

    __slots__ = ('window_size', '_cache')

    def __init__(self, window_size: int = 32):
        """
        Args:
            window_size: Tamano de ventana para analisis de textura
        """
        self.window_size = window_size
        self._cache: Dict[str, SubstrateResult] = {}

    def classify_region(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> SubstrateResult:
        """
        Clasifica el sustrato en una region de imagen.

        Args:
            image: Imagen RGB (numpy array)
            mask: Mascara opcional (255=analizar, 0=ignorar)

        Returns:
            SubstrateResult con tipo y probabilidades
        """
        if not HAS_CV2:
            logger.error("OpenCV no instalado")
            return SubstrateResult(
                substrate_type=SubstrateType.UNKNOWN,
                confidence=0.0,
                rock_probability=0.5,
                sand_probability=0.5,
                texture_variance=0.0,
                mean_brightness=0.0
            )

        # Convertir a escala de grises para analisis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Aplicar mascara si existe
        if mask is not None:
            gray = cv2.bitwise_and(gray, gray, mask=mask)

        # 1. Analisis de brillo
        mean_brightness = self._calculate_brightness(gray, mask)

        # 2. Analisis de textura
        texture_variance = self._calculate_texture_variance(gray, mask)

        # 3. Analisis de color HSV
        color_features = self._analyze_color(image, mask)

        # 4. Combinar features para clasificacion
        rock_prob, sand_prob = self._calculate_probabilities(
            mean_brightness, texture_variance, color_features
        )

        # 5. Determinar tipo
        substrate_type, confidence = self._determine_type(rock_prob, sand_prob)

        return SubstrateResult(
            substrate_type=substrate_type,
            confidence=confidence,
            rock_probability=rock_prob,
            sand_probability=sand_prob,
            texture_variance=texture_variance,
            mean_brightness=mean_brightness
        )

    def classify_grid(
        self,
        image: np.ndarray,
        grid_size: int = 64,
        water_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Clasifica sustrato en una grilla sobre la imagen.

        Args:
            image: Imagen RGB
            grid_size: Tamano de celda en pixeles
            water_mask: Mascara de agua (255=agua, 0=tierra)

        Returns:
            Array 2D con tipo de sustrato por celda (0=rock, 1=sand, 2=mixed)
        """
        if not HAS_CV2:
            return np.zeros((1, 1), dtype=np.uint8)

        h, w = image.shape[:2]
        rows = h // grid_size
        cols = w // grid_size

        result = np.zeros((rows, cols), dtype=np.uint8)

        for r in range(rows):
            for c in range(cols):
                y1, y2 = r * grid_size, (r + 1) * grid_size
                x1, x2 = c * grid_size, (c + 1) * grid_size

                cell = image[y1:y2, x1:x2]

                # Mascara de celda
                cell_mask = None
                if water_mask is not None:
                    cell_water = water_mask[y1:y2, x1:x2]
                    # Invertir: analizar donde NO hay agua
                    cell_mask = cv2.bitwise_not(cell_water)

                    # Si >80% es agua, marcar como unknown
                    water_ratio = np.sum(cell_water > 0) / (grid_size * grid_size)
                    if water_ratio > 0.8:
                        result[r, c] = 3  # Unknown/water
                        continue

                classification = self.classify_region(cell, cell_mask)

                # Mapear a valores
                type_map = {
                    SubstrateType.ROCK: 0,
                    SubstrateType.SAND: 1,
                    SubstrateType.MIXED: 2,
                    SubstrateType.UNKNOWN: 3
                }
                result[r, c] = type_map[classification.substrate_type]

        return result

    def _calculate_brightness(
        self,
        gray: np.ndarray,
        mask: Optional[np.ndarray]
    ) -> float:
        """Calcula brillo promedio de la region."""
        if mask is not None:
            pixels = gray[mask > 0]
            if len(pixels) == 0:
                return 128.0
            return float(np.mean(pixels))
        return float(np.mean(gray))

    def _calculate_texture_variance(
        self,
        gray: np.ndarray,
        mask: Optional[np.ndarray]
    ) -> float:
        """
        Calcula varianza de textura usando Laplaciano.

        Mayor varianza = textura mas rugosa (rocas)
        Menor varianza = textura mas suave (arena)
        """
        # Aplicar Laplaciano para detectar bordes/textura
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        if mask is not None:
            pixels = laplacian[mask > 0]
            if len(pixels) == 0:
                return 0.0
            return float(np.var(pixels))

        return float(np.var(laplacian))

    def _analyze_color(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray]
    ) -> Dict[str, float]:
        """Analiza caracteristicas de color HSV."""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        features = {}

        # Deteccion de arena por color
        sand_mask = cv2.inRange(hsv, SAND_HSV['lower'], SAND_HSV['upper'])
        if mask is not None:
            sand_mask = cv2.bitwise_and(sand_mask, mask)

        total_pixels = np.sum(mask > 0) if mask is not None else image.shape[0] * image.shape[1]
        if total_pixels > 0:
            features['sand_color_ratio'] = np.sum(sand_mask > 0) / total_pixels
        else:
            features['sand_color_ratio'] = 0.0

        # Saturacion promedio (rocas suelen tener baja saturacion)
        if mask is not None:
            sat_pixels = hsv[:, :, 1][mask > 0]
            features['mean_saturation'] = float(np.mean(sat_pixels)) if len(sat_pixels) > 0 else 0.0
        else:
            features['mean_saturation'] = float(np.mean(hsv[:, :, 1]))

        return features

    def _calculate_probabilities(
        self,
        brightness: float,
        texture: float,
        color_features: Dict[str, float]
    ) -> Tuple[float, float]:
        """
        Calcula probabilidades de roca vs arena.

        Combina multiples features con pesos.
        """
        # Probabilidad base por brillo
        if brightness < BRIGHTNESS_THRESHOLDS['rock_max']:
            brightness_rock = 0.8
        elif brightness > BRIGHTNESS_THRESHOLDS['sand_min']:
            brightness_rock = 0.2
        else:
            # Interpolacion lineal
            range_size = BRIGHTNESS_THRESHOLDS['sand_min'] - BRIGHTNESS_THRESHOLDS['rock_max']
            brightness_rock = 1.0 - (brightness - BRIGHTNESS_THRESHOLDS['rock_max']) / range_size
            brightness_rock = max(0.2, min(0.8, brightness_rock))

        # Probabilidad por textura
        if texture > TEXTURE_THRESHOLDS['rough_min']:
            texture_rock = 0.8
        elif texture < TEXTURE_THRESHOLDS['smooth_max']:
            texture_rock = 0.2
        else:
            range_size = TEXTURE_THRESHOLDS['rough_min'] - TEXTURE_THRESHOLDS['smooth_max']
            texture_rock = (texture - TEXTURE_THRESHOLDS['smooth_max']) / range_size
            texture_rock = max(0.2, min(0.8, texture_rock))

        # Probabilidad por color
        sand_color = color_features.get('sand_color_ratio', 0.0)
        color_rock = 1.0 - sand_color

        # Combinar con pesos
        weights = {
            'brightness': 0.35,
            'texture': 0.40,
            'color': 0.25
        }

        rock_prob = (
            weights['brightness'] * brightness_rock +
            weights['texture'] * texture_rock +
            weights['color'] * color_rock
        )

        sand_prob = 1.0 - rock_prob

        return rock_prob, sand_prob

    def _determine_type(
        self,
        rock_prob: float,
        sand_prob: float
    ) -> Tuple[SubstrateType, float]:
        """Determina tipo de sustrato y confianza."""
        if rock_prob > 0.65:
            return SubstrateType.ROCK, rock_prob
        elif sand_prob > 0.65:
            return SubstrateType.SAND, sand_prob
        else:
            # Mixto cuando no hay dominancia clara
            confidence = 1.0 - abs(rock_prob - sand_prob)
            return SubstrateType.MIXED, confidence

    def get_species_affinity(self, substrate: SubstrateType) -> Dict[str, float]:
        """
        Retorna afinidad de especies por tipo de sustrato.

        Returns:
            Dict con species -> affinity_score (0-1)
        """
        affinities = {
            SubstrateType.ROCK: {
                'corvina': 0.9,      # Corvina ama las rocas
                'lenguado': 0.3,     # Prefiere arena
                'cabrilla': 0.95,    # Muy rocoso
                'chita': 0.85,       # Rocoso
                'pejerrey': 0.5,     # Indiferente
                'lorna': 0.6,        # Algo rocoso
                'pintadilla': 0.9,   # Rocoso
            },
            SubstrateType.SAND: {
                'corvina': 0.5,
                'lenguado': 0.95,    # Arena es su habitat
                'cabrilla': 0.2,
                'chita': 0.3,
                'pejerrey': 0.7,
                'lorna': 0.8,        # Arena/fango
                'pintadilla': 0.3,
            },
            SubstrateType.MIXED: {
                'corvina': 0.75,
                'lenguado': 0.6,
                'cabrilla': 0.5,
                'chita': 0.6,
                'pejerrey': 0.6,
                'lorna': 0.7,
                'pintadilla': 0.6,
            },
            SubstrateType.UNKNOWN: {
                species: 0.5 for species in
                ['corvina', 'lenguado', 'cabrilla', 'chita', 'pejerrey', 'lorna', 'pintadilla']
            }
        }

        return affinities.get(substrate, affinities[SubstrateType.UNKNOWN])


# =============================================================================
# FUNCION DE CONVENIENCIA
# =============================================================================

def classify_substrate_from_image(
    image: np.ndarray,
    water_mask: Optional[np.ndarray] = None
) -> SubstrateResult:
    """
    Funcion de conveniencia para clasificar sustrato.

    Args:
        image: Imagen RGB
        water_mask: Mascara de agua opcional

    Returns:
        SubstrateResult
    """
    classifier = SubstrateClassifier()

    # Invertir mascara de agua para analizar tierra
    land_mask = None
    if water_mask is not None:
        land_mask = cv2.bitwise_not(water_mask)

    return classifier.classify_region(image, land_mask)
