"""
Calculo de gradientes y deteccion de frentes termicos.
"""

import numpy as np
from typing import Tuple, Optional
from scipy import ndimage


class GradientCalculator:
    """Calcula gradientes espaciales y detecta frentes termicos."""

    def __init__(self, grid_resolution_km: float = 0.5):
        """
        Inicializa el calculador.

        Args:
            grid_resolution_km: Resolucion de la grilla en km
        """
        self.grid_resolution_km = grid_resolution_km

    def sobel_gradient(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula el gradiente usando filtro Sobel.

        Args:
            data: Array 2D con datos (ej: SST)

        Returns:
            Tupla de (magnitud, direccion) del gradiente
        """
        # Filtros Sobel
        sobel_x = ndimage.sobel(data, axis=1, mode="constant")
        sobel_y = ndimage.sobel(data, axis=0, mode="constant")

        # Magnitud del gradiente (en unidades/km)
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2) / self.grid_resolution_km

        # Direccion del gradiente (grados, 0=norte, 90=este)
        direction = np.degrees(np.arctan2(sobel_x, sobel_y))
        direction = (direction + 360) % 360  # Normalizar a 0-360

        return magnitude, direction

    def scharr_gradient(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula el gradiente usando filtro Scharr (mas preciso que Sobel).

        Args:
            data: Array 2D con datos

        Returns:
            Tupla de (magnitud, direccion) del gradiente
        """
        # Kernel Scharr
        scharr_x = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]) / 32.0
        scharr_y = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]]) / 32.0

        grad_x = ndimage.convolve(data, scharr_x, mode="constant")
        grad_y = ndimage.convolve(data, scharr_y, mode="constant")

        magnitude = np.sqrt(grad_x**2 + grad_y**2) / self.grid_resolution_km
        direction = np.degrees(np.arctan2(grad_x, grad_y))
        direction = (direction + 360) % 360

        return magnitude, direction

    def detect_thermal_fronts(
        self,
        sst: np.ndarray,
        threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detecta frentes termicos basados en gradiente de SST.

        Args:
            sst: Array 2D de temperatura superficial del mar
            threshold: Umbral de gradiente para detectar frente (°C/km)

        Returns:
            Tupla de (mascara_frentes, intensidad_frentes)
        """
        magnitude, _ = self.sobel_gradient(sst)

        # Mascara binaria de frentes
        front_mask = magnitude > threshold

        # Intensidad del frente (cuanto supera el umbral)
        front_intensity = np.where(front_mask, magnitude, 0)

        return front_mask, front_intensity

    def distance_to_front(
        self,
        front_mask: np.ndarray
    ) -> np.ndarray:
        """
        Calcula la distancia de cada punto al frente termico mas cercano.

        Args:
            front_mask: Mascara booleana de frentes

        Returns:
            Array con distancias en km
        """
        if not front_mask.any():
            # Si no hay frentes, retornar infinito
            return np.full(front_mask.shape, np.inf)

        # Distancia euclidiana al punto True mas cercano
        distance_pixels = ndimage.distance_transform_edt(~front_mask)

        # Convertir a km
        distance_km = distance_pixels * self.grid_resolution_km

        return distance_km

    def front_proximity_score(
        self,
        sst: np.ndarray,
        threshold: float = 0.5,
        max_distance_km: float = 5.0
    ) -> np.ndarray:
        """
        Calcula un score de proximidad a frentes termicos.

        Args:
            sst: Array 2D de SST
            threshold: Umbral para deteccion de frente
            max_distance_km: Distancia maxima para considerar

        Returns:
            Array con scores 0-100 (100 = en el frente)
        """
        front_mask, intensity = self.detect_thermal_fronts(sst, threshold)
        distance = self.distance_to_front(front_mask)

        # Score inversamente proporcional a distancia
        # En el frente = 100, a max_distance = 0
        score = np.clip(100 * (1 - distance / max_distance_km), 0, 100)

        # Bonus por estar directamente en un frente intenso
        max_intensity = intensity.max() if intensity.max() > 0 else 1
        intensity_bonus = (intensity / max_intensity) * 20
        score = np.clip(score + intensity_bonus, 0, 100)

        return score

    def calculate_upwelling_index(
        self,
        wind_u: np.ndarray,
        wind_v: np.ndarray,
        coast_angle: float = 135.0
    ) -> np.ndarray:
        """
        Calcula indice de surgencia basado en viento paralelo a la costa.

        La surgencia ocurre cuando el viento sopla paralelo a la costa
        con la costa a la izquierda (en hemisferio sur).

        Args:
            wind_u: Componente U del viento (oeste-este)
            wind_v: Componente V del viento (sur-norte)
            coast_angle: Angulo de la costa (grados desde norte)

        Returns:
            Indice de surgencia (positivo = favorable)
        """
        # Angulo del viento
        wind_dir = np.degrees(np.arctan2(wind_u, wind_v))
        wind_speed = np.sqrt(wind_u**2 + wind_v**2)

        # Componente paralela a la costa
        # En hemisferio sur, surgencia cuando viento va hacia el norte
        # con costa al este
        angle_diff = np.radians(wind_dir - coast_angle)
        parallel_component = wind_speed * np.sin(angle_diff)

        # Coeficiente de Ekman simplificado
        # Positivo = surgencia favorable
        ekman_coefficient = 0.1  # Factor de escala
        upwelling_index = parallel_component * ekman_coefficient

        return upwelling_index
