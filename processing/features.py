"""
Ingenieria de features para el modelo de prediccion de pesca.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import THRESHOLDS

from .gradients import GradientCalculator
from .preprocessing import GridProcessor


@dataclass
class FeatureSet:
    """Conjunto de features calculadas."""
    # SST
    sst: np.ndarray
    sst_gradient: np.ndarray
    sst_gradient_direction: np.ndarray
    is_thermal_front: np.ndarray
    front_proximity_score: np.ndarray

    # Clorofila
    chlorophyll: np.ndarray
    chlorophyll_log: np.ndarray
    chlorophyll_score: np.ndarray

    # Meteorologicas
    wind_speed: np.ndarray
    wind_direction: np.ndarray
    wave_height: np.ndarray
    wave_period: np.ndarray

    # Derivadas
    safety_score: np.ndarray
    upwelling_index: np.ndarray

    # Geograficas
    distance_to_coast: np.ndarray


class FeatureEngineer:
    """Calcula y transforma features para el modelo de scoring."""

    def __init__(
        self,
        grid_processor: Optional[GridProcessor] = None,
        gradient_calculator: Optional[GradientCalculator] = None
    ):
        """
        Inicializa el ingeniero de features.

        Args:
            grid_processor: Procesador de grilla
            gradient_calculator: Calculador de gradientes
        """
        self.grid_processor = grid_processor or GridProcessor()
        self.gradient_calc = gradient_calculator or GradientCalculator()

    def calculate_sst_features(
        self,
        sst: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Calcula features basadas en SST.

        Args:
            sst: Array 2D de temperatura superficial

        Returns:
            Dict con features de SST
        """
        # Gradiente
        gradient_mag, gradient_dir = self.gradient_calc.sobel_gradient(sst)

        # Frentes termicos
        front_mask, front_intensity = self.gradient_calc.detect_thermal_fronts(
            sst,
            threshold=THRESHOLDS.SST_GRADIENT_THRESHOLD
        )

        # Score de proximidad a frentes
        front_score = self.gradient_calc.front_proximity_score(
            sst,
            threshold=THRESHOLDS.SST_GRADIENT_THRESHOLD
        )

        return {
            "sst": sst,
            "sst_gradient": gradient_mag,
            "sst_gradient_direction": gradient_dir,
            "is_thermal_front": front_mask.astype(float),
            "front_proximity_score": front_score
        }

    def calculate_chlorophyll_features(
        self,
        chlorophyll: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Calcula features basadas en clorofila-a.

        Args:
            chlorophyll: Array 2D de concentracion de clorofila

        Returns:
            Dict con features de clorofila
        """
        # Log para normalizar distribucion
        chl_log = np.log10(np.clip(chlorophyll, 0.01, None))

        # Score basado en umbrales
        chl_score = self._calculate_chlorophyll_score(chlorophyll)

        return {
            "chlorophyll": chlorophyll,
            "chlorophyll_log": chl_log,
            "chlorophyll_score": chl_score
        }

    def _calculate_chlorophyll_score(
        self,
        chlorophyll: np.ndarray
    ) -> np.ndarray:
        """
        Calcula score de clorofila (0-100).

        Alta clorofila = alta productividad = mejor pesca
        """
        # Normalizar entre umbrales bajo y alto
        low = THRESHOLDS.CHL_LOW_THRESHOLD
        high = THRESHOLDS.CHL_HIGH_THRESHOLD

        # Score lineal entre umbrales
        score = (chlorophyll - low) / (high - low) * 100
        score = np.clip(score, 0, 100)

        return score

    def calculate_safety_score(
        self,
        wave_height: np.ndarray,
        wind_speed: np.ndarray
    ) -> np.ndarray:
        """
        Calcula score de seguridad (0-100).

        100 = condiciones perfectas
        0 = condiciones peligrosas
        """
        # Normalizar olas (0-2m rango seguro)
        wave_factor = np.clip(wave_height / THRESHOLDS.WAVE_SAFETY_THRESHOLD, 0, 1)

        # Normalizar viento (0-25 km/h rango seguro)
        wind_factor = np.clip(wind_speed / THRESHOLDS.WIND_SAFETY_THRESHOLD, 0, 1)

        # Score combinado
        safety = (1 - (wave_factor + wind_factor) / 2) * 100
        safety = np.clip(safety, 0, 100)

        return safety

    def calculate_all_features(
        self,
        sst_df: pd.DataFrame,
        chl_df: pd.DataFrame,
        weather_data: Dict[str, float]
    ) -> FeatureSet:
        """
        Calcula todas las features.

        Args:
            sst_df: DataFrame con datos de SST
            chl_df: DataFrame con datos de clorofila
            weather_data: Dict con datos meteorologicos

        Returns:
            FeatureSet con todas las features
        """
        # Interpolar SST a grilla
        if not sst_df.empty:
            sst_grid = self.grid_processor.interpolate_to_grid(sst_df, "sst")
            sst_grid = self.grid_processor.fill_gaps(sst_grid)
        else:
            sst_grid = np.full(
                self.grid_processor.grid.lat_grid.shape,
                17.0  # SST promedio para la zona
            )

        # Interpolar clorofila a grilla
        if not chl_df.empty:
            chl_grid = self.grid_processor.interpolate_to_grid(chl_df, "chlorophyll")
            chl_grid = self.grid_processor.fill_gaps(chl_grid)
        else:
            chl_grid = np.full(
                self.grid_processor.grid.lat_grid.shape,
                1.5  # Clorofila promedio
            )

        # Features de SST
        sst_features = self.calculate_sst_features(sst_grid)

        # Features de clorofila
        chl_features = self.calculate_chlorophyll_features(chl_grid)

        # Datos meteorologicos (broadcast a toda la grilla)
        grid_shape = self.grid_processor.grid.lat_grid.shape
        wind_speed = np.full(grid_shape, weather_data.get("wind_speed", 10.0))
        wind_direction = np.full(grid_shape, weather_data.get("wind_direction", 180.0))
        wave_height = np.full(grid_shape, weather_data.get("wave_height", 1.0))
        wave_period = np.full(grid_shape, weather_data.get("wave_period", 10.0))

        # Safety score
        safety = self.calculate_safety_score(wave_height, wind_speed)

        # Upwelling index
        # Convertir direccion a componentes U, V
        wind_rad = np.radians(wind_direction)
        wind_u = wind_speed * np.sin(wind_rad)
        wind_v = wind_speed * np.cos(wind_rad)
        upwelling = self.gradient_calc.calculate_upwelling_index(wind_u, wind_v)

        # Distancia a costa
        distance_to_coast = self.grid_processor.calculate_distance_to_coast()

        return FeatureSet(
            sst=sst_features["sst"],
            sst_gradient=sst_features["sst_gradient"],
            sst_gradient_direction=sst_features["sst_gradient_direction"],
            is_thermal_front=sst_features["is_thermal_front"],
            front_proximity_score=sst_features["front_proximity_score"],
            chlorophyll=chl_features["chlorophyll"],
            chlorophyll_log=chl_features["chlorophyll_log"],
            chlorophyll_score=chl_features["chlorophyll_score"],
            wind_speed=wind_speed,
            wind_direction=wind_direction,
            wave_height=wave_height,
            wave_period=wave_period,
            safety_score=safety,
            upwelling_index=upwelling,
            distance_to_coast=distance_to_coast
        )

    def features_to_dataframe(
        self,
        features: FeatureSet
    ) -> pd.DataFrame:
        """
        Convierte FeatureSet a DataFrame.

        Args:
            features: Conjunto de features

        Returns:
            DataFrame con todas las features
        """
        data_dict = {
            "sst": features.sst,
            "sst_gradient": features.sst_gradient,
            "sst_gradient_direction": features.sst_gradient_direction,
            "is_thermal_front": features.is_thermal_front,
            "front_proximity_score": features.front_proximity_score,
            "chlorophyll": features.chlorophyll,
            "chlorophyll_log": features.chlorophyll_log,
            "chlorophyll_score": features.chlorophyll_score,
            "wind_speed": features.wind_speed,
            "wind_direction": features.wind_direction,
            "wave_height": features.wave_height,
            "wave_period": features.wave_period,
            "safety_score": features.safety_score,
            "upwelling_index": features.upwelling_index,
            "distance_to_coast": features.distance_to_coast
        }

        return self.grid_processor.grid_to_dataframe(data_dict)
