"""
Prediccion temporal de condiciones de pesca.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import PREDICTION_HORIZONS
from data.fetchers.astronomical import AstronomicalCalculator
from data.fetchers.openmeteo import OpenMeteoFetcher
from processing.features import FeatureSet, FeatureEngineer
from .scoring import FishingScorer, ScoringResult


@dataclass
class PredictionResult:
    """Resultado de prediccion para un horizonte temporal."""
    horizon_hours: int
    timestamp: datetime
    scoring_result: ScoringResult
    confidence: float  # 0-1, menor confianza para predicciones mas lejanas


class TemporalPredictor:
    """Genera predicciones para diferentes horizontes temporales."""

    def __init__(
        self,
        feature_engineer: Optional[FeatureEngineer] = None,
        scorer: Optional[FishingScorer] = None,
        astronomical_calc: Optional[AstronomicalCalculator] = None,
        meteo_fetcher: Optional[OpenMeteoFetcher] = None
    ):
        """
        Inicializa el predictor.

        Args:
            feature_engineer: Ingeniero de features
            scorer: Modelo de scoring
            astronomical_calc: Calculador astronomico
            meteo_fetcher: Fetcher de datos meteorologicos
        """
        self.feature_engineer = feature_engineer or FeatureEngineer()
        self.scorer = scorer or FishingScorer()
        self.astronomical_calc = astronomical_calc or AstronomicalCalculator()
        self.meteo_fetcher = meteo_fetcher or OpenMeteoFetcher()

    def predict(
        self,
        current_features: FeatureSet,
        base_time: Optional[datetime] = None,
        horizons: Optional[List[int]] = None,
        lat: float = -17.85,
        lon: float = -71.15
    ) -> Dict[int, PredictionResult]:
        """
        Genera predicciones para multiples horizontes.

        Args:
            current_features: Features actuales
            base_time: Tiempo base para predicciones
            horizons: Lista de horizontes en horas
            lat: Latitud de referencia
            lon: Longitud de referencia

        Returns:
            Dict de {horizon_hours: PredictionResult}
        """
        if base_time is None:
            base_time = datetime.now()
        if horizons is None:
            horizons = PREDICTION_HORIZONS

        predictions = {}

        for hours in horizons:
            future_time = base_time + timedelta(hours=hours)

            # Obtener datos astronomicos para el tiempo futuro
            astro_data = self.astronomical_calc.get_all_astronomical_data(
                future_time, lat, lon
            )

            # Ajustar features para el tiempo futuro
            future_features = self._project_features(
                current_features,
                hours,
                lat,
                lon
            )

            # Calcular score
            scoring_result = self.scorer.score(future_features, astro_data)

            # Calcular confianza (decrece con el tiempo)
            confidence = self._calculate_confidence(hours)

            predictions[hours] = PredictionResult(
                horizon_hours=hours,
                timestamp=future_time,
                scoring_result=scoring_result,
                confidence=confidence
            )

        return predictions

    def _project_features(
        self,
        current: FeatureSet,
        hours_ahead: int,
        lat: float,
        lon: float
    ) -> FeatureSet:
        """
        Proyecta features al futuro.

        La SST cambia lentamente, el clima cambia segun pronostico.
        """
        # SST: persistencia con relajacion leve hacia climatologia
        decay_factor = 0.99 ** (hours_ahead / 24)  # ~1% por dia
        climatology_sst = 17.0  # SST promedio de la zona

        projected_sst = current.sst * decay_factor + climatology_sst * (1 - decay_factor)

        # Clorofila: persistencia similar
        climatology_chl = 1.5
        projected_chl = current.chlorophyll * decay_factor + climatology_chl * (1 - decay_factor)

        # Obtener pronostico meteorologico
        try:
            weather_df = self.meteo_fetcher.fetch_weather_data(lat, lon, forecast_days=4)
            marine_df = self.meteo_fetcher.fetch_marine_data(lat, lon, forecast_days=4)

            target_time = datetime.now() + timedelta(hours=hours_ahead)

            # Encontrar datos mas cercanos al tiempo objetivo
            if not weather_df.empty:
                weather_df["time_diff"] = abs(weather_df["time"] - pd.Timestamp(target_time))
                closest_weather = weather_df.loc[weather_df["time_diff"].idxmin()]
                wind_speed = closest_weather.get("wind_speed", current.wind_speed[0, 0])
                wind_direction = closest_weather.get("wind_direction", current.wind_direction[0, 0])
            else:
                wind_speed = current.wind_speed[0, 0]
                wind_direction = current.wind_direction[0, 0]

            if not marine_df.empty:
                marine_df["time_diff"] = abs(marine_df["time"] - pd.Timestamp(target_time))
                closest_marine = marine_df.loc[marine_df["time_diff"].idxmin()]
                wave_height = closest_marine.get("wave_height", current.wave_height[0, 0])
                wave_period = closest_marine.get("wave_period", current.wave_period[0, 0])
            else:
                wave_height = current.wave_height[0, 0]
                wave_period = current.wave_period[0, 0]

        except Exception:
            # Usar valores actuales si falla el fetch
            wind_speed = current.wind_speed[0, 0]
            wind_direction = current.wind_direction[0, 0]
            wave_height = current.wave_height[0, 0]
            wave_period = current.wave_period[0, 0]

        # Recalcular features derivadas con SST proyectada
        from processing.gradients import GradientCalculator
        grad_calc = GradientCalculator()

        gradient_mag, gradient_dir = grad_calc.sobel_gradient(projected_sst)
        front_mask, _ = grad_calc.detect_thermal_fronts(projected_sst)
        front_score = grad_calc.front_proximity_score(projected_sst)

        # Recalcular clorofila features
        chl_log = np.log10(np.clip(projected_chl, 0.01, None))
        chl_score = np.clip((projected_chl - 0.5) / (2.0 - 0.5) * 100, 0, 100)

        # Safety score
        wave_grid = np.full(current.sst.shape, wave_height)
        wind_grid = np.full(current.sst.shape, wind_speed)
        safety = self.feature_engineer.calculate_safety_score(wave_grid, wind_grid)

        # Upwelling
        wind_rad = np.radians(wind_direction)
        wind_u = wind_speed * np.sin(wind_rad)
        wind_v = wind_speed * np.cos(wind_rad)
        wind_u_grid = np.full(current.sst.shape, wind_u)
        wind_v_grid = np.full(current.sst.shape, wind_v)
        upwelling = grad_calc.calculate_upwelling_index(wind_u_grid, wind_v_grid)

        return FeatureSet(
            sst=projected_sst,
            sst_gradient=gradient_mag,
            sst_gradient_direction=gradient_dir,
            is_thermal_front=front_mask.astype(float),
            front_proximity_score=front_score,
            chlorophyll=projected_chl,
            chlorophyll_log=chl_log,
            chlorophyll_score=chl_score,
            wind_speed=wind_grid,
            wind_direction=np.full(current.sst.shape, wind_direction),
            wave_height=wave_grid,
            wave_period=np.full(current.sst.shape, wave_period),
            safety_score=safety,
            upwelling_index=upwelling,
            distance_to_coast=current.distance_to_coast
        )

    def _calculate_confidence(self, hours_ahead: int) -> float:
        """
        Calcula confianza de la prediccion.

        Decrece exponencialmente con el tiempo.
        """
        # 100% en t=0, ~85% en 24h, ~70% en 48h, ~60% en 72h
        decay_rate = 0.007
        confidence = np.exp(-decay_rate * hours_ahead)
        return float(confidence)

    def get_best_time(
        self,
        predictions: Dict[int, PredictionResult],
        lat: float,
        lon: float,
        df_current: pd.DataFrame
    ) -> Dict:
        """
        Encuentra el mejor momento para pescar.

        Args:
            predictions: Dict de predicciones
            lat: Latitud del spot
            lon: Longitud del spot
            df_current: DataFrame actual con scores

        Returns:
            Dict con recomendacion del mejor momento
        """
        best_horizon = 0
        best_score = 0
        scores_by_horizon = {}

        for hours, pred in predictions.items():
            # Obtener score para la ubicacion especifica
            df = self.scorer.score_to_dataframe(
                pred.scoring_result,
                self.feature_engineer.grid_processor.grid.lat_grid,
                self.feature_engineer.grid_processor.grid.lon_grid
            )

            df["distance"] = np.sqrt(
                (df["latitude"] - lat)**2 + (df["longitude"] - lon)**2
            )
            closest = df.loc[df["distance"].idxmin()]
            score = closest["score"]

            scores_by_horizon[hours] = {
                "score": score,
                "confidence": pred.confidence,
                "timestamp": pred.timestamp.isoformat()
            }

            # Ponderar por confianza
            weighted_score = score * pred.confidence
            if weighted_score > best_score:
                best_score = weighted_score
                best_horizon = hours

        return {
            "best_horizon_hours": best_horizon,
            "best_time": predictions[best_horizon].timestamp.isoformat(),
            "expected_score": scores_by_horizon[best_horizon]["score"],
            "confidence": predictions[best_horizon].confidence,
            "all_predictions": scores_by_horizon
        }
